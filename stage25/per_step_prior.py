"""Stage 25: sharpen the differences between time steps, instead of flattening them.

Run from the repo root:
    python3 stage25/per_step_prior.py                       # measure on saved predictions
    python3 stage25/per_step_prior.py --apply FILE.csv --gamma 0.5

WHY THIS IS NEW
---------------
AUC is computed over all 15,329 test rows pooled together, so the ranking BETWEEN time
steps counts just as much as the ranking within them. Two facts about this dataset make
that worth attacking directly:

  * The illicit rate per step swings from 0.4% to 36.0% in training (train_by_time_step.csv).
    Steps are genuinely not alike.
  * Stage 7 tested per-step normalisation, which equalises the steps, and it was the most
    harmful post-processing tried: -0.033 mean AUC on the long splits, 0% of cases
    improved. That is strong evidence that the step-level differences in the model's
    scores carry real information.

Removing those differences hurts a lot. The opposite operation - amplifying them - has
never been tested. If the model under-states how much the steps differ, pushing the step
means apart should raise the pooled AUC without changing any within-step ordering.

The adjustment, in log-odds space:

    z'(i) = z(i) + gamma * (mean_z[step of i] - mean_z[all rows])

gamma = 0 leaves the scores untouched, gamma > 0 pushes the steps apart, gamma < 0
flattens them towards each other (the Stage 7 direction, included as a control: it should
come out clearly negative, which also confirms the harness is measuring the right thing).

Nothing here uses labels, so it is valid as post-processing on the test predictions.

A second variant uses step SIZE as a prior. In training, the per-step illicit rate has a
Spearman correlation of -0.37 with the number of transactions in the step: smaller steps
tend to be dirtier. The coefficient is fitted on the training steps that precede each
validation block, never on the block itself.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.validation import PRED_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sharpen(pred, steps, gamma):
    """Push each step's mean log-odds further from (gamma>0) or closer to (gamma<0) the overall mean."""
    z = logit(pred)
    s = pd.Series(z).groupby(pd.Series(steps).values).transform("mean").to_numpy()
    return 1.0 / (1.0 + np.exp(-(z + gamma * (s - z.mean()))))


def size_prior(pred, steps, sizes, coef):
    """Shift each step by coef * (log step size - mean log step size)."""
    z = logit(pred)
    ls = np.log(pd.Series(steps).map(sizes).to_numpy().astype("float64"))
    return 1.0 / (1.0 + np.exp(-(z + coef * (ls - ls.mean()))))


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--contains", default="ppr", help="which saved prediction files to measure on")
p.add_argument("--max-files", type=int, default=8)
p.add_argument("--gammas", nargs="*", type=float, default=[-0.5, -0.25, 0.25, 0.5, 1.0, 2.0])
p.add_argument("--apply", default=None, help="submission CSV to adjust and rewrite")
p.add_argument("--gamma", type=float, default=0.5, help="gamma for --apply")
args = p.parse_args()

train, test, _ = load_data(args.data_dir)

if args.apply:
    path = Path(args.apply)
    if not path.exists():
        path = ROOT / "submissions" / args.apply
    sub = pd.read_csv(path)
    if list(sub.columns) != ["index", "target"]:
        sys.exit(f"{path}: columns {list(sub.columns)} are not ['index', 'target']")
    steps = test["time_step"].to_numpy()
    adjusted = sharpen(sub["target"].to_numpy(), steps, args.gamma)
    out = pd.DataFrame({"index": sub["index"], "target": adjusted})
    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    assert out["index"].equals(sample["index"]) and np.isfinite(out["target"]).all()
    rho = pd.Series(sub["target"]).corr(pd.Series(adjusted), method="spearman")
    new = ROOT / "submissions" / f"{path.stem}_gamma{args.gamma:g}.csv"
    out.to_csv(new, index=False)
    print(f"Spearman with the original ranking: {rho:.5f}")
    print("Per-step mean prediction, before and after:")
    comp = pd.DataFrame({"step": steps, "before": sub["target"].to_numpy(), "after": adjusted})
    print(comp.groupby("step").mean().round(4).to_string())
    print(f"\nWrote {new}")
    sys.exit(0)

files = sorted(PRED_DIR.glob("*.csv.gz"))
if args.contains:
    files = [f for f in files if args.contains in f.name]
files = files[-args.max_files:]
if not files:
    sys.exit(f"No saved predictions matching '{args.contains}' in {PRED_DIR}")
print(f"Measuring on {len(files)} saved prediction files")

labelled = train[train["label"].notna()]
step_size = train.groupby("time_step").size()          # all rows per step, not just labelled

rows = []
for f in files:
    df = pd.read_csv(f)
    for split, g in df.groupby("split"):
        if g["label"].nunique() < 2 or g["time_step"].nunique() < 3:
            continue
        y = g["label"].to_numpy()
        steps = g["time_step"].to_numpy()
        base = roc_auc_score(y, g["pred"])
        for gamma in args.gammas:
            auc = roc_auc_score(y, sharpen(g["pred"].to_numpy(), steps, gamma))
            rows.append({"run": f.name.split(".csv")[0], "split": split, "method": "sharpen",
                         "setting": gamma, "n_steps": g["time_step"].nunique(),
                         "delta": auc - base})
        # size prior: coefficient fitted only on steps strictly before this block
        first = int(steps.min())
        hist = labelled[labelled["time_step"] < first]
        if hist["time_step"].nunique() >= 8:
            rate = hist.groupby("time_step")["label"].mean()
            ls = np.log(step_size.loc[rate.index].to_numpy())
            lo = np.log(np.clip(rate.to_numpy(), 1e-3, 1 - 1e-3) / (1 - np.clip(rate.to_numpy(), 1e-3, 1 - 1e-3)))
            coef = float(np.polyfit(ls, lo, 1)[0])
            auc = roc_auc_score(y, size_prior(g["pred"].to_numpy(), steps, step_size, coef))
            rows.append({"run": f.name.split(".csv")[0], "split": split, "method": "size_prior",
                         "setting": round(coef, 3), "n_steps": g["time_step"].nunique(),
                         "delta": auc - base})

res = pd.DataFrame(rows)
res.round(5).to_csv(OUT / "per_step_prior.csv", index=False)
long = res[res["n_steps"] >= 10]

print("\n=== Long splits (10+ steps) ===")
for method in ["sharpen", "size_prior"]:
    sub = long[long["method"] == method]
    if sub.empty:
        continue
    print(f"\n{method}:")
    print(sub.groupby("setting").agg(mean_delta=("delta", "mean"), min_delta=("delta", "min"),
                                     max_delta=("delta", "max"),
                                     share_improved=("delta", lambda d: (d > 0).mean()),
                                     n=("delta", "size")).round(4).to_string())

print("\n=== Short folds (under 10 steps) ===")
short = res[res["n_steps"] < 10]
print(short.groupby(["method", "setting"]).agg(mean_delta=("delta", "mean"),
                                               share_improved=("delta", lambda d: (d > 0).mean())
                                               ).round(4).to_string())

best = long[long["method"] == "sharpen"].groupby("setting")["delta"].mean()
if len(best):
    g = best.idxmax()
    print(f"\nBest sharpening setting on long splits: gamma = {g}, mean delta {best.max():+.4f}")
    print("Negative gammas are the Stage 7 direction and should be clearly harmful; if they are\n"
          "not, treat the whole measurement with suspicion.")
    print(f"\nIf a positive gamma helps consistently:\n"
          f"  python3 stage25/per_step_prior.py --apply <best_blend>.csv --gamma {g}")
