"""Stage 23: smooth predictions among look-alike transactions in the SAME time step.

Run from the repo root:
    python3 stage23/knn_score_smoothing.py                      # measure on saved predictions
    python3 stage23/knn_score_smoothing.py --apply FILE.csv --alpha 0.2 --k 10

THE GAP THIS TARGETS
--------------------
Stage 8a smoothed predictions over the PAYMENT GRAPH and gained +0.0006 at a tiny
weight. But the payment graph is sparse: a test transaction has a median of two
neighbours, and 39% of the neighbours it does have are missing from the files. Most
look-alike transactions are therefore invisible to it.

Feature space has no such limit. A laundering operation that splits funds across many
transactions in the same period produces transactions that look alike even when no edge
connects them, and the model scores each one in isolation. Averaging a transaction's
score with its nearest look-alikes in the same time step pools exactly that evidence.

Two rules keep it honest and matched between validation and test:
  * neighbours are drawn only from the SAME time step, so nothing crosses periods;
  * no labels are used anywhere, so it is pure post-processing and valid at submission
    time.

The script first measures the effect on the validation predictions already saved in
experiments/oof/, across a grid of k and alpha, before anything is submitted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data, raw_feature_columns  # noqa: E402
from src.validation import PRED_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent
N_COMPONENTS = 32


def embedding(frame, cols, seed=42):
    Z = StandardScaler().fit_transform(frame[cols].to_numpy("float32"))
    return PCA(n_components=N_COMPONENTS, random_state=seed).fit_transform(Z).astype("float32")


def smooth_within_step(scores, Z, steps, k, alpha):
    """Blend each score with the mean score of its k nearest look-alikes in the same step."""
    out = np.asarray(scores, dtype="float64").copy()
    for step in np.unique(steps):
        m = np.where(steps == step)[0]
        if len(m) <= k + 1:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[m])
        _, ind = nn.kneighbors(Z[m])
        neigh = out[m][ind[:, 1:]]           # drop self
        out[m] = (1 - alpha) * out[m] + alpha * neigh.mean(axis=1)
    return out


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--contains", default="ppr_cmp", help="which saved prediction files to measure on")
p.add_argument("--max-files", type=int, default=6)
p.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20])
p.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5])
p.add_argument("--apply", default=None, help="submission CSV to smooth and rewrite")
p.add_argument("--k", type=int, default=10, help="k for --apply")
p.add_argument("--alpha", type=float, default=0.2, help="alpha for --apply")
args = p.parse_args()

train, test, _ = load_data(args.data_dir)
RAW = raw_feature_columns(train, exclude_time_candidates=True)

if args.apply:
    # ── apply the chosen setting to a submission file ─────────────────────────
    sub = pd.read_csv(args.apply if Path(args.apply).exists() else ROOT / "submissions" / args.apply)
    if list(sub.columns) != ["index", "target"]:
        sys.exit(f"{args.apply}: columns {list(sub.columns)} are not ['index', 'target']")
    Z = embedding(test, RAW)
    smoothed = smooth_within_step(sub["target"].to_numpy(), Z, test["time_step"].to_numpy(),
                                  args.k, args.alpha)
    out = pd.DataFrame({"index": sub["index"], "target": smoothed})
    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    assert out["index"].equals(sample["index"]) and np.isfinite(out["target"]).all()
    name = Path(args.apply).stem + f"_knnsmooth_k{args.k}_a{args.alpha:g}.csv"
    path = ROOT / "submissions" / name
    out.to_csv(path, index=False)
    print(f"Spearman correlation with the original ranking: "
          f"{pd.Series(sub['target']).corr(pd.Series(smoothed), method='spearman'):.4f}")
    print(f"Wrote {path}")
    sys.exit(0)

# ── measure on saved validation predictions ───────────────────────────────────
files = sorted(PRED_DIR.glob("*.csv.gz"))
if args.contains:
    files = [f for f in files if args.contains in f.name]
files = files[-args.max_files:]
if not files:
    sys.exit(f"No saved predictions matching '{args.contains}' in {PRED_DIR}")
print(f"Measuring on {len(files)} saved prediction files, k in {args.ks}, alpha in {args.alphas}")

Z_train = embedding(train, RAW)
tx_pos = pd.Series(np.arange(len(train)), index=train["txId"].values)

rows = []
for f in files:
    df = pd.read_csv(f)
    for split, g in df.groupby("split"):
        if g["label"].nunique() < 2:
            continue
        pos = tx_pos.loc[g["txId"]].to_numpy()
        Z = Z_train[pos]
        steps = g["time_step"].to_numpy()
        base = roc_auc_score(g["label"], g["pred"])
        for k in args.ks:
            for alpha in args.alphas:
                s = smooth_within_step(g["pred"].to_numpy(), Z, steps, k, alpha)
                rows.append({"run": f.name.split(".csv")[0], "split": split, "k": k, "alpha": alpha,
                             "n_steps": g["time_step"].nunique(),
                             "auc": roc_auc_score(g["label"], s), "delta": roc_auc_score(g["label"], s) - base})

res = pd.DataFrame(rows)
res.round(5).to_csv(OUT / "knn_score_smoothing.csv", index=False)

long = res[res["n_steps"] >= 10]
print("\n=== Long splits (10+ steps): effect of each setting ===")
summary = (long.groupby(["k", "alpha"])
           .agg(mean_delta=("delta", "mean"), min_delta=("delta", "min"), max_delta=("delta", "max"),
                share_improved=("delta", lambda d: (d > 0).mean()), n=("delta", "size"))
           .sort_values("mean_delta", ascending=False))
print(summary.round(4).to_string())

print("\n=== Same settings on the 5-step folds ===")
short = res[res["n_steps"] < 10]
print(short.groupby(["k", "alpha"]).agg(mean_delta=("delta", "mean"),
                                        share_improved=("delta", lambda d: (d > 0).mean())
                                        ).sort_values("mean_delta", ascending=False).head(6).round(4).to_string())

best = summary.index[0]
print(f"\nBest on long splits: k={best[0]}, alpha={best[1]}, mean delta "
      f"{summary.iloc[0]['mean_delta']:+.4f}, improved in {summary.iloc[0]['share_improved']:.0%} of cases")
print(f"\nSaved {OUT / 'knn_score_smoothing.csv'}")
print("\nApply it to your best submission only if the mean delta is clearly positive AND most\n"
      "cases improve, e.g.:\n"
      "  python3 stage23/knn_score_smoothing.py --apply <best_blend>.csv --k 10 --alpha 0.2")
