"""Stage 7a: does normalising predictions within each time step help?

Run from the repo root:  python3 stage7/per_step_normalisation.py [--contains raw_cmp]

The competition's AUC pools all 15,329 test rows across 14 time steps. If a model's
scores sit systematically higher in some periods than others, that pooled ranking
suffers even when the ordering inside each step is good. Converting predictions to
percentile ranks within each time step removes those level differences.

This needs no retraining: it reads the saved validation predictions in
experiments/oof/ and recomputes each split's AUC with and without per-step
normalisation. Long-horizon splits matter most, because they span many steps as the
test set does; the 5-step folds span fewer, so any effect there will be smaller.

Normalisation uses predictions and time steps only, never labels, so it is valid at
submission time (stage3/make_submission.py --per-step-normalise).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ensembles import per_step_rank  # noqa: E402
from src.validation import PRED_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="*", default=None, help="run ids to analyse (default: all saved)")
p.add_argument("--contains", default="", help="only runs whose id contains this text")
args = p.parse_args()

files = sorted(PRED_DIR.glob("*.csv.gz"))
if args.runs:
    files = [f for f in files if f.name.split(".csv")[0] in set(args.runs)]
if args.contains:
    files = [f for f in files if args.contains in f.name]
if not files:
    sys.exit(f"No prediction files found in {PRED_DIR}")

rows = []
for f in files:
    run_id = f.name.split(".csv")[0]
    df = pd.read_csv(f)
    for split, g in df.groupby("split"):
        if g["label"].nunique() < 2:
            continue
        raw = roc_auc_score(g["label"], g["pred"])
        norm = roc_auc_score(g["label"], per_step_rank(g["pred"], g["time_step"]))
        rows.append({"run_id": run_id, "split": split, "n_steps": g["time_step"].nunique(),
                     "auc_raw": raw, "auc_per_step_normalised": norm, "delta": norm - raw})

res = pd.DataFrame(rows)
res.round(5).to_csv(OUT / "per_step_normalisation.csv", index=False)
print(f"Analysed {len(files)} saved prediction files\n")

by_split = res.groupby("split").agg(n_runs=("delta", "size"), mean_steps=("n_steps", "mean"),
                                    mean_auc_raw=("auc_raw", "mean"),
                                    mean_auc_norm=("auc_per_step_normalised", "mean"),
                                    mean_delta=("delta", "mean"), min_delta=("delta", "min"),
                                    max_delta=("delta", "max"),
                                    share_improved=("delta", lambda d: (d > 0).mean()))
print("=== Effect by split (positive delta = normalisation helps) ===")
print(by_split.round(4).to_string())

long_splits = res[res["n_steps"] >= 10]
if not long_splits.empty:
    print(f"\n=== Long splits only ({len(long_splits)} results, 10+ steps) ===")
    print(f"mean delta {long_splits['delta'].mean():+.4f}, improved in "
          f"{(long_splits['delta'] > 0).mean():.0%} of cases, range "
          f"{long_splits['delta'].min():+.4f} to {long_splits['delta'].max():+.4f}")

best = res.sort_values("auc_raw", ascending=False).head(10)
print("\n=== Ten strongest individual results, before and after ===")
print(best[["run_id", "split", "n_steps", "auc_raw", "auc_per_step_normalised", "delta"]]
      .round(4).to_string(index=False, max_colwidth=45))
print(f"\nSaved {OUT / 'per_step_normalisation.csv'}")
print("\nIf the mean delta on long splits is clearly positive, add --per-step-normalise when\n"
      "building the submission. If it is near zero or negative, the pooled ranking is not the\n"
      "problem and nothing changes.")
