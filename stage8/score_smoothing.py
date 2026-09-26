"""Stage 8a: does smoothing predictions over the graph help?

Run from the repo root:  python3 stage8/score_smoothing.py [--contains raw_cmp]

Each transaction's score is blended with the scores of the transactions it is connected
to (or of its whole component). If illicit activity clusters in the graph, neighbours'
predictions carry information the model cannot see, because it scores each row alone.
This is post-processing: no labels, no retraining, valid at submission time.

The analysis reads saved validation predictions from experiments/oof/ and rescores each
split under every (mode, alpha, iterations) combination. Long-horizon splits matter
most: they span many steps, as the test set does.

Only scored transactions contribute, which is what keeps validation comparable to test:
a validation row's scored neighbours are the labelled rows of its block (~63% of its
neighbours), and a test row's are the test rows (~61%).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.smoothing import MODES, smooth_scores  # noqa: E402
from src.validation import PRED_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--contains", default="raw_cmp", help="only prediction files whose name contains this")
p.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5])
p.add_argument("--iters", nargs="*", type=int, default=[1, 2])
p.add_argument("--max-files", type=int, default=12, help="most recent N prediction files")
args = p.parse_args()

_, _, edges = load_data(args.data_dir)
files = sorted(PRED_DIR.glob("*.csv.gz"))
if args.contains:
    files = [f for f in files if args.contains in f.name]
files = files[-args.max_files:]
if not files:
    sys.exit(f"No prediction files found in {PRED_DIR}")
print(f"Analysing {len(files)} prediction files, {len(MODES)} modes, "
      f"{len(args.alphas)} alphas, iterations {args.iters}")

rows = []
for f in files:
    run_id = f.name.split(".csv")[0]
    df = pd.read_csv(f)
    for split, g in df.groupby("split"):
        if g["label"].nunique() < 2:
            continue
        base = roc_auc_score(g["label"], g["pred"])
        rows.append({"run_id": run_id, "split": split, "n_steps": g["time_step"].nunique(),
                     "mode": "none", "alpha": 0.0, "iters": 0, "auc": base, "delta": 0.0})
        for mode in MODES:
            for alpha in args.alphas:
                for iters in args.iters:
                    sm = smooth_scores(g["pred"].values, g["txId"].values, edges, mode, alpha, iters)
                    auc = roc_auc_score(g["label"], sm)
                    rows.append({"run_id": run_id, "split": split, "n_steps": g["time_step"].nunique(),
                                 "mode": mode, "alpha": alpha, "iters": iters,
                                 "auc": auc, "delta": auc - base})

res = pd.DataFrame(rows)
res.round(5).to_csv(OUT / "score_smoothing.csv", index=False)
sm = res[res["mode"] != "none"]

long = sm[sm["n_steps"] >= 10]
print("\n=== Long splits (10+ steps): effect of each setting ===")
summary = (long.groupby(["mode", "alpha", "iters"])
           .agg(mean_delta=("delta", "mean"), min_delta=("delta", "min"), max_delta=("delta", "max"),
                share_improved=("delta", lambda d: (d > 0).mean()), n=("delta", "size"))
           .sort_values("mean_delta", ascending=False))
print(summary.round(4).to_string())

print("\n=== Same settings on the 5-step folds ===")
short = sm[sm["n_steps"] < 10]
print((short.groupby(["mode", "alpha", "iters"])
       .agg(mean_delta=("delta", "mean"), share_improved=("delta", lambda d: (d > 0).mean()))
       .sort_values("mean_delta", ascending=False).head(8).round(4).to_string()))

best = summary.index[0]
best_row = summary.iloc[0]
print(f"\n=== Best setting on long splits: mode={best[0]}, alpha={best[1]}, iters={best[2]} ===")
print(f"mean delta {best_row['mean_delta']:+.4f}, improved in {best_row['share_improved']:.0%} of cases, "
      f"range {best_row['min_delta']:+.4f} to {best_row['max_delta']:+.4f}")
detail = long[(long["mode"] == best[0]) & (long["alpha"] == best[1]) & (long["iters"] == best[2])]
print("\nPer split (mean over runs):")
print(detail.groupby("split")[["auc", "delta"]].mean().round(4).to_string())
print(f"\nSaved {OUT / 'score_smoothing.csv'}")
print("\nAccept only if the mean delta is clearly positive AND most cases improve. Then submit with\n"
      "  python3 stage3/make_submission.py ... --smooth-mode <mode> --smooth-alpha <a> --smooth-iters <n>")
