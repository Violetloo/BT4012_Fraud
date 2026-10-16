"""Stage 24: propagation iteration count, and histogram binning.

Run from the repo root:  python3 stage24/run_stage24.py [--version s24v1]

Two parameters that have never been varied, both on the representation that works.

A. ITERATIONS. Every PageRank block used ten iterations. That is not the converged
   representation: the weight still sitting on the ten-hop term is (1 - alpha)^10,
   which is 0.197 at alpha = 0.15 and 0.599 at alpha = 0.05. Only the decay was ever
   varied, never the number of steps, and until the patch in this stage the iteration
   argument was silently ignored.
       ppr_i30_cmp       alpha 0.15, 30 iterations
       ppr_i100_cmp      alpha 0.15, 100 iterations (effectively converged)
       ppr05_i100_cmp    alpha 0.05, 100 iterations (the long-tail decay, converged)
       ppr_mixiters_cmp  the current 10-iteration block AND the 100-iteration block

B. BINNING (max_bins). Four tuning searches varied tree size, learning rate, iteration
   count, regularisation and feature sampling, but never the number of histogram bins.
   Fewer bins mean fewer candidate split points, which can make a model less sensitive
   to fine distinctions that shift between periods. Compared at 63, 127 and the default
   255 on ppr_multi_cmp.

Both sections report against the current baseline (ppr_cmp) on the usual splits.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import SEED, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
ITER_SETS = ["ppr_i30_cmp", "ppr_i100_cmp", "ppr05_i100_cmp", "ppr_mixiters_cmp"]
BINS = [63, 127]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s24v1")
p.add_argument("--baseline", default="ppr_cmp")
p.add_argument("--bins-set", default="ppr_multi_cmp")
p.add_argument("--sets", nargs="*", default=ITER_SETS)
p.add_argument("--skip-bins", action="store_true")
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline, args.bins_set] + args.sets,
                         rebuild=args.rebuild_features)

rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- hgb_balanced on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"hgb_balanced_{name}", get_model_factory("hgb_balanced"), train, cols,
                           args.version, description=f"Stage 24 iterations: {name}"))

if not args.skip_bins:
    bins_cols = get_feature_columns(args.bins_set, train)
    print(f"\n--- hgb_balanced on {args.bins_set}, default 255 bins ---")
    rows.append(run_config(f"hgb_balanced_{args.bins_set}_bins255", get_model_factory("hgb_balanced"),
                           train, bins_cols, args.version,
                           description=f"Stage 24 binning: 255 bins on {args.bins_set}"))
    for b in BINS:
        def factory(b=b):
            return HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
                l2_regularization=0.0, max_bins=b, early_stopping=False,
                class_weight="balanced", random_state=SEED)
        print(f"\n--- hgb_balanced on {args.bins_set}, {b} bins ---")
        rows.append(run_config(f"hgb_bins{b}_{args.bins_set}", factory, train, bins_cols,
                               args.version, description=f"Stage 24 binning: {b} bins on {args.bins_set}"))

tab, dec, best = summarise(rows, f"hgb_balanced_{args.baseline}", OUT, args.version,
                           title="configuration")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
print("\nNote: the binning rows share the same feature set, so compare them with each other\n"
      "(255 vs 127 vs 63) rather than with the baseline row.")
