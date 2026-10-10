"""Stage 19: similarity-neighbour label features.

Run from the repo root:  python3 stage19/run_knn.py [--version s19v1]

The gap this closes. Every graph feature so far propagates FEATURE VALUES, because the
payment graph cannot carry labels: each edge lies inside one time step, so a test
transaction has no labelled neighbour and never will. Feature space has no such
restriction. A test transaction can sit close to labelled training transactions, and
"which known transactions does this resemble, and were they illicit" is information no
block in this project has been able to express.

How it is computed. All rows are standardised and reduced to 32 principal components,
then each row's nearest labelled rows are found. Only labelled rows from STRICTLY
EARLIER time steps count, so a training row never sees its own label or its own step's
labels, and the rule matches test conditions exactly (every labelled row precedes every
test row). Rows with no earlier labelled neighbour get NaN.

In a quick check on the training rows, the single feature knn_label_mean separated the
classes with AUC 0.94 on its own, which is why this is worth a full run.

Feature sets:
  knn_cmp          raw + similarity-neighbour features + component features
  ppr_knn_cmp      the current best PageRank block as well
  prop34_knn_cmp   the 2/3/4-hop block as well
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
SETS = ["knn_cmp", "ppr_knn_cmp", "prop34_knn_cmp"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s19v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--baseline", default="ppr_cmp")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline] + args.sets,
                         rebuild=args.rebuild_features)

knn_cols = [c for c in train.columns if c.startswith("knn_")]
print("\nSimilarity-neighbour features, by label (training rows):")
print(train[train["label"].notna()].groupby("label")[knn_cols].mean().round(3).to_string())
print("\nMissing rate among labelled training rows: "
      f"{train[train['label'].notna()][knn_cols].isna().mean().mean():.1%}")

rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {args.model} on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 19: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{args.baseline}", OUT, args.version,
                           title="feature set")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
