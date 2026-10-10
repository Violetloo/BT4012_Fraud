"""Stage 17: who takes part in propagation.

Run from the repo root:  python3 stage17/run_population.py [--version s17v1]

Your propagation currently runs over labelled training rows and test rows only. That
rule came from Stage 5, where it matched one-hop coverage between periods. For MULTI-HOP
propagation it throws away far more than it was meant to, and the cost has never been
measured:

  * 110,537 unlabelled training rows have complete features and are excluded. They do
    not merely drop out of the averages: a two-hop path running through an unlabelled
    transaction is severed.
  * 46,668 graph nodes have no feature row, and 24,245 test-incident edges point at
    them. Removing them cuts precisely the routes that join test transactions together.

Three variants (no labels involved in any of them):
  pprall_cmp           unlabelled training rows take part
  pprrelay_cmp         featureless nodes stay in as pass-through (they relay, never restart)
  pprallrelay_cmp      both
  ppr_pprallrelay_cmp  the current best block plus the both-variant

The coverage asymmetry Stage 5 worried about is real, which is why this is measured
rather than assumed: compare each against ppr_cmp on the usual splits.
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
SETS = ["pprall_cmp", "pprrelay_cmp", "pprallrelay_cmp", "ppr_pprallrelay_cmp"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s17v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--baseline", default="ppr_cmp")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline] + args.sets,
                         rebuild=args.rebuild_features)
rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    miss = train[cols].isna().mean().mean()
    print(f"\n--- {args.model} on {name} ({len(cols)} features, {miss:.1%} missing) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 17: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{args.baseline}", OUT, args.version,
                           title="propagation population")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
