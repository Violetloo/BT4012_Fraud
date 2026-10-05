"""Stage 15: max-propagation, symmetric normalisation, and combined blocks.

Run from the repo root:  python3 stage15/run_max_sym.py [--version s15v1]

Where you are: ppr_cmp (PageRank propagation, alpha 0.15) scored 0.96034, and blending
it with prop34_cmp reached 0.96085. Every gain since Stage 13 has come from a different
way of summarising the neighbourhood, so this stage attacks the two assumptions the
current propagation still makes.

1. It averages. A mean washes out one extreme neighbour, but for fraud the presence of
   a single suspicious transaction nearby matters more than the neighbourhood's typical
   value. Stage 8a already pointed this way: smoothing predictions with the neighbour
   MAX beat the neighbour mean (94% of cases improved). `mx1_`/`mx2_` propagate the max
   over one and two hops.

2. It weights every neighbour equally. Row normalisation (D^-1 A) ignores how connected
   a neighbour is; symmetric normalisation (D^-1/2 A D^-1/2) down-weights hubs, which
   here are exactly what exchanges and mixers look like. `sym_` is PageRank propagation
   with that weighting.

Feature sets (all keep the 159 raw features and the component block):
  max_cmp        + max-propagated features (1 and 2 hops)
  sym_cmp        + symmetric-normalised PageRank
  ppr_max_cmp    + the current best PageRank block AND the max block
  ppr_sym_cmp    + both normalisations of PageRank

ppr_max_cmp and ppr_sym_cmp also test something the blends could not: giving ONE model
both representations, rather than training two models and averaging their predictions.

Selection: neither the fold mean nor the multi-horizon mean has reliably predicted the
leaderboard (ppr30_cmp led both and lost 0.0027). Prefer a set that is strong on both,
and favour the ones that are genuinely new representations when choosing what to submit.
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
DEFAULT_BASELINE = "ppr_cmp"
SETS = ["max_cmp", "sym_cmp", "ppr_max_cmp", "ppr_sym_cmp"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s15v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--baseline", default=DEFAULT_BASELINE)
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
                           args.version, description=f"Stage 15: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{args.baseline}", OUT, args.version,
                           title="feature set")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
print("\nSubmit the strongest one or two:")
print(f"  python3 stage3/make_submission.py --model {args.model} --features <set> --run-id <run_id>")
