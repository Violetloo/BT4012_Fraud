"""Stage 14: directed propagation and personalised-PageRank propagation.

Run from the repo root:  python3 stage14/run_directed.py [--version s14v1]

Where Stage 13 got to: hops 2-4 over the SYMMETRISED graph scored 0.95937 public,
and dropping hop 1 helped because the supplied features already encode the one-hop
neighbourhood. Two things that version throws away:

1. Direction. Averaging a transaction's upstream and downstream neighbourhoods
   together loses the distinction between where money came from and where it went.
   In laundering those are not interchangeable. `dir34_*` propagates separately over
   incoming and outgoing edges (din2_/din3_/din4_ and dout2_/dout3_/dout4_).

2. Hop mixing. Taking hop distances one at a time forces a choice of distance.
   Personalised PageRank (`ppr_*`) mixes every distance with a geometric decay:
   H <- (1 - alpha) P H + alpha X, ten iterations, alpha 0.15.

Feature sets tested (all keep the 159 raw features and the component features):
  dir34_cmp          + directed hops 2-4, both directions
  dir34_avl_cmp      + the availability group as well
  ppr_cmp            + the PageRank block only
  dir34_ppr_cmp      + directed hops and the PageRank block
  dir34_prop34_cmp   + directed hops and the symmetric hops 2-4 (the Stage 13 winner)

Selection note: on the propagation family the FOLD mean has tracked the public
leaderboard better than the multi-horizon mean (prop34_cmp had the best fold mean of
any configuration and the best public score; the multi-horizon ordering was inverted).
Both are printed; weigh the fold mean and the worst fold when choosing what to submit.
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
DEFAULT_BASELINE = "prop34_cmp"  # the current best feature set (public 0.95937)
SETS = ["dir34_cmp", "dir34_avl_cmp", "ppr_cmp", "dir34_ppr_cmp", "dir34_prop34_cmp"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s14v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--baseline", default=DEFAULT_BASELINE, help="feature set to compare against")
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

BASELINE = args.baseline
train, _ = load_prepared(args.data_dir, feature_sets=[BASELINE] + args.sets,
                         rebuild=args.rebuild_features)
rows = []
for name in [BASELINE] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {args.model} on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 14: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{BASELINE}", OUT, args.version, title="feature set")
print("\nRanked by fold mean (the metric that has tracked the leaderboard on this family):")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
print("\nBuild a submission for the top one or two by fold mean:")
print(f"  python3 stage3/make_submission.py --model {args.model} --features <set> --run-id <run_id>")
