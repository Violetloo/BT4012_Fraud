"""Stage 13a: multi-hop feature propagation.

Run from the repo root:  python3 stage13/run_propagation.py [--version s13v1]

Stage 12 showed that stacking found nothing left to combine: the meta-model split its
weight almost evenly between hgb and LightGBM, which is the 50/50 blend Stage 7 had
already tried. That says the models are near-duplicates, and that a gap of 0.013 to the
leader is unlikely to come from better combination. It has to come from a representation
none of these models can currently form.

Multi-hop propagation is that representation. The whole feature matrix is pushed through
a row-normalised adjacency matrix once and twice, so each transaction's row includes
information from transactions two hops away that it never touches directly. Stage 5's
neighbour aggregates were one hop, per direction, per feature; two-hop mixing is the part
that is genuinely new, and it is the mechanism a graph neural network relies on.

Feature sets:
  prop1_cmp       raw + 1-hop propagated + component features
  prop2_cmp       raw + 1-hop + 2-hop + component features
  prop2_avl_cmp   the same plus the availability group
  prop2_only      propagated features alone (no raw), as a check on what they carry

Coverage is matched as in Stage 5 (only labelled-train and test rows contribute), and no
labels are used at any point.
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
BASELINE = "raw_cmp"
SETS = ["prop1_cmp", "prop2_cmp", "prop2_avl_cmp", "prop2_only"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s13v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[BASELINE] + args.sets,
                         rebuild=args.rebuild_features)
rows = []
for name in [BASELINE] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {args.model} on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 13 propagation: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{BASELINE}", OUT, args.version, title="feature set")
if best is not None:
    print(f"\nSubmit with:\n  python3 stage3/make_submission.py --model {args.model} "
          f"--features {best['config'].replace(args.model + '_', '')} --run-id {best['run_id']}")
