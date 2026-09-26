"""Stage 10: blending one model across DIFFERENT FEATURE SETS.

Run from the repo root:  python3 stage10/run_feature_blends.py [--version s10v1]

Stage 7 blended hgb, LightGBM and XGBoost on the same features and gained nothing: the
members made nearly the same errors. This blends the same model across different feature
views, the one axis of diversity left untested. The views have measurably different
strengths: raw_avl was the best set on h15 (0.9319), raw_cmp on h18 and h21 (0.9096,
0.9172), and raw_all had the best fold mean in Stage 3.

Each member is fitted on its own view and the members are combined by average rank
(their probability scales differ). evaluate() receives the union of the views' columns;
each member sees only its own.

Blends tested (all with hgb_balanced):
  fsblend_cmp_avl          raw_cmp + raw_avl
  fsblend_cmp2_avl1        the same, weighted 2:1 toward raw_cmp
  fsblend_cmp_avl_all      raw_cmp + raw_avl + raw_all
  fsblend_cmp_avl_notime   raw_cmp + raw_avl + raw_notime
  fsblend_cmp_avl_drop5    raw_cmp + raw_avl + raw_cmp_drop5
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.ensembles import register_views, union_columns  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE_VIEW = "raw_cmp"
VIEW_NAMES = ["raw_cmp", "raw_avl", "raw_all", "raw_notime", "raw_cmp_drop5"]
BLENDS = ["fsblend_cmp_avl", "fsblend_cmp2_avl1", "fsblend_cmp_avl_all", "fsblend_cmp_avl_notime",
          "fsblend_cmp_avl_drop5"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s10v1")
p.add_argument("--models", nargs="*", default=None)
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=VIEW_NAMES)
views = {}
for name in VIEW_NAMES:
    try:
        views[name] = get_feature_columns(name, train)
    except FileNotFoundError as e:  # raw_cmp_drop5 needs stage6's drop lists
        print(f"  view {name} unavailable: {e}")
register_views(views)
for name, cols in views.items():
    print(f"  view {name:15s} {len(cols):4d} columns")

rows = []
base_cols = views[BASELINE_VIEW]
print(f"\n--- hgb_balanced on {BASELINE_VIEW} (baseline) ---")
rows.append(run_config(f"hgb_balanced_{BASELINE_VIEW}", get_model_factory("hgb_balanced"), train,
                       base_cols, args.version, description=f"Stage 10 baseline on {BASELINE_VIEW}"))

for name in (args.models or BLENDS):
    factory = get_model_factory(name)
    if any(v not in views for v in factory().views):
        print(f"\n--- {name}: skipped (needs views {[v for v in factory().views if v not in views]}) ---")
        continue
    cols = union_columns(factory().views)
    print(f"\n--- {name} ({len(cols)} union columns) ---")
    rows.append(run_config(name, factory, train, cols, args.version,
                           description=f"Stage 10 feature-set blend: {name}"))

tab, dec, best = summarise(rows, f"hgb_balanced_{BASELINE_VIEW}", OUT, args.version, title="blend")
if best is not None:
    print(f"\nSubmit with:\n  python3 stage3/make_submission.py --model {best['config']} "
          f"--features {BASELINE_VIEW} --run-id {best['run_id']}")
    print("(the submission script registers the same views before fitting)")
