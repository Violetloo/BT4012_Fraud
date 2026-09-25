"""Stage 7b: seed ensembling and model blends.

Run from the repo root:  python3 stage7/run_ensembles.py [--version s7v1]

All members are combined by average RANK, not average probability: AUC depends only on
ordering, and these models are calibrated differently (their mean predictions differ),
so rank averaging avoids one model's scale dominating.

Configurations, all on raw_cmp and judged by the usual rule against hgb_balanced:
  hgb_balanced_seed10   the same model with 10 random seeds, ranks averaged
  blend_hgb_lgbm        hgb_balanced + lgbm_balanced, 50/50
  blend_hgb75_lgbm25    hgb_balanced + lgbm_balanced, 75/25
  blend_hgb_xgb         hgb_balanced + xgb, 50/50
  blend_hgb_lgbm_xgb    all three, equal weights
  blend_seed10_lgbm     the seed ensemble + lgbm_balanced, 75/25

Only a handful of fixed weights are tried. Searching many weights on the same splits
would fit the validation data rather than reveal a real effect.
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
from src.models import MISSING_LIBRARIES, MODELS, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
FEATURES = "raw_cmp"
BASELINE = "hgb_balanced"
CANDIDATES = ["hgb_balanced_seed10", "blend_hgb_lgbm", "blend_hgb75_lgbm25", "blend_hgb_xgb",
              "blend_hgb_lgbm_xgb", "blend_seed10_lgbm"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s7v1")
p.add_argument("--models", nargs="*", default=None)
args = p.parse_args()

if MISSING_LIBRARIES:
    print(f"Optional libraries missing, their blends are skipped: {sorted(MISSING_LIBRARIES)}")
names = args.models or [BASELINE] + [c for c in CANDIDATES if c in MODELS]
train, _ = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)

rows = []
for name in names:
    print(f"\n--- {name} ---")
    rows.append(run_config(f"{name}_{FEATURES}", get_model_factory(name), train, cols, args.version,
                           description=f"Stage 7 ensemble: {name} on {FEATURES}"))

tab, dec, best = summarise(rows, f"{BASELINE}_{FEATURES}", OUT, args.version, title="ensemble")
if best is not None:
    model_name = best["config"].replace(f"_{FEATURES}", "")
    print(f"Submit with:\n  python3 stage3/make_submission.py --model {model_name} "
          f"--features {FEATURES} --run-id {best['run_id']}")
