"""Stage 8b: pseudo-labelling the unlabelled training rows.

Run from the repo root:  python3 stage8/run_pseudo_labelling.py [--version s8b]

78% of train.csv (110,537 rows) has no label and has never been used for fitting. Here a
first model scores them, the most confident predictions are added as training rows, and
the model is refitted.

Leak-free by construction: inside every fit, the pool is restricted to unlabelled rows
whose time step is at or before the last step of that split's training rows. A split
validating steps 26-30 never sees unlabelled rows from those steps.

Variants (all on raw_cmp, judged against hgb_balanced by the usual rule):
  pseudo_hgb          p >= 0.99 or <= 0.01, added at full weight, one round
  pseudo_hgb_w05      the same rows at half weight
  pseudo_hgb_loose    a looser band (0.95 / 0.05) at half weight
  pseudo_hgb_2rounds  two rounds of confident pseudo-labelling at half weight

Stage 1 found unlabelled rows differ from labelled ones (median KS 0.14), so confident
predictions may reflect the model's existing biases rather than new information. The
script reports how many rows each fit added and how many were pseudo-positive.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.pseudo import FIT_LOG, register_default_pool  # noqa: E402

OUT = Path(__file__).resolve().parent
FEATURES = "raw_cmp"
BASELINE = "hgb_balanced"
VARIANTS = ["pseudo_hgb", "pseudo_hgb_w05", "pseudo_hgb_loose", "pseudo_hgb_2rounds"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s8b")
p.add_argument("--models", nargs="*", default=None)
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)
register_default_pool(train, cols)
n_pool = int(train["label"].isna().sum())
print(f"Unlabelled pool: {n_pool:,} rows, steps {int(train.loc[train['label'].isna(), 'time_step'].min())}"
      f"-{int(train.loc[train['label'].isna(), 'time_step'].max())}")

rows = []
for name in [BASELINE] + (args.models or VARIANTS):
    print(f"\n--- {name} ---")
    start = len(FIT_LOG)
    rows.append(run_config(f"{name}_{FEATURES}", get_model_factory(name), train, cols, args.version,
                           description=f"Stage 8b pseudo-labelling: {name} on {FEATURES}"))
    fits = FIT_LOG[start:]
    if fits:
        added = [r["n_added"] for f in fits for r in f["rounds"]]
        pos = [r["n_pseudo_positive"] for f in fits for r in f["rounds"]]
        print(f"  pseudo-labels added per fit: min {min(added):,}, median {int(pd.Series(added).median()):,}, "
              f"max {max(added):,}; pseudo-positive share "
              f"{sum(pos) / max(sum(added), 1):.1%} (labelled data is 11.7% illicit)")

tab, dec, best = summarise(rows, f"{BASELINE}_{FEATURES}", OUT, args.version, title="variant")
pd.DataFrame(FIT_LOG).to_json(OUT / f"pseudo_fit_log_{args.version}.json", orient="records", indent=2)
if best is not None:
    model_name = best["config"].replace(f"_{FEATURES}", "")
    print(f"Submit with:\n  python3 stage3/make_submission.py --model {model_name} "
          f"--features {FEATURES} --run-id {best['run_id']}")
