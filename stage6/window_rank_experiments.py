"""Stage 6d: rolling training windows and matched-population ranks.

Run from the repo root:  python3 stage6/window_rank_experiments.py [--version s6win]

Two corrections to the earlier Stage 6 experiments:

1. `minstep7` and `minstep12` removed fixed early periods, so the training window still
   grew from split to split. They showed that dropping those periods hurt, not that
   recent-window training fails. Here `window10/15/20` keep only the last N time steps
   of each split's training rows: a rolling window of constant length.

2. `rank_cmp` ranked training rows against every row of their step (~4,000, mostly
   unlabelled) while test rows were ranked against ~1,095 rows, all of the labelled
   kind. Different reference populations can create a shift rather than remove one.
   `rankm_cmp` and `rankmraw_cmp` rank every row against the same population in both
   periods: labelled training rows and test rows.

Everything is judged by the usual rule against hgb_balanced on raw_cmp.
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
MODEL = "hgb_balanced"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s6win")
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

feature_sets = [BASELINE, "rankm_cmp", "rankmraw_cmp"]
train, _ = load_prepared(args.data_dir, feature_sets=feature_sets, rebuild=args.rebuild_features)
factory = get_model_factory(MODEL)

configs = [(f"{MODEL}_{BASELINE}", BASELINE, {}),
           (f"{MODEL}_{BASELINE}_window10", BASELINE, {"train_window": 10}),
           (f"{MODEL}_{BASELINE}_window15", BASELINE, {"train_window": 15}),
           (f"{MODEL}_{BASELINE}_window20", BASELINE, {"train_window": 20}),
           (f"{MODEL}_rankm_cmp", "rankm_cmp", {}),
           (f"{MODEL}_rankmraw_cmp", "rankmraw_cmp", {})]

rows = []
for label, fs, kwargs in configs:
    cols = get_feature_columns(fs, train)
    print(f"\n--- {label} ({len(cols)} features) ---")
    rows.append(run_config(label, factory, train, cols, args.version,
                           description=f"Stage 6d: {label}", **kwargs))

tab, dec, best = summarise(rows, f"{MODEL}_{BASELINE}", OUT, args.version, title="configuration")
print("\nA winning rolling window is a training argument, not a feature set: submit it with\n"
      "  python3 stage3/make_submission.py --model ... --features ... --run-id ... --train-window N")
