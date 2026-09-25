"""Stage 6b: per-step rank normalisation, training window and recency weighting.

Run from the repo root:  python3 stage6/run_experiments.py [--version s6exp]

Configurations, all against the current best set (raw_cmp) with hgb_balanced:
  rank_cmp        per-step percentile ranks of the features, plus component features
  rankraw_cmp     raw features and their per-step ranks, plus component features
  minstep7 / 12   raw_cmp but training rows before step 7 / 12 are dropped
                  (steps 1-6 are almost all licit; validation blocks are untouched)
  halflife5 / 10  raw_cmp with recency weights 0.5 ** ((last_train_step - step) / h)

Ranks are computed within each time step over all rows of that step (train, unknown
and test), using feature values only. A row's rank therefore never depends on another
period, which is what removes the level shifts between periods.

Each configuration is judged by the Stage 5 rule against the baseline.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE = "raw_cmp"


def recency_weights(half_life):
    def weights(tr):
        last = tr["time_step"].max()
        return 0.5 ** ((last - tr["time_step"].values) / half_life)
    weights.__name__ = f"recency_halflife_{half_life}"
    return weights


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s6exp")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

feature_sets = [BASELINE, "rank_cmp", "rankraw_cmp"]
train, _ = load_prepared(args.data_dir, feature_sets=feature_sets, rebuild=args.rebuild_features)
factory = get_model_factory(args.model)

configs = [(f"{args.model}_{BASELINE}", BASELINE, {}),
           (f"{args.model}_rank_cmp", "rank_cmp", {}),
           (f"{args.model}_rankraw_cmp", "rankraw_cmp", {}),
           (f"{args.model}_{BASELINE}_minstep7", BASELINE, {"min_train_step": 7}),
           (f"{args.model}_{BASELINE}_minstep12", BASELINE, {"min_train_step": 12}),
           (f"{args.model}_{BASELINE}_halflife5", BASELINE, {"sample_weight_fn": recency_weights(5)}),
           (f"{args.model}_{BASELINE}_halflife10", BASELINE, {"sample_weight_fn": recency_weights(10)})]

rows = []
for label, fs, kwargs in configs:
    cols = get_feature_columns(fs, train)
    print(f"\n--- {label} ({len(cols)} features) ---")
    rows.append(run_config(label, factory, train, cols, args.version,
                           description=f"Stage 6: {label}", **kwargs))

tab, dec, best = summarise(rows, f"{args.model}_{BASELINE}", OUT, args.version, title="configuration")
print("\nNote: min_train_step and recency weights change how the model is trained, not the feature\n"
      "set, so a winning one must be reproduced with the same argument at submission time.")
