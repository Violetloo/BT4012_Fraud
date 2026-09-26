"""Stage 9: adapting the model to the period it will be scored on.

Run from the repo root:  python3 stage9/run_adaptation.py [--version s9v1]

Everything before this stage learned from the training periods and hoped the result
carried forward. These configurations use the FEATURES of the period being predicted
(and, at submission time, of the test rows). Labels are never used, and the competition
supplies test.csv, so this is transductive learning rather than leakage.

  shift_hgb, shift_hgb_clip5, shift_hgb_clip3
      Importance weighting for covariate shift. A classifier is trained to tell
      training rows from target-period rows; each training row is weighted by its
      density ratio p / (1 - p), clipped at 10 / 5 / 3. Rows resembling the period
      being predicted count more. Stage 6 measured the shift (early-vs-late AUC 0.961)
      but only tried removing features or dropping periods; this reweights instead.
      Weights are cross-fitted, so a row's weight never comes from a model fitted on it.

  selftrain_target, selftrain_target_2rounds
      Confident predictions on the target period's rows are added back as training
      rows. Stage 8b did this with unlabelled TRAINING rows and failed (they were 97.5%
      pseudo-licit and diluted the rare class); here the purpose is different, adapting
      the decision boundary to the distribution being scored.

  per_step_balanced
      Training rows weighted so every time step contributes equally, instead of large
      steps dominating.

  rank_xgb (if XGBoost is installed)
      A pairwise ranking objective instead of log loss, since the metric is AUC.

The target window is inferred inside each fit: the steps after the last training step,
up to 14 ahead. In validation that is the block being predicted; at submission time it
is the test period.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.adaptation import FIT_LOG, set_target, split_target_hook  # noqa: E402
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import MODELS, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
FEATURES = "raw_cmp"
BASELINE = "hgb_balanced"
CANDIDATES = ["shift_hgb", "shift_hgb_clip3", "shift_hgb_reg", "shift_hgb_reg_clip3",
              "selftrain_target", "rank_xgb"]


def per_step_balanced(tr):
    """Each time step contributes the same total weight."""
    counts = tr["time_step"].map(tr["time_step"].value_counts())
    w = 1.0 / counts.values
    return w / w.mean()


per_step_balanced.__name__ = "per_step_balanced"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s9v1")
p.add_argument("--models", nargs="*", default=None)
p.add_argument("--skip-per-step", action="store_true")
args = p.parse_args()

train, test = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)
hook = split_target_hook(cols)
print("Target for each split: exactly the rows that split scores, labels withheld "
      "(at submission time: the test rows)")

names = [BASELINE] + [c for c in (args.models or CANDIDATES) if c in MODELS and c != BASELINE]
rows = []
for name in names:
    print(f"\n--- {name} ---")
    start = len(FIT_LOG)
    needs_target = name.startswith(("shift", "selftrain"))
    rows.append(run_config(f"{name}_{FEATURES}", get_model_factory(name), train, cols, args.version,
                           description=f"Stage 9 adaptation: {name} on {FEATURES}",
                           **({"split_hook": hook} if needs_target else {})))
    fits = FIT_LOG[start:]
    # written after every configuration, so an interrupted run keeps its evidence
    pd.DataFrame(FIT_LOG).to_json(OUT / f"adaptation_fit_log_{args.version}.json",
                                  orient="records", indent=2)
    if fits and fits[0]["model"] == "covariate_shift":
        f = pd.DataFrame(fits)
        q = np.vstack(f["source_p_quantiles"].values).mean(axis=0)
        print(f"  target rows per fit {int(f['n_target'].mean()):,}; source P(target) quantiles "
              f"1% {q[0]:.3g}, 50% {q[2]:.3g}, 99% {q[4]:.3g}")
        print(f"  weights: sd {f['weight_sd'].mean():.3f}, at clip floor "
              f"{f['share_at_clip_floor'].mean():.1%}, at ceiling {f['share_at_clip_ceiling'].mean():.1%}, "
              f"effective sample size {f['ess_fraction'].mean():.1%} of rows")
        if f["weight_sd"].mean() < 1e-6:
            print("  -> weights collapsed to a constant: this configuration cannot differ from the "
                  "baseline")
    elif fits and fits[0]["model"] == "self_train_target":
        added = [r["n_added"] for f in fits for r in f["rounds"]]
        pos = [r["pseudo_positive"] for f in fits for r in f["rounds"]]
        if added:
            print(f"  target rows added per fit: median {int(np.median(added)):,}; "
                  f"pseudo-positive share {sum(pos) / max(sum(added), 1):.1%} "
                  "(labelled data is 11.7% illicit)")

if not args.skip_per_step:
    print("\n--- per_step_balanced (weights, not a model) ---")
    rows.append(run_config(f"{BASELINE}_{FEATURES}_per_step_balanced", get_model_factory(BASELINE),
                           train, cols, args.version, description="Stage 9: per-step balanced weights",
                           sample_weight_fn=per_step_balanced))

tab, dec, best = summarise(rows, f"{BASELINE}_{FEATURES}", OUT, args.version, title="configuration")

if best is not None:
    config = best["config"]
    if config.endswith("per_step_balanced"):
        print(f"\nSubmit with:\n  python3 stage3/make_submission.py --model {BASELINE} "
              f"--features {FEATURES} --run-id {best['run_id']} --sample-weight per_step_balanced")
    else:
        print(f"\nSubmit with:\n  python3 stage3/make_submission.py --model "
              f"{config.replace(f'_{FEATURES}', '')} --features {FEATURES} --run-id {best['run_id']}")
        print("(make_submission registers the target rows automatically for model names starting\n"
              " with 'shift' or 'selftrain')")
