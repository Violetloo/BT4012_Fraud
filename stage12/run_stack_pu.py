"""Stage 12: stacking and positive-unlabelled learning.

Run from the repo root:  python3 stage12/run_stack_pu.py [--version s12v1]

Two techniques not tried before, both capable of more than the ~0.002 that refinements
have been producing.

Stacking (stack_hgb_lgbm, stack_three, stack_four)
    Every blend so far used hand-picked weights. Stacking LEARNS the combination: inside
    fit(), chronological blocks are carved from the training rows, each base predicts a
    block it did not train on, and a meta-model is fitted on those out-of-fold
    predictions. The bases are then refitted on all training rows. The outer validation
    block is never involved, and base predictions are rank-transformed so no base's
    probability scale dominates. The meta-model's coefficients are logged, which also
    shows which bases actually carry weight.

Positive-unlabelled learning (pu_w005, pu_w01, pu_w03)
    Stage 8b added only CONFIDENT pseudo-labels and lost 0.04-0.06: thresholding picked
    rows the model already agreed with. The standard positive-unlabelled treatment adds
    EVERY unlabelled row as a negative with a small weight (0.05 / 0.1 / 0.3), with no
    threshold and no selection. The premise is supported by Stage 8b's own diagnostic:
    97.5% of unlabelled rows were predicted licit. The pool is restricted per fit to
    steps at or before the last training step, so validation-period rows are never used.

Everything is judged by the usual rule against hgb_balanced on raw_cmp. Given that
validation has shown no correlation with the leaderboard, treat a configuration that is
merely close to the baseline as a candidate worth probing rather than a failure.
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
from src.models import MODELS, get_model_factory  # noqa: E402
from src.pseudo import FIT_LOG as PU_LOG, register_default_pool  # noqa: E402
from src.stacking import FIT_LOG as STACK_LOG, register_steps  # noqa: E402

OUT = Path(__file__).resolve().parent
FEATURES = "raw_cmp"
BASELINE = "hgb_balanced"
CANDIDATES = ["pu_w005", "pu_w01", "pu_w03", "stack_hgb_lgbm", "stack_three", "stack_four"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s12v1")
p.add_argument("--models", nargs="*", default=None)
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)
register_default_pool(train, cols)          # unlabelled rows, for the PU models
register_steps(train["time_step"])          # row -> time step, for the stacking inner blocks
print(f"Unlabelled pool: {int(train['label'].isna().sum()):,} rows")

rows = []
for name in [BASELINE] + [c for c in (args.models or CANDIDATES) if c in MODELS and c != BASELINE]:
    print(f"\n--- {name} ---")
    s_start, p_start = len(STACK_LOG), len(PU_LOG)
    rows.append(run_config(f"{name}_{FEATURES}", get_model_factory(name), train, cols, args.version,
                           description=f"Stage 12: {name} on {FEATURES}"))
    for log, path in [(STACK_LOG, "stack"), (PU_LOG, "pu")]:
        if log:
            pd.DataFrame(log).to_json(OUT / f"{path}_fit_log_{args.version}.json", orient="records",
                                      indent=2)
    new_stack = STACK_LOG[s_start:]
    if new_stack:
        coefs = pd.DataFrame([f["meta_coefficients"] for f in new_stack],
                             columns=new_stack[0]["bases"]).mean()
        print(f"  meta-model weights (mean over fits): "
              + ", ".join(f"{k} {v:+.2f}" for k, v in coefs.items())
              + f" | inner rows {int(pd.Series([f['n_meta_rows'] for f in new_stack]).mean()):,}")
    new_pu = [f for f in PU_LOG[p_start:] if f.get("model") == "pu_weak_negatives"]
    if new_pu:
        f = pd.DataFrame(new_pu)
        print(f"  unlabelled rows added per fit: {int(f['n_unlabelled_added'].mean()):,} at weight "
              f"{f['unlabelled_weight'].iloc[0]} (effective extra weight "
              f"{int(f['effective_extra_weight'].mean()):,} vs {int(len(train[train['label'].notna()]))} real rows)")

tab, dec, best = summarise(rows, f"{BASELINE}_{FEATURES}", OUT, args.version, title="technique")
print("\nTo build a submission from any of these (the script registers the pool and steps):")
print(f"  python3 stage3/make_submission.py --model <name> --features {FEATURES} --run-id <run_id>")
