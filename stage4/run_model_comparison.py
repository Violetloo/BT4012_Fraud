"""Stage 4: compare model families and class weighting on identical splits.

Run from the repo root:  python3 stage4/run_model_comparison.py [--version v1]

Every model runs through evaluate() on the Stage 3 feature set (raw_notime) with
fixed settings and no early stopping, so any model can be retrained for
submission exactly as validated. LightGBM and XGBoost are included only if
installed; missing ones are reported and skipped.

Selection rule (fixed before running; see stage4/README.md). A candidate replaces
the Stage 3 baseline (hgb) only if all three hold:
  1. fold-mean AUC is not worse by more than 0.002,
  2. horizon AUC is strictly better, and
  3. weak-step AUC is not worse by more than 0.005.
Accepted candidates are ordered by a blend of fold mean and horizon AUC, weighted
equally because the test period spans 14 steps while folds look 5 ahead.

Outputs: stage4/model_comparison_<version>.csv, stage4/class_weight_comparison_<version>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.models import CLASS_WEIGHT_PAIRS, MISSING_LIBRARIES, MODELS, get_model_factory  # noqa: E402
from src.reporting import HORIZON_WEAK_STEPS, weak_step_auc  # noqa: E402
from src.validation import evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE = "hgb"
TOL_FOLD_MEAN = 0.002

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s4v1", help="suffix for experiment names and output files")
p.add_argument("--features", default="raw_notime")
p.add_argument("--models", nargs="*", default=None, help="subset of model names (default: all available)")
args = p.parse_args()

if MISSING_LIBRARIES:
    print(f"Optional libraries not installed, their models are skipped: {sorted(MISSING_LIBRARIES)}")
    print("  macOS: brew install libomp && pip install lightgbm xgboost\n")

names = args.models or list(MODELS)
unknown = [n for n in names if n not in MODELS]
if unknown:
    sys.exit(f"Unknown models: {unknown}. Available: {sorted(MODELS)}")
if BASELINE not in names:
    sys.exit(f"The baseline '{BASELINE}' must be included for the comparison to work.")

train, _, _ = load_data(args.data_dir)
cols = get_feature_columns(args.features, train)
print(f"Comparing {len(names)} models on {args.features} ({len(cols)} features): {', '.join(names)}\n")

rows, results = [], {}
for n in names:
    r = evaluate(f"{n}_{args.features}_{args.version}", get_model_factory(n), train, cols,
                 description=f"Stage 4 model comparison: {n} on {args.features}")
    results[n] = r
    rows.append({"model": n, **{k: v for k, v in r["summary"].items()},
                 "auc_weak_steps": weak_step_auc(r),
                 "runtime_warnings": r["warnings"]["n_runtime_warnings"], "run_id": r["run_id"]})

tab = pd.DataFrame(rows).set_index("model")
base = tab.loc[BASELINE]
tab["blend_fold_horizon"] = 0.5 * tab["auc_mean"] + 0.5 * tab["auc_horizon"]
tab["d_fold_mean"] = tab["auc_mean"] - base["auc_mean"]
tab["d_horizon"] = tab["auc_horizon"] - base["auc_horizon"]
tab["d_weak_steps"] = tab["auc_weak_steps"] - base["auc_weak_steps"]
tab["accepted"] = (tab["d_fold_mean"] >= -TOL_FOLD_MEAN) & (tab["d_horizon"] > 0)
tab.loc[BASELINE, "accepted"] = False  # the baseline is the thing being beaten
tab.round(5).to_csv(OUT / f"model_comparison_{args.version}.csv")

show = ["auc_mean", "auc_std", "auc_min", "auc_horizon", "auc_weak_steps", "ap_mean",
        "blend_fold_horizon", "runtime_s", "runtime_warnings"]
print(f"=== All models (weak steps = horizon steps {HORIZON_WEAK_STEPS}) ===")
print(tab.sort_values("blend_fold_horizon", ascending=False)[show].round(4).to_string())

print(f"\n=== Against the Stage 3 baseline ({BASELINE}); d_weak_steps is diagnostic only ===")
print(tab.drop(index=BASELINE)[["d_fold_mean", "d_horizon", "d_weak_steps", "accepted"]]
      .sort_values("d_horizon", ascending=False).round(4).to_string())

# Class weighting, pairwise
pairs = [(a, b) for a, b in CLASS_WEIGHT_PAIRS if a in results and b in results]
if pairs:
    cw = pd.DataFrame([{
        "unweighted": a, "weighted": b,
        "d_fold_mean": tab.loc[b, "auc_mean"] - tab.loc[a, "auc_mean"],
        "d_horizon": tab.loc[b, "auc_horizon"] - tab.loc[a, "auc_horizon"],
        "d_weak_steps": tab.loc[b, "auc_weak_steps"] - tab.loc[a, "auc_weak_steps"],
        "d_ap_mean": tab.loc[b, "ap_mean"] - tab.loc[a, "ap_mean"],
    } for a, b in pairs])
    cw.round(5).to_csv(OUT / f"class_weight_comparison_{args.version}.csv", index=False)
    print("\n=== Class weighting (weighted minus unweighted) ===")
    print(cw.round(4).to_string(index=False))
    helped = int((cw["d_horizon"] > 0).sum())
    print(f"Class weighting improved horizon AUC for {helped} of {len(cw)} families. "
          "Compare within a family only: the libraries differ in settings, in how weights enter the\n"
          "objective and regularisation, and in how the ratio is derived (sklearn 'balanced' per fit,\n"
          "'balanced_subsample' per bootstrap sample, XGBoost scale_pos_weight per training split).")

accepted = tab[tab["accepted"]].sort_values("blend_fold_horizon", ascending=False)
print("\n=== Recommendation ===")
if accepted.empty:
    print(f"No candidate meets the rule; keep the Stage 3 baseline ({BASELINE}).")
    best, best_run = BASELINE, tab.loc[BASELINE, "run_id"]
else:
    best = accepted.index[0]
    best_run = accepted.iloc[0]["run_id"]
    print(f"{best} meets the rule: fold mean {accepted.iloc[0]['auc_mean']:.4f} "
          f"({accepted.iloc[0]['d_fold_mean']:+.4f}), horizon {accepted.iloc[0]['auc_horizon']:.4f} "
          f"({accepted.iloc[0]['d_horizon']:+.4f}); weak-step change {accepted.iloc[0]['d_weak_steps']:+.4f} "
          "(diagnostic, not part of the rule).")
    if len(accepted) > 1:
        print(f"Other accepted candidates: {', '.join(accepted.index[1:])}")
print(f"Submit with:\n  python3 stage3/make_submission.py --model {best} --features {args.features} --run-id {best_run}")
print(f"\nSaved model_comparison_{args.version}.csv"
      + (f" and class_weight_comparison_{args.version}.csv" if pairs else "") + f" in {OUT}")
