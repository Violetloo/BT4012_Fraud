"""Stage 3: first baselines and the time-feature comparison.

Run from the repo root:  python3 stage3/run_baselines.py [--version v1]

Runs 2 models x 2 feature sets through evaluate(), on identical splits:
    models:       logreg (standardised logistic regression), hgb (histogram gradient boosting)
    feature sets: raw_notime (159 features, time-encoding candidates removed), raw_all (165)
Each run is logged as <model>_<featureset>_<version>, e.g. hgb_raw_notime_v1.

Decision order (see stage3/README.md):
1. Feature-set decision first, by the conservative time-feature rule, judged on hgb.
2. Then the best model within the chosen feature set is the recommended submission.
The ranking of all four configurations is printed for reference only.

Outputs (version-specific, so reruns never overwrite earlier tables):
    stage3/time_feature_comparison_<version>.csv
    stage3/baseline_ranking_<version>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.reporting import HORIZON_WEAK_STEPS, compare_results, weak_step_auc  # noqa: E402
from src.validation import evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent
MODELS = ["logreg", "hgb"]
FEATURE_SETS = ["raw_notime", "raw_all"]
DECIDING_MODEL = "hgb"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="v1", help="suffix for experiment names and output files")
args = p.parse_args()

train, _, _ = load_data(args.data_dir)
results = {}
for m in MODELS:
    for fs in FEATURE_SETS:
        cols = get_feature_columns(fs, train)
        name = f"{m}_{fs}_{args.version}"
        results[(m, fs)] = evaluate(
            name, get_model_factory(m), train, cols,
            description=f"Stage 3 baseline: {m} on {fs} ({len(cols)} features)")

# 1. Time-feature comparison, per model
comparison_rows = []
print("\n=== Time-encoding features: raw_all minus raw_notime (same model, same splits) ===")
for m in MODELS:
    table, s = compare_results(results[(m, "raw_notime")], results[(m, "raw_all")], "raw_notime", "raw_all")
    print(f"\n{m}")
    print(table.round(4).to_string())
    print(f"fold-mean delta {s['delta_fold_mean']:+.4f} | folds improved {s['folds_improved']}/4 | "
          f"min-fold delta {s['delta_fold_min']:+.4f} | horizon delta {s['delta_horizon']:+.4f} | "
          f"weak-steps delta {s['delta_weak_steps']:+.4f}")
    include = s["delta_fold_mean"] > 0 and s["folds_improved"] >= 3 and s["delta_horizon"] >= 0
    print(f"rule-based suggestion: {'include' if include else 'keep excluded'} the six time-encoding features")
    comparison_rows.append({"model": m, **s, "suggestion": "include" if include else "keep excluded"})
comp = pd.DataFrame(comparison_rows)
comp.round(5).to_csv(OUT / f"time_feature_comparison_{args.version}.csv", index=False)

# 2. Ranking of all configurations (reference only)
rank = pd.DataFrame([{"config": f"{m}_{fs}", "model": m, "feature_set": fs,
                      **{k: v for k, v in r["summary"].items() if k != "runtime_s"},
                      "auc_weak_steps": weak_step_auc(r),
                      "runtime_warnings": r["warnings"]["n_runtime_warnings"],
                      "run_id": r["run_id"]} for (m, fs), r in results.items()])
rank = rank.sort_values("auc_mean", ascending=False)
rank.round(5).to_csv(OUT / f"baseline_ranking_{args.version}.csv", index=False)
print(f"\n=== All configurations by mean fold AUC (reference; weak steps = horizon steps {HORIZON_WEAK_STEPS}) ===")
print(rank.drop(columns="run_id").round(4).to_string(index=False))

# 3. Recommendation: feature decision first, then best model within that feature set
decided = comp.set_index("model").loc[DECIDING_MODEL, "suggestion"]
chosen_fs = "raw_all" if decided == "include" else "raw_notime"
best = rank[rank["feature_set"] == chosen_fs].iloc[0]
print(f"\n=== Recommendation ===")
print(f"Feature set: {chosen_fs} (time-feature rule on {DECIDING_MODEL}: {decided}). "
      "This is a conservative rule-based choice; it is not necessarily the highest validation AUC.")
print(f"Best model within {chosen_fs}: {best['model']} (fold mean {best['auc_mean']:.4f}, "
      f"horizon {best['auc_horizon']:.4f}).")
if best["runtime_warnings"] > 0:
    print("Note: this run recorded runtime warnings; run stage3/investigate_lr_warnings.py before relying on it.")
print(f"Submit with:\n  python3 stage3/make_submission.py --model {best['model']} --features {chosen_fs} "
      f"--run-id {best['run_id']}")
print(f"\nSaved time_feature_comparison_{args.version}.csv and baseline_ranking_{args.version}.csv in {OUT}")
