"""Stage 3: first baselines and the time-feature comparison.

Run from the repo root:  python3 stage3/run_baselines.py [--version v1]

Runs 2 models x 2 feature sets through evaluate(), on identical splits:
    models:       logreg (standardised logistic regression), hgb (histogram gradient boosting)
    feature sets: raw_notime (159 features, time-encoding candidates removed), raw_all (165)
Each run is logged as <model>_<featureset>_<version>, e.g. hgb_raw_notime_v1.

Then prints
1. the time-feature comparison per model (raw_all minus raw_notime, per split), and
2. a ranking of the four configurations by mean fold AUC,
and saves both to stage3/.
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
from src.reporting import compare_results  # noqa: E402
from src.validation import evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent
MODELS = ["logreg", "hgb"]
FEATURE_SETS = ["raw_notime", "raw_all"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="v1", help="suffix for experiment names; change it to rerun")
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
          f"min-fold delta {s['delta_fold_min']:+.4f} | horizon delta {s['delta_horizon']:+.4f}")
    keep = s["delta_fold_mean"] > 0 and s["folds_improved"] >= 3 and s["delta_horizon"] >= 0
    print(f"rule-based suggestion: {'include' if keep else 'keep excluded'} the six time-encoding features")
    comparison_rows.append({"model": m, **s, "suggestion": "include" if keep else "keep excluded"})
pd.DataFrame(comparison_rows).round(5).to_csv(OUT / "time_feature_comparison.csv", index=False)

# 2. Ranking of all configurations
rank = pd.DataFrame([{"config": f"{m}_{fs}", "model": m, "feature_set": fs,
                      **{k: v for k, v in r["summary"].items() if k != "runtime_s"},
                      "run_id": r["run_id"]} for (m, fs), r in results.items()])
rank = rank.sort_values("auc_mean", ascending=False)
rank.round(5).to_csv(OUT / "baseline_ranking.csv", index=False)
print("\n=== Ranking by mean fold AUC ===")
print(rank.drop(columns="run_id").round(4).to_string(index=False))
print(f"\nSaved {OUT / 'time_feature_comparison.csv'} and {OUT / 'baseline_ranking.csv'}")
