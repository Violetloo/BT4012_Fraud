"""Stage 6c: tune the model with an inner chronological holdout.

Run from the repo root:  python3 stage6/tune_hgb.py [--n-trials 20] [--version s6tune]

Hyperparameters are chosen on INNER splits carved out of training windows only:
for k in 15, 20, 25, 30 the model trains on steps 1..k-5 and is scored on steps
k-4..k. The outer folds and long-horizon blocks are never touched during the search,
so the winning settings can then be evaluated on them once, honestly.

The search is a random search with a fixed seed over the hgb_balanced settings.
Early stopping is not used: the number of iterations is a searched parameter, so the
final model retrained on all labelled data uses exactly the validated configuration.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import SEED, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE = "hgb_balanced"
FEATURES = "raw_cmp"
INNER_ENDS = (15, 20, 25, 30)
INNER_VAL_STEPS = 5

SPACE = {"learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
         "max_iter": [200, 300, 500, 800],
         "max_leaf_nodes": [15, 31, 63],
         "min_samples_leaf": [10, 20, 50, 100],
         "l2_regularization": [0.0, 0.1, 1.0, 10.0],
         "max_features": [0.5, 0.8, 1.0]}


def sample_params(rng):
    return {k: v[rng.integers(len(v))] for k, v in SPACE.items()}


def inner_score(params, lab, cols):
    """Mean AUC over the inner chronological holdouts."""
    aucs = []
    for end in INNER_ENDS:
        cut = end - INNER_VAL_STEPS
        tr = lab[lab["time_step"] <= cut]
        va = lab[lab["time_step"].between(cut + 1, end)]
        m = HistGradientBoostingClassifier(early_stopping=False, class_weight="balanced",
                                           random_state=SEED, **params)
        m.fit(tr[cols], tr["label"].astype(int))
        aucs.append(roc_auc_score(va["label"], m.predict_proba(va[cols])[:, 1]))
    return float(np.mean(aucs)), aucs


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s6tune")
p.add_argument("--n-trials", type=int, default=20)
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)
lab = train[train["label"].notna()]
rng = np.random.default_rng(SEED)

print(f"Random search: {args.n_trials} trials, inner holdouts ending at steps {INNER_ENDS} "
      f"(last {INNER_VAL_STEPS} steps of each window)")
default = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20,
           "l2_regularization": 0.0, "max_features": 1.0}
trials = []
base_score, _ = inner_score(default, lab, cols)
print(f"  baseline settings: inner AUC {base_score:.4f}")
for i in range(args.n_trials):
    params = sample_params(rng)
    score, per_split = inner_score(params, lab, cols)
    trials.append({**params, "inner_auc": score})
    flag = " <- best so far" if score >= max(t["inner_auc"] for t in trials) else ""
    print(f"  trial {i + 1:2d}/{args.n_trials}: inner AUC {score:.4f}{flag}")

res = pd.DataFrame(trials).sort_values("inner_auc", ascending=False)
res.round(5).to_csv(OUT / f"tuning_trials_{args.version}.csv", index=False)
best_params = {k: v for k, v in res.iloc[0].items() if k != "inner_auc"}
best_params["max_iter"] = int(best_params["max_iter"])
best_params["max_leaf_nodes"] = int(best_params["max_leaf_nodes"])
best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])
print(f"\nBest inner AUC {res.iloc[0]['inner_auc']:.4f} (baseline {base_score:.4f}) with:")
print(json.dumps(best_params, indent=2, default=float))
(OUT / f"tuned_params_{args.version}.json").write_text(json.dumps(best_params, indent=2, default=float))

# Evaluate the tuned settings against the baseline on the outer splits, once.
def tuned():
    return HistGradientBoostingClassifier(early_stopping=False, class_weight="balanced",
                                          random_state=SEED, **best_params)


rows = []
print(f"\n--- {BASELINE} (baseline settings) ---")
rows.append(run_config(f"{BASELINE}_{FEATURES}", get_model_factory(BASELINE), train, cols, args.version,
                       description=f"Stage 6 tuning baseline: {BASELINE} on {FEATURES}"))
print("\n--- tuned settings ---")
rows.append(run_config(f"hgb_tuned_{FEATURES}", tuned, train, cols, args.version,
                       description=f"Stage 6 tuned hgb on {FEATURES}: {best_params}"))
tab, dec, best = summarise(rows, f"{BASELINE}_{FEATURES}", OUT, args.version, title="settings")
if best is not None:
    print("\nTo submit the tuned model, add it to src/models.py as a named factory with these\n"
          "parameters, rerun its evaluation under that name, then use make_submission.py.")
