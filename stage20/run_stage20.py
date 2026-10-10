"""Stage 20: extend what the leaderboard rewarded, and tune on the winning features.

Run from the repo root:  python3 stage20/run_stage20.py [--version s20v1]

Everything in recent stages was chosen because a mechanism sounded promising, and almost
all of it failed. This stage instead extends the two things the LEADERBOARD actually
rewarded, and closes one gap that was never revisited.

1. MIX MORE DECAYS. The best single model is ppr_multi_cmp (0.96057), which mixes
   PageRank decays 0.05 / 0.15 / 0.30. Each decay alone scored lower (0.96034 at 0.15,
   0.95769 at 0.30), so the mixture is doing the work. `ppr_fine_cmp` and
   `ppr_fine_multi_cmp` add the intermediate decays 0.10, 0.20 and 0.25.

2. UNION THE TWO BEST REPRESENTATIONS. The best submission (0.96120) blends the PageRank
   and discrete-hop models. Giving one model both blocks has never been tested for this
   particular pair: `ppr_multi_prop34_cmp`.

3. TUNE ON THE WINNING FEATURE SET. The only tuning search in the project ran on
   raw_cmp (162 features) and was rejected. The feature set has changed completely since
   -- 321 to 639 columns, two thirds of them propagated and highly correlated -- and
   settings suited to that regime (feature subsampling, stronger regularisation, more
   shallow trees) were never searched. The search here is nested: candidates are scored
   on inner blocks carved from each split's own training rows, never on the outer block.
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
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
SETS = ["ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp"]
SPACE = {"learning_rate": [0.03, 0.05, 0.08], "max_iter": [300, 500, 800],
         "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 50],
         "l2_regularization": [0.0, 1.0, 10.0], "max_features": [0.3, 0.5, 0.8, 1.0]}
BASELINE_PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31,
                   "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0}
INNER = [(15, (11, 15)), (20, (16, 20)), (25, (21, 25))]   # (cutoff, inner block) pairs

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s20v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--baseline", default="ppr_multi_cmp")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--tune-trials", type=int, default=12)
p.add_argument("--tune-set", default="ppr_multi_cmp")
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline, args.tune_set] + args.sets,
                         rebuild=args.rebuild_features)

rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {args.model} on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 20: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{args.baseline}", OUT, args.version,
                           title="feature set")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())

# ── Nested search on the winning feature set ──────────────────────────────────
print(f"\n=== Nested search on {args.tune_set}: {args.tune_trials} random candidates "
      "plus the current settings ===")
cols = get_feature_columns(args.tune_set, train)
lab = train[train["label"].notna()]
rng = np.random.default_rng(42)
candidates = [dict(BASELINE_PARAMS)]
while len(candidates) < args.tune_trials + 1:
    c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
    if c not in candidates:
        candidates.append(c)


def inner_score(params):
    scores = []
    for cutoff, (lo, hi) in INNER:
        tr = lab[lab["time_step"] <= lo - 1]
        va = lab[lab["time_step"].between(lo, hi)]
        if tr["label"].nunique() < 2 or va["label"].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(early_stopping=False, class_weight="balanced",
                                           random_state=42, **params)
        m.fit(tr[cols], tr["label"].astype(int))
        scores.append(roc_auc_score(va["label"], m.predict_proba(va[cols])[:, 1]))
    return float(np.mean(scores))


trials = []
for i, params in enumerate(candidates):
    s = inner_score(params)
    trials.append({**params, "inner_auc": s})
    tag = " (current settings)" if params == BASELINE_PARAMS else ""
    print(f"  trial {i:2d}: inner AUC {s:.4f}{tag}")

res = pd.DataFrame(trials).sort_values("inner_auc", ascending=False)
res.round(5).to_csv(OUT / f"tuning_{args.version}.csv", index=False)
best_params = {k: v for k, v in res.iloc[0].items() if k != "inner_auc"}
for k in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
    best_params[k] = int(best_params[k])
(OUT / f"tuned_params_{args.version}.json").write_text(json.dumps(best_params, indent=2, default=float))
print(f"\nBest inner AUC {res.iloc[0]['inner_auc']:.4f} "
      f"(current settings {res[res.apply(lambda r: all(r[k] == v for k, v in BASELINE_PARAMS.items()), axis=1)]['inner_auc'].iloc[0]:.4f})")
print(json.dumps(best_params, indent=2, default=float))
print("\nIf the search found something clearly better, add it to src/models.py as a named\n"
      "factory, evaluate it under that name, and only then submit it.")
