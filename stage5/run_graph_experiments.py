"""Stage 5: add graph feature groups one at a time and test them across long horizons.

Run from the repo root:  python3 stage5/run_graph_experiments.py [--version s5v1]

Feature groups are added cumulatively to the Stage 3 feature set (raw_notime):
    raw_g1  + deg       degree from the complete edge list
    raw_g2  + avail     usable-neighbour counts and shares
    raw_g3  + nbrmean   mean of each supplied feature over predecessors/successors
    raw_g4  + nbrstats  max/min/std over the highest-variance features
    raw_g5  + comp      component size and usable share within the time step

The model is the Stage 4 selection (hgb_balanced), which handles missing values
natively: aggregates are NaN for nodes with no usable neighbour, and are left as NaN
rather than filled.

Acceptance rule (fixed before running; see stage5/README.md). A cumulative set
replaces the previous one only if all three hold, judged against the previous set:
  1. fold-mean AUC is not worse by more than 0.002,
  2. the mean AUC over the four long-horizon splits improves, and
  3. at least 3 of those 4 splits improve.
Condition 3 is the Stage 4 lesson: a gain on one horizon split did not transfer.

Outputs: stage5/graph_feature_results_<version>.csv, stage5/multi_horizon_<version>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.features import GRAPH_STAGES, get_feature_columns  # noqa: E402
from src.graph_features import CACHE_PATH, attach, build_or_load, group_columns  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.reporting import weak_step_auc  # noqa: E402
from src.validation import HORIZON, MULTI_HORIZONS, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent
TOL_FOLD_MEAN = 0.002

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s5v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--rebuild-features", action="store_true", help="rebuild the graph feature cache")
p.add_argument("--sets", nargs="*", default=["raw_notime"] + list(GRAPH_STAGES))
args = p.parse_args()

train, test, edges = load_data(args.data_dir)
print("Building graph features (cached after the first run)...")
graph = build_or_load(train, test, edges, rebuild=args.rebuild_features)
print(f"Graph features: {graph.shape[1]} columns for {graph.shape[0]:,} transactions -> {CACHE_PATH}")
train = attach(train, graph)

for g in ["deg", "avail", "nbrmean", "nbrstats", "comp"]:
    cols = group_columns(train, [g])
    miss = train[cols].isna().mean().mean() if cols else 0
    print(f"  {g:9s} {len(cols):4d} columns, mean missing rate {miss:.1%}")

rows, results = [], {}
for name in args.sets:
    cols = get_feature_columns(name, train)
    r = evaluate(f"{args.model}_{name}_{args.version}", get_model_factory(args.model), train, cols,
                 description=f"Stage 5 graph features: {args.model} on {name} ({len(cols)} features)")
    mh = evaluate(f"{args.model}_{name}_{args.version}_mh", get_model_factory(args.model), train, cols,
                  folds=MULTI_HORIZONS, horizon=HORIZON, log=False, save_predictions=False, verbose=False)
    results[name] = (r, mh)
    mh_aucs = mh["folds"]["auc"].loc[[s.name for s in MULTI_HORIZONS]]
    print(f"  multi-horizon AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in mh_aucs.items())
          + f" | mean {mh_aucs.mean():.4f}")
    rows.append({"feature_set": name, "n_features": len(cols),
                 "auc_fold_mean": r["summary"]["auc_mean"], "auc_fold_min": r["summary"]["auc_min"],
                 "auc_horizon": r["summary"]["auc_horizon"], "auc_weak_steps": weak_step_auc(r),
                 **{f"auc_{k}": v for k, v in mh_aucs.items()},
                 "auc_mh_mean": mh_aucs.mean(), "auc_mh_min": mh_aucs.min(),
                 "run_id": r["run_id"]})

tab = pd.DataFrame(rows).set_index("feature_set")
mh_cols = [f"auc_{s.name}" for s in MULTI_HORIZONS]
tab.round(5).to_csv(OUT / f"graph_feature_results_{args.version}.csv")
tab[mh_cols + ["auc_mh_mean", "auc_mh_min"]].round(5).to_csv(OUT / f"multi_horizon_{args.version}.csv")

print("\n=== Results by feature set ===")
print(tab[["n_features", "auc_fold_mean", "auc_fold_min", "auc_horizon", "auc_weak_steps",
           "auc_mh_mean", "auc_mh_min"]].round(4).to_string())
print("\n=== Long-horizon splits ===")
print(tab[mh_cols].round(4).to_string())

# Step-by-step acceptance against the previous cumulative set
print("\n=== Each group against the previous set ===")
order = [s for s in args.sets]
decisions = []
current = order[0]
for prev, name in zip(order, order[1:]):
    base_row, row = tab.loc[current], tab.loc[name]
    improved = int(sum(row[c] > base_row[c] for c in mh_cols))
    d_fold = row["auc_fold_mean"] - base_row["auc_fold_mean"]
    d_mh = row["auc_mh_mean"] - base_row["auc_mh_mean"]
    keep = bool(d_fold >= -TOL_FOLD_MEAN and d_mh > 0 and improved >= 3)
    print(f"{name:8s} vs {current:10s}: fold mean {d_fold:+.4f}, multi-horizon mean {d_mh:+.4f}, "
          f"splits improved {improved}/4 -> {'KEEP' if keep else 'reject'}")
    decisions.append({"feature_set": name, "compared_with": current, "d_fold_mean": d_fold,
                      "d_mh_mean": d_mh, "splits_improved": improved, "kept": keep})
    if keep:
        current = name
pd.DataFrame(decisions).round(5).to_csv(OUT / f"graph_feature_decisions_{args.version}.csv", index=False)

print("\n=== Recommendation ===")
best = tab.loc[current]
print(f"Feature set: {current} ({int(best['n_features'])} features), fold mean {best['auc_fold_mean']:.4f}, "
      f"multi-horizon mean {best['auc_mh_mean']:.4f} (min {best['auc_mh_min']:.4f}).")
if current == order[0]:
    print("No graph group met the rule; the Stage 3 feature set stays.")
print(f"Submit with:\n  python3 stage3/make_submission.py --model {args.model} --features {current} "
      f"--run-id {best['run_id']}")
print(f"\nSaved graph_feature_results_{args.version}.csv, multi_horizon_{args.version}.csv and "
      f"graph_feature_decisions_{args.version}.csv in {OUT}")
