"""Run one configuration on the standard folds and the long-horizon splits, and
compare configurations against a baseline with the Stage 5 acceptance rule."""
import pandas as pd

from src.reporting import weak_step_auc
from src.validation import HORIZON, MULTI_HORIZONS, evaluate

MH_NAMES = [s.name for s in MULTI_HORIZONS]
TOL_FOLD_MEAN = 0.002


def run_config(label, make_model, frame, feature_cols, version, description="", **kwargs):
    """Evaluate one configuration twice: standard folds (logged) and long horizons (recorded)."""
    r = evaluate(f"{label}_{version}", make_model, frame, feature_cols,
                 description=description or label, **kwargs)
    mh = evaluate(f"{label}_{version}_mh", make_model, frame, feature_cols, folds=MULTI_HORIZONS,
                  horizon=HORIZON, verbose=False, description=f"long-horizon check: {description or label}",
                  **kwargs)
    aucs = mh["folds"]["auc"].loc[MH_NAMES]
    print("  multi-horizon AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in aucs.items())
          + f" | mean {aucs.mean():.4f}")
    return {"config": label, "n_features": len(feature_cols),
            "auc_fold_mean": r["summary"]["auc_mean"], "auc_fold_min": r["summary"]["auc_min"],
            "auc_horizon": r["summary"]["auc_horizon"], "auc_weak_steps": weak_step_auc(r),
            **{f"auc_{k}": v for k, v in aucs.items()},
            "auc_mh_mean": aucs.mean(), "auc_mh_min": aucs.min(), "run_id": r["run_id"]}


def summarise(rows, baseline_label, out_dir, version, title="configurations"):
    """Print the results, the long-horizon splits and each configuration's verdict."""
    tab = pd.DataFrame(rows).set_index("config")
    mh_cols = [f"auc_{n}" for n in MH_NAMES]
    tab.round(5).to_csv(out_dir / f"results_{version}.csv")
    print(f"\n=== Results by {title} ===")
    print(tab[["n_features", "auc_fold_mean", "auc_fold_min", "auc_horizon", "auc_weak_steps",
               "auc_mh_mean", "auc_mh_min"]].round(4).to_string())
    print("\n=== Long-horizon splits ===")
    print(tab[mh_cols].round(4).to_string())

    base = tab.loc[baseline_label]
    print(f"\n=== Against the baseline ({baseline_label}) ===")
    decisions = []
    for label, row in tab.drop(index=baseline_label).iterrows():
        improved = int(sum(row[c] > base[c] for c in mh_cols))
        d_fold = row["auc_fold_mean"] - base["auc_fold_mean"]
        d_mh = row["auc_mh_mean"] - base["auc_mh_mean"]
        keep = bool(d_fold >= -TOL_FOLD_MEAN and d_mh > 0 and improved >= 3)
        print(f"{label:22s}: fold mean {d_fold:+.4f}, multi-horizon mean {d_mh:+.4f}, "
              f"splits improved {improved}/4 -> {'PASS' if keep else 'reject'}")
        decisions.append({"config": label, "d_fold_mean": d_fold, "d_mh_mean": d_mh,
                          "splits_improved": improved, "passes_rule": keep,
                          "run_id": row["run_id"]})
    dec = pd.DataFrame(decisions)
    dec.round(5).to_csv(out_dir / f"decisions_{version}.csv", index=False)

    passing = dec[dec["passes_rule"]].sort_values("d_mh_mean", ascending=False)
    print("\n=== Recommendation ===")
    if passing.empty:
        print(f"Nothing beats the baseline ({baseline_label}); keep it.")
        return tab, dec, None
    best = passing.iloc[0]
    print(f"{best['config']} passes: multi-horizon mean {best['d_mh_mean']:+.4f}, "
          f"fold mean {best['d_fold_mean']:+.4f}, {int(best['splits_improved'])}/4 splits improved.")
    if len(passing) > 1:
        print("Others passing: " + ", ".join(passing["config"][1:]))
    return tab, dec, best
