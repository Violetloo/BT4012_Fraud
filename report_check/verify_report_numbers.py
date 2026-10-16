"""Check the report's numbers against the project's own logs.

Run from the repo root:  python3 report_check/verify_report_numbers.py

Column map (from experiments/experiment_log.csv):
    auc_mean      mean AUC over the four 5-step folds  -> the report's "fold mean"
    auc_horizon   AUC on the single 1-21 / 22-35 split
The multi-horizon mean (h15/h18/h21/h24) is not in the main log; the stage scripts wrote
it to stage*/results_*.csv, so those files are read as well and matched by config name.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

log = pd.read_csv(ROOT / "experiments" / "experiment_log.csv")
print(f"experiment log: {len(log):,} runs\n")

# multi-horizon means live in the per-stage summary files
mh = {}
for f in sorted(ROOT.glob("stage*/results_*.csv")) + sorted(ROOT.glob("*/stage*/results_*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    key = "config" if "config" in d.columns else d.columns[0]
    if "auc_mh_mean" in d.columns:
        for _, r in d.iterrows():
            mh.setdefault(str(r[key]), []).append(float(r["auc_mh_mean"]))
print(f"multi-horizon values recovered for {len(mh)} configurations "
      f"from {len(list(ROOT.glob('stage*/results_*.csv')))} stage summary files\n")


def find(prefix):
    rows = log[log["name"].str.startswith(prefix)]
    return None if rows.empty else rows.iloc[-1]


def mh_of(prefix):
    vals = [v for k, vlist in mh.items() if k.startswith(prefix) for v in vlist]
    return float(np.median(vals)) if vals else np.nan


def check(label, prefix, rep_fold, rep_mh):
    row = find(prefix)
    if row is None:
        print(f"{label:34s} NOT FOUND in log (prefix '{prefix}')")
        return False
    lf, lm = float(row["auc_mean"]), mh_of(prefix)
    okf = abs(lf - rep_fold) < 5e-4
    okm = np.isnan(lm) or abs(lm - rep_mh) < 5e-4
    mh_txt = "not logged" if np.isnan(lm) else f"{lm:.4f}"
    print(f"{label:34s} fold {rep_fold:.4f} vs {lf:.4f} {'ok ' if okf else 'DIFF'}   "
          f"mh {rep_mh:.4f} vs {mh_txt} {'ok' if okm else 'DIFF'}")
    return okf and okm


print("=== Section 3: feature-set progression ===")
rows_ok = []
for label, prefix, fold, m in [
    ("raw features only",      "hgb_balanced_raw_notime", 0.9830, 0.9279),
    ("+ component features",   "hgb_balanced_raw_cmp",    0.9826, 0.9362),
    ("+ 1/2-step propagation", "hgb_balanced_prop2_cmp",  0.9821, 0.9380),
    ("2/3/4-step propagation", "hgb_balanced_prop34_cmp", 0.9865, 0.9311),
    ("PageRank a=0.15",        "hgb_balanced_ppr_cmp",    0.9865, 0.9364),
    ("three decays",           "hgb_balanced_ppr_multi_cmp", 0.9859, 0.9390),
    ("three decays, 127 bins", "hgb_bins127_ppr_multi_cmp",  0.9854, 0.9400),
]:
    rows_ok.append(check(label, prefix, fold, m))

print("\n=== Section 2.3: model families (all on raw_notime, Stage 4) ===")
fam_ok = []
for label, prefix, fold, m in [
    ("histogram gradient boosting", "hgb_balanced_raw_notime", 0.9830, 0.9279),
    ("LightGBM",                    "lgbm_balanced_raw_notime", 0.9826, np.nan),
    ("XGBoost",                     "xgb_raw_notime",           0.9812, np.nan),
    ("random forest",               "rf_balanced_raw_notime",   0.9677, np.nan),
    ("extra trees",                 "extratrees_raw_notime",    0.9697, np.nan),
    ("logistic regression",         "logreg_raw_notime",        0.9283, np.nan),
]:
    row = find(prefix)
    if row is None:
        print(f"{label:34s} NOT FOUND (prefix '{prefix}')")
        fam_ok.append(False)
        continue
    lf, lh = float(row["auc_mean"]), float(row["auc_horizon"])
    ok = abs(lf - fold) < 5e-4
    print(f"{label:34s} fold {fold:.4f} vs {lf:.4f} {'ok ' if ok else 'DIFF'}   horizon {lh:.4f}")
    fam_ok.append(ok)

print("\n   (CatBoost and GraphSAGE were run on ppr_cmp, not raw_notime - checked separately)")
for label, prefix in [("CatBoost on ppr_cmp", "catboost_ppr_cmp"), ("hgb on ppr_cmp", "hgb_balanced_ppr_cmp")]:
    row = find(prefix)
    if row is not None:
        print(f"{label:34s} fold {float(row['auc_mean']):.4f}   horizon {float(row['auc_horizon']):.4f}   "
              f"mh {mh_of(prefix) if not np.isnan(mh_of(prefix)) else float('nan'):.4f}")

# ---- leaderboard and figure 3 ----------------------------------------------
lb_path = ROOT / "experiments" / "leaderboard_scores.csv"
if lb_path.exists():
    lb = pd.read_csv(lb_path)
    print(f"\n=== Leaderboard file: {len(lb)} scores, best {lb['public_lb'].max():.5f} ===")
    merged = []
    for _, r in lb.iterrows():
        name = str(r["submission"])
        for prefix in (f"hgb_balanced_{name}", name):
            v = mh_of(prefix)
            if not np.isnan(v):
                merged.append({"name": name, "mh": v, "lb": float(r["public_lb"])})
                break
    m = pd.DataFrame(merged).drop_duplicates("name")
    print(f"matched {len(m)} submissions to a logged multi-horizon score")
    if len(m) >= 5:
        rho = spearmanr(m.mh, m.lb)[0]
        print(f"Spearman(validation, public) over those {len(m)}: {rho:+.3f}")
        print(m.sort_values("lb", ascending=False).round(4).to_string(index=False))
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        ax.scatter(m.mh, m.lb, s=30, color="#39506b")
        ax.set_xlabel("local validation AUC (multi-horizon mean)")
        ax.set_ylabel("public AUC")
        ax.set_title(f"Figure 3. Validation against public score (n = {len(m)}, Spearman {rho:+.2f})",
                     loc="left", fontsize=9.5)
        fig.savefig(OUT / "fig3_from_log.png"); plt.close(fig)
        print(f"\nsaved {OUT / 'fig3_from_log.png'}")
else:
    print(f"\n{lb_path} not found")

bad = rows_ok.count(False) + fam_ok.count(False)
print("\n" + ("Every checked value matches the log." if bad == 0 else
              f"{bad} value(s) differ from the log - send this output and they will be corrected."))
