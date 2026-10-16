"""Rebuild the report's member table, the ensemble metrics and the figure from the exact
validation run each submission points to.

Run from the repo root:  python3 report_check/rebuild_from_submissions.py

Why: the earlier checking script matched runs by name prefix and took medians across
configurations sharing that prefix. That is unsafe, because `hgb_balanced_prop34_cmp`
matches both the real Stage 13 run (639 predictors) and a later Stage 14 run with the
same name but only 162 predictors and no propagation columns. This script instead reads
each submission's JSON record, follows its validated_run_id, checks the feature count,
and recomputes everything from the saved out-of-fold predictions.
"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SUBS, RUNS, OOF = ROOT / "submissions", ROOT / "experiments" / "runs", ROOT / "experiments" / "oof"
LONG = ["h15", "h18", "h21", "h24"]
FOLDS = ["fold1", "fold2", "fold3", "fold4"]

MEMBERS = [  # (label, substring identifying the submission file)
    ("ppr_cmp", "hgb_balanced_ppr_cmp"),
    ("prop34_cmp", "hgb_balanced_prop34_cmp"),
    ("ppr_multi_cmp", "hgb_balanced_ppr_multi_cmp"),
    ("ppr_fine_multi_cmp", "hgb_balanced_ppr_fine_multi_cmp"),
    ("ppr_multi_cmp_bins127", "hgb_bins127_ppr_multi_cmp"),
]


def submission_record(substr):
    """The newest submission JSON whose name contains substr, excluding blends."""
    cands = [p for p in SUBS.glob("*.json") if substr in p.name and "blend" not in p.name]
    if not cands:
        return None
    return sorted(cands)[-1]


# run_id -> configuration name, and the reverse, so the long-horizon run that goes with a
# standard run can be found by name (it is called "<name>_mh" but has its own timestamp).
RUN_NAME, NAME_RUNS = {}, {}
for _p in RUNS.glob("*.json"):
    try:
        _n = json.loads(_p.read_text()).get("name")
    except Exception:
        continue
    if _n:
        RUN_NAME[_p.stem] = _n
        NAME_RUNS.setdefault(_n, []).append(_p.stem)


def _read(run_id):
    for name in (f"{run_id}.csv.gz", f"{run_id}.csv"):
        q = OOF / name
        if q.exists():
            return pd.read_csv(q)
    return None


def load_preds(run_id):
    """Predictions for a run plus its matching long-horizon run, concatenated."""
    parts = [_read(run_id)]
    name = RUN_NAME.get(run_id)
    if name:
        for sibling in NAME_RUNS.get(f"{name}_mh", []):
            parts.append(_read(sibling))
    parts = [d for d in parts if d is not None]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["split", "txId"], keep="first")


print("=== submission -> validated run ===")
frames, used = {}, {}
for label, substr in MEMBERS:
    rec_path = submission_record(substr)
    if rec_path is None:
        print(f"{label:24s} NO submission JSON found containing '{substr}'")
        continue
    rec = json.loads(rec_path.read_text())
    run_id = rec.get("validated_run_id") or rec.get("run_id")
    run_json = RUNS / f"{run_id}.json"
    nfeat = "?"
    if run_json.exists():
        rj = json.loads(run_json.read_text())
        feats = rj.get("features") or rj.get("feature_columns") or []
        nfeat = rj.get("n_features", len(feats))
    print(f"{label:24s} {rec_path.name}\n{'':24s}   -> {run_id}  ({nfeat} predictors)")
    used[label] = {"submission": rec_path.name, "run_id": run_id, "n_features": nfeat,
                   "public_lb": rec.get("public_lb")}
    d = load_preds(run_id)
    if d is None:
        print(f"{'':24s}   !! no saved predictions found in {OOF}")
        continue
    print(f"{'':24s}   splits: {sorted(set(d['split']))}")
    frames[label] = d

missing = [l for l, _ in MEMBERS if l not in frames]
if missing:
    sys.exit(f"\nMissing predictions for: {missing}. Cannot rebuild without them.")

# ---- align members within each split ---------------------------------------
r01 = lambda p: rankdata(p) / (len(p) + 1)
rows_auc, rows_ap = [], []
splits = sorted(set.intersection(*[set(d["split"]) for d in frames.values()]) - {"horizon"})
print(f"\nsplits common to all five members: {splits}")

for split in splits:
    per = {}
    for label, d in frames.items():
        g = d[d["split"] == split][["txId", "label", "pred"]].set_index("txId")
        per[label] = g
    idx = sorted(set.intersection(*[set(g.index) for g in per.values()]))
    y = per[MEMBERS[0][0]].loc[idx, "label"].to_numpy()
    p = {label: per[label].loc[idx, "pred"].to_numpy() for label, _ in MEMBERS}
    b1 = 0.5 * r01(p["ppr_cmp"]) + 0.5 * r01(p["prop34_cmp"])
    b2 = (2 / 3) * r01(b1) + (1 / 3) * r01(p["ppr_multi_cmp"])
    b3 = 0.75 * r01(b2) + 0.25 * r01(p["ppr_fine_multi_cmp"])
    p["ENSEMBLE"] = 0.75 * r01(b3) + 0.25 * r01(p["ppr_multi_cmp_bins127"])
    rows_auc.append({"split": split, "n": len(idx), **{k: roc_auc_score(y, v) for k, v in p.items()}})
    rows_ap.append({"split": split, **{k: average_precision_score(y, v) for k, v in p.items()}})

auc = pd.DataFrame(rows_auc).set_index("split")
ap = pd.DataFrame(rows_ap).set_index("split")
cols = [l for l, _ in MEMBERS] + ["ENSEMBLE"]

print("\n=== ROC AUC by split (submission-linked runs) ===")
print(auc.round(4).to_string())

print("\n=== condensed table for the report ===")
summary = pd.DataFrame({
    "fold_mean_auc": auc.loc[[s for s in FOLDS if s in auc.index], cols].mean(),
    "long_mean_auc": auc.loc[[s for s in LONG if s in auc.index], cols].mean(),
    "worst_long_auc": auc.loc[[s for s in LONG if s in auc.index], cols].min(),
    "long_mean_ap": ap.loc[[s for s in LONG if s in ap.index], cols].mean(),
})
summary["public_lb"] = [used.get(c, {}).get("public_lb", np.nan) for c in cols[:-1]] + [0.96137]
print(summary.round(5).to_string())
summary.round(6).to_csv(OUT / "member_table.csv")

best = summary.drop(index="ENSEMBLE")
print("\nensemble vs best member:")
for col in ["fold_mean_auc", "long_mean_auc", "worst_long_auc", "long_mean_ap"]:
    e, b = summary.loc["ENSEMBLE", col], best[col].max()
    print(f"  {col:16s} ensemble {e:.5f}  best member {b:.5f}  {'ensemble leads' if e > b else 'member leads'}")

# ---- figure, from the same exact mapping ------------------------------------
lb_path = ROOT / "experiments" / "leaderboard_scores.csv"
pairs = []
if lb_path.exists():
    lb = pd.read_csv(lb_path)
    for _, r in lb.iterrows():
        name = str(r["submission"])
        if "blend" in name:
            continue
        rec = submission_record(name) or submission_record(f"hgb_balanced_{name}")
        if rec is None:
            continue
        rid = (json.loads(rec.read_text()).get("validated_run_id") or "")
        dd = load_preds(rid)
        if dd is None:
            continue
        have = [s for s in LONG if s in set(dd["split"])]
        if len(have) < 4:
            continue
        aucs = [roc_auc_score(dd[dd.split == s].label, dd[dd.split == s].pred) for s in have]
        pairs.append({"submission": name, "run_id": rid, "long_mean_auc": float(np.mean(aucs)),
                      "public_lb": float(r["public_lb"])})

if len(pairs) >= 5:
    m = pd.DataFrame(pairs).drop_duplicates("submission")
    rho = spearmanr(m.long_mean_auc, m.public_lb)[0]
    m.round(5).to_csv(OUT / "figure_points.csv", index=False)
    print(f"\n=== figure: {len(m)} single-model submissions with all four long splits logged ===")
    print(m.round(4).to_string(index=False))
    print(f"Spearman = {rho:+.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    ax.scatter(m.long_mean_auc, m.public_lb, s=30, color="#39506b")
    for _, r in m.iterrows():
        if r.public_lb > m.public_lb.quantile(0.75) or r.public_lb < m.public_lb.quantile(0.2):
            ax.annotate(r.submission, (r.long_mean_auc, r.public_lb), fontsize=6.5,
                        textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("validation AUC, mean of four long horizons")
    ax.set_ylabel("public AUC")
    ax.set_title("Validation versus public leaderboard AUC", loc="left", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "figure_validation_vs_lb.png", bbox_inches="tight")
    print(f"saved {OUT / 'figure_validation_vs_lb.png'} and figure_points.csv")
else:
    print(f"\nOnly {len(pairs)} submissions could be mapped to a complete long-split record; "
          "the figure was not rebuilt.")
