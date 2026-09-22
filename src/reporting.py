"""Helpers to compare evaluate() results and read or update the experiment log."""
import csv

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.validation import FOLDS, LOG_COLUMNS, LOG_PATH

FOLD_NAMES = [s.name for s in FOLDS]

# Horizon steps where the Stage 3 baseline (hgb_raw_notime_v1) was clearly weaker.
# Used as a diagnostic in every later comparison. Because they were identified from
# a validation result, do not tune models specifically to maximise this number.
HORIZON_WEAK_STEPS = [24, 29, 31, 32]


def weak_step_auc(result, steps=HORIZON_WEAK_STEPS, split="horizon"):
    """AUC over the horizon predictions restricted to the given steps."""
    p = result["predictions"]
    p = p[(p["split"] == split) & p["time_step"].isin(steps)]
    return float(roc_auc_score(p["label"], p["pred"])) if p["label"].nunique() == 2 else float("nan")


def compare_results(base, other, base_label, other_label):
    """Per-split AUCs of two evaluate() results, their difference, and a summary."""
    a, b = base["folds"]["auc"], other["folds"]["auc"]
    table = pd.DataFrame({base_label: a, other_label: b, "delta": b - a})
    d = table.loc[FOLD_NAMES, "delta"]
    summary = {
        "delta_fold_mean": float(d.mean()),
        "folds_improved": int((d > 0).sum()),
        "delta_fold_min": float(b.loc[FOLD_NAMES].min() - a.loc[FOLD_NAMES].min()),
        "delta_horizon": float(table.loc["horizon", "delta"]),
        "delta_weak_steps": weak_step_auc(other) - weak_step_auc(base),
    }
    return table, summary


def read_log():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No experiment log at {LOG_PATH}")
    return pd.read_csv(LOG_PATH)


def update_log_row(run_id, **fields):
    """Set fields (e.g. public_lb, notes) on one logged run, preserving everything else."""
    with open(LOG_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if r["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one log row with run_id {run_id}, found {len(matches)}")
    for k, v in fields.items():
        if k not in LOG_COLUMNS:
            raise ValueError(f"Unknown log column {k}")
        matches[0][k] = v
    with open(LOG_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
