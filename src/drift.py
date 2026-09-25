"""Adversarial validation: which features separate earlier periods from later ones.

A classifier is trained to tell "early" rows from "late" rows using the features
alone. If it succeeds, the feature distributions move over time, and the features it
leans on are the drifting ones. Its AUC is a drift score for the feature set as a
whole; its permutation-free importances rank individual features.

The comparison uses steps inside the earliest training window only (default: steps
1-7 vs 8-15). Every validation split trains on steps 1..k with k >= 15, so this
ranking never sees a period that lies in the future of a split's validation block.
Labels are never used; all rows of those steps take part, including unknown ones.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
EARLY_STEPS = (1, 7)
LATE_STEPS = (8, 15)


def adversarial_ranking(frame, feature_cols, early=EARLY_STEPS, late=LATE_STEPS, seed=SEED,
                        n_repeats=3, subsample=20000):
    """Return (auc, importances DataFrame) for separating early from late rows."""
    steps = frame["time_step"]
    early_rows = frame[steps.between(*early)]
    late_rows = frame[steps.between(*late)]
    X = pd.concat([early_rows[feature_cols], late_rows[feature_cols]])
    y = np.r_[np.zeros(len(early_rows)), np.ones(len(late_rows))]
    if subsample and len(X) > subsample:
        idx = np.random.default_rng(seed).choice(len(X), subsample, replace=False)
        X, y = X.iloc[idx], y[idx]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, early_stopping=False,
                                           random_state=seed).fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    imp = permutation_importance(model, X_va, y_va, scoring="roc_auc", n_repeats=n_repeats,
                                 random_state=seed, n_jobs=-1)
    table = (pd.DataFrame({"feature": feature_cols, "drift_importance": imp.importances_mean,
                           "drift_importance_std": imp.importances_std})
             .sort_values("drift_importance", ascending=False).reset_index(drop=True))
    return float(auc), table
