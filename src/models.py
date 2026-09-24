"""Named model factories. Each call returns a NEW unfitted model with every
random_state set, as evaluate() requires.

Stage 3 baselines use fixed settings and no early stopping, so a model retrained
on all labelled data uses exactly the configuration that was validated (no number
of boosting rounds chosen on the outer validation blocks). Stage 4 keeps that
rule while comparing model families and class weighting.

LightGBM and XGBoost are optional. If they are not installed, their entries are
absent from MODELS and the comparison script skips them with a note.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


class AutoScalePosWeight(BaseEstimator, ClassifierMixin):
    """Wraps an XGBoost model and sets scale_pos_weight inside each fit.

    The ratio is negatives / positives among the labels passed to fit, so it uses
    only that split's training rows. A ratio computed once over all labelled data
    would include the validation block's labels.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        y = np.asarray(y)
        n_pos = int((y == 1).sum())
        if n_pos == 0:
            raise ValueError("No positive labels in this training split")
        self.ratio_ = float((y == 0).sum() / n_pos)
        self.model_ = clone(self.base).set_params(scale_pos_weight=self.ratio_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# --- linear ---------------------------------------------------------------
def logreg():
    """Standardised logistic regression: a simple linear reference."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def logreg_balanced():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))


# --- scikit-learn trees ---------------------------------------------------
def hgb():
    """Histogram gradient boosting (scikit-learn), fixed settings, no early stopping."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=20, l2_regularization=0.0,
                                          early_stopping=False, random_state=SEED)


def hgb_balanced():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=20, l2_regularization=0.0,
                                          early_stopping=False, class_weight="balanced", random_state=SEED)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  n_jobs=-1, random_state=SEED)


def rf_balanced():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def extratrees():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                n_jobs=-1, random_state=SEED)


MODELS = {"logreg": logreg, "logreg_balanced": logreg_balanced,
          "hgb": hgb, "hgb_balanced": hgb_balanced,
          "rf": rf, "rf_balanced": rf_balanced, "extratrees": extratrees}

# --- optional libraries ---------------------------------------------------
MISSING_LIBRARIES = {}

try:
    from lightgbm import LGBMClassifier

    def _lgbm(**kw):
        # deterministic settings so repeated runs match exactly
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              subsample=1.0, colsample_bytree=1.0, n_jobs=-1, verbose=-1,
                              deterministic=True, force_row_wise=True, random_state=SEED, **kw)

    MODELS["lgbm"] = lambda: _lgbm()
    MODELS["lgbm_balanced"] = lambda: _lgbm(class_weight="balanced")
except ImportError as e:  # pragma: no cover
    MISSING_LIBRARIES["lightgbm"] = str(e)

try:
    from xgboost import XGBClassifier

    def _xgb(**kw):
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=1,
                             subsample=1.0, colsample_bytree=1.0, tree_method="hist",
                             eval_metric="auc", n_jobs=-1, random_state=SEED, **kw)

    MODELS["xgb"] = lambda: _xgb()
    # scale_pos_weight (negatives / positives) is recomputed inside every fit from
    # that split's training labels only; see AutoScalePosWeight above.
    MODELS["xgb_weighted"] = lambda: AutoScalePosWeight(_xgb())
except ImportError as e:  # pragma: no cover
    MISSING_LIBRARIES["xgboost"] = str(e)


def get_model_factory(name):
    if name not in MODELS:
        extra = f" Missing optional libraries: {sorted(MISSING_LIBRARIES)}." if MISSING_LIBRARIES else ""
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}.{extra}")
    return MODELS[name]


# Pairs compared in Stage 4: (unweighted, weighted)
CLASS_WEIGHT_PAIRS = [("logreg", "logreg_balanced"), ("hgb", "hgb_balanced"), ("rf", "rf_balanced"),
                      ("lgbm", "lgbm_balanced"), ("xgb", "xgb_weighted")]
