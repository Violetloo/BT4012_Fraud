"""Stacking: learn how to combine models instead of fixing the weights by hand.

Every blend so far used weights chosen by hand (50/50, 75/25, equal). Stacking learns
them: inside fit(), chronological blocks are carved out of the TRAINING rows, each base
model is fitted on the earlier part and predicts the held-out block, and a meta-model is
fitted on those out-of-fold predictions. The bases are then refitted on all the training
rows, and the meta-model combines their predictions at scoring time.

Leak-free by construction: the meta-model only ever sees predictions for rows the base
models did not train on, and every one of those rows comes from the training window. The
outer validation block is untouched.

Base predictions are converted to within-set ranks before the meta-model sees them, so
one base's probability scale cannot dominate another's.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

SEED = 42
ROW_STEPS = {}
FIT_LOG = []


def register_steps(steps, name="default"):
    """steps: time step of every row of the training frame, indexed like it."""
    ROW_STEPS[name] = pd.Series(np.asarray(steps.values), index=steps.index)


def _rank01(p):
    return rankdata(p) / (len(p) + 1)


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """Several base models combined by a meta-model fitted on inner out-of-fold predictions."""

    def __init__(self, bases=("hgb_balanced", "lgbm_balanced", "xgb"), steps="default",
                 n_blocks=3, block_len=5, meta_C=1.0, random_state=SEED):
        self.bases = bases
        self.steps = steps
        self.n_blocks = n_blocks
        self.block_len = block_len
        self.meta_C = meta_C
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import MODELS, get_model_factory
        # Optional libraries may be absent; drop bases that are not registered.
        bases = [b for b in self.bases if b in MODELS]
        if len(bases) < 2:
            raise ValueError(f"stacking needs at least two available bases; have {bases} "
                             f"from {list(self.bases)}")
        self.bases_used_ = bases
        steps = ROW_STEPS[self.steps].loc[X.index]
        y = pd.Series(np.asarray(y), index=X.index)
        last = int(steps.max())

        oof_rows, oof_y = [], []
        for b in range(self.n_blocks):
            hi = last - b * self.block_len
            lo = hi - self.block_len + 1
            tr_mask = (steps <= lo - 1).values
            va_mask = ((steps >= lo) & (steps <= hi)).values
            if tr_mask.sum() == 0 or va_mask.sum() == 0:
                continue
            if y[tr_mask].nunique() < 2 or y[va_mask].nunique() < 2:
                continue
            preds = []
            for name in bases:
                m = get_model_factory(name)().fit(X[tr_mask], y[tr_mask].values)
                p = m.predict_proba(X[va_mask])[:, list(m.classes_).index(1)]
                preds.append(_rank01(p))
            oof_rows.append(np.column_stack(preds))
            oof_y.append(y[va_mask].values)

        if not oof_rows:
            raise ValueError("no usable inner blocks for stacking")
        Z = np.vstack(oof_rows)
        z_y = np.concatenate(oof_y)
        self.meta_ = LogisticRegression(C=self.meta_C, max_iter=2000, class_weight="balanced",
                                        random_state=self.random_state).fit(Z, z_y)
        self.models_ = [get_model_factory(n)().fit(X, y.values) for n in bases]
        self.classes_ = np.array([0, 1])
        FIT_LOG.append({"n_train": int(len(X)), "last_train_step": last, "n_meta_rows": int(len(Z)),
                        "bases": list(bases),
                        "meta_coefficients": [float(c) for c in self.meta_.coef_[0]]})
        return self

    def predict_proba(self, X):
        Z = np.column_stack([_rank01(m.predict_proba(X)[:, list(m.classes_).index(1)])
                             for m in self.models_])
        p = self.meta_.predict_proba(Z)[:, list(self.meta_.classes_).index(1)]
        return np.column_stack([1 - p, p])
