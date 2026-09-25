"""Seed ensembles and model blends, combined by average rank.

AUC depends only on the ordering of predictions, and the models here are calibrated
differently (their mean predictions differ), so probabilities are converted to ranks
before averaging. Rank averaging also stops one model's extreme scores dominating.

Both classes fit every member on the rows they are given, so they work inside
evaluate() with no special handling and can be registered in src/models.py like any
other model.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42


def _rank01(p):
    """Ranks scaled to (0, 1); ties share the average rank."""
    return rankdata(p) / (len(p) + 1)


class SeedEnsemble(BaseEstimator, ClassifierMixin):
    """One model fitted several times with different randomness, combined by average rank.

    Changing only the seed is not enough for histogram gradient boosting: with no row
    or feature subsampling and fewer rows than its binning subsample, it is
    deterministic given the data, so every member would be identical. Diversity is
    therefore injected two ways:
      - feature_fraction: each member considers a random subset of features per split
        (max_features, when the estimator supports it);
      - bootstrap: Poisson(1) sample weights, the standard online-bagging equivalent of
        resampling rows with replacement.
    Both are per-member randomness only; nothing about the data or the labels changes.
    """

    def __init__(self, base="hgb_balanced", n_seeds=10, feature_fraction=0.8, bootstrap=True,
                 random_state=SEED):
        self.base = base
        self.n_seeds = n_seeds
        self.feature_fraction = feature_fraction
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        self.models_ = []
        for i in range(self.n_seeds):
            seed = self.random_state + i
            m = get_model_factory(self.base)()
            params = m.get_params(deep=True)
            updates = {k: seed for k in params if k == "random_state" or k.endswith("__random_state")}
            if self.feature_fraction is not None:
                updates.update({k: self.feature_fraction for k in params
                                if k in ("max_features", "colsample_bytree")
                                or k.endswith("__max_features") or k.endswith("__colsample_bytree")})
            m.set_params(**updates)
            w = None
            if self.bootstrap:
                w = np.random.default_rng(seed).poisson(1.0, size=len(X)).astype(float)
                if w.sum() == 0:
                    w = None
            self.models_.append(m.fit(X, y) if w is None else m.fit(X, y, sample_weight=w))
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        ranks = np.mean([_rank01(m.predict_proba(X)[:, list(m.classes_).index(1)])
                         for m in self.models_], axis=0)
        return np.column_stack([1 - ranks, ranks])


class Blend(BaseEstimator, ClassifierMixin):
    """Several named models combined by weighted average rank."""

    def __init__(self, members=("hgb_balanced", "lgbm_balanced"), weights=None, random_state=SEED):
        self.members = members
        self.weights = weights
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        self.models_ = [get_model_factory(n)().fit(X, y) for n in self.members]
        w = np.ones(len(self.models_)) if self.weights is None else np.asarray(self.weights, dtype=float)
        self.weights_ = w / w.sum()
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        ranks = np.zeros(len(X))
        for w, m in zip(self.weights_, self.models_):
            ranks += w * _rank01(m.predict_proba(X)[:, list(m.classes_).index(1)])
        return np.column_stack([1 - ranks, ranks])


def per_step_rank(pred, time_step):
    """Percentile rank of each prediction within its own time step.

    Pooled AUC ranks every test row against every other, across all 14 steps. If a
    model's scores sit systematically higher in some periods than others, that pooled
    ranking suffers even when the ordering inside each step is good. Normalising per
    step removes those level differences. It uses predictions and time steps only, no
    labels, so it is valid at submission time.
    """
    pred, time_step = np.asarray(pred, dtype=float), np.asarray(time_step)
    out = np.empty_like(pred)
    for s in np.unique(time_step):
        m = time_step == s
        out[m] = _rank01(pred[m])
    return out
