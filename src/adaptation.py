"""Adapting a model to the rows it will be scored on.

The target is the set of rows being scored, with their labels withheld: in validation
that is the split's validation block, at submission time it is the test rows. Nothing
here uses the target's labels.

How the target reaches the model
--------------------------------
evaluate(split_hook=...) hands each split's validation rows to set_target() before the
fit, so validation reproduces the submission setting exactly. An earlier version
inferred the target as "the next 14 steps after the last training step", which for
fold1 meant adapting to steps 16-29 while scoring 16-20, and for fold4 meant including
test-period rows. It also included unlabelled training rows, a population the test file
does not have.

CovariateShiftWeighted
    A classifier is trained to tell training rows from target rows; each training row is
    weighted by its estimated density ratio, so rows resembling the rows being scored
    count more. The ratio is p/(1-p) x n_source/n_target: the sample-size factor would
    cancel under mean normalisation alone, but clipping happens first, so it matters.
    Weights are cross-fitted, and every fit records diagnostics (probability quantiles,
    how many weights hit each clip boundary, the weight spread and the effective sample
    size) so a collapse to constant weights is visible rather than silent.

SelfTrainTarget
    Confident predictions on the target rows are added back as training rows.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

SEED = 42
TARGET = {"features": None, "name": None}
FIT_LOG = []

DOMAIN_MODELS = {
    # deliberately flexible: fits the boundary between periods closely
    "flexible": dict(learning_rate=0.1, max_iter=150, max_leaf_nodes=31, l2_regularization=0.0),
    # deliberately smooth: less able to separate the periods perfectly, which keeps the
    # estimated probabilities away from 0 and 1 and the density ratios usable
    "regularised": dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=8, min_samples_leaf=100,
                        l2_regularization=10.0),
}


def set_target(features):
    """Register the rows being scored (features only, labels withheld)."""
    TARGET["features"] = features


def split_target_hook(feature_cols, name="validation_block"):
    """A split_hook for evaluate(): registers each split's validation rows as the target."""

    def hook(split, train_rows, validation_rows):
        set_target(validation_rows[feature_cols])

    hook.__name__ = f"split_target_hook[{name}]"
    return hook


def _target(columns):
    feats = TARGET["features"]
    if feats is None:
        raise ValueError("no target registered; pass split_hook=split_target_hook(cols) to evaluate, "
                         "or call set_target(test_features) before fitting")
    return feats[list(columns)]


class CovariateShiftWeighted(BaseEstimator, ClassifierMixin):
    """Weight training rows by how much they resemble the rows being scored."""

    def __init__(self, base="hgb_balanced", clip=10.0, n_folds=2, domain_model="flexible",
                 random_state=SEED):
        self.base = base
        self.clip = clip
        self.n_folds = n_folds
        self.domain_model = domain_model
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        tgt = _target(X.columns)
        Z = pd.concat([X, tgt])
        is_target = np.r_[np.zeros(len(X)), np.ones(len(tgt))]

        # Cross-fitted probabilities: a row's weight never comes from a model fitted on it.
        p = np.zeros(len(Z))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, te_idx in skf.split(Z, is_target):
            clf = HistGradientBoostingClassifier(early_stopping=False, random_state=self.random_state,
                                                 **DOMAIN_MODELS[self.domain_model])
            clf.fit(Z.iloc[tr_idx], is_target[tr_idx])
            p[te_idx] = clf.predict_proba(Z.iloc[te_idx])[:, 1]

        ps = np.clip(p[: len(X)], 1e-9, 1 - 1e-9)
        ratio = (ps / (1 - ps)) * (len(X) / max(len(tgt), 1))  # sample-size factor: clipping is next
        lo, hi = 1 / self.clip, self.clip
        w = np.clip(ratio, lo, hi)
        w = w / w.mean()
        ess = float(w.sum() ** 2 / np.sum(w ** 2))
        FIT_LOG.append({
            "model": "covariate_shift", "domain_model": self.domain_model, "clip": self.clip,
            "n_source": int(len(X)), "n_target": int(len(tgt)),
            "domain_auc_like": float(p[len(X):].mean() - p[: len(X)].mean()),
            "source_p_quantiles": [float(q) for q in np.quantile(ps, [0.01, 0.25, 0.5, 0.75, 0.99])],
            "ratio_quantiles": [float(q) for q in np.quantile(ratio, [0.01, 0.5, 0.99])],
            "share_at_clip_floor": float((ratio <= lo).mean()), "share_at_clip_ceiling": float((ratio >= hi).mean()),
            "weight_sd": float(w.std()), "effective_sample_size": ess,
            "ess_fraction": ess / len(X)})
        self.model_ = get_model_factory(self.base)().fit(X, np.asarray(y), sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class SelfTrainTarget(BaseEstimator, ClassifierMixin):
    """Add confident predictions on the rows being scored, then refit."""

    def __init__(self, base="hgb_balanced", high=0.99, low=0.01, pseudo_weight=0.5, rounds=1,
                 random_state=SEED):
        self.base = base
        self.high = high
        self.low = low
        self.pseudo_weight = pseudo_weight
        self.rounds = rounds
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        tgt = _target(X.columns)
        y = np.asarray(y)
        model = get_model_factory(self.base)().fit(X, y)
        rounds = []
        for _ in range(self.rounds):
            if len(tgt) == 0:
                break
            p = model.predict_proba(tgt)[:, list(model.classes_).index(1)]
            keep = (p >= self.high) | (p <= self.low)
            if keep.sum() == 0:
                break
            add_X, add_y = tgt[keep], (p[keep] >= self.high).astype(int)
            w = np.r_[np.ones(len(X)), np.full(len(add_X), self.pseudo_weight)]
            model = get_model_factory(self.base)().fit(pd.concat([X, add_X]), np.r_[y, add_y],
                                                       sample_weight=w)
            rounds.append({"n_added": int(len(add_X)), "pseudo_positive": int(add_y.sum()),
                           "n_target": int(len(tgt))})
        FIT_LOG.append({"model": "self_train_target", "n_source": int(len(X)),
                        "n_target": int(len(tgt)), "rounds": rounds})
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
