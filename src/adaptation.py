"""Adapting a model to the period it will be scored on.

Both classes here use the FEATURES of later periods (and, at submission time, of the
test rows), never their labels. The competition supplies test.csv, so this is
transductive learning, not leakage — but it is the first thing in this project to use
the target period's feature distribution at all.

CovariateShiftWeighted
    Stage 6 measured how far the feature distributions move: a classifier separates
    early from late periods with AUC 0.961. The standard correction is importance
    weighting. A classifier is trained to tell training rows from target-period rows;
    each training row is then weighted by p / (1 - p), its estimated density ratio, so
    rows that look like the period being predicted count more. Unlike recency
    weighting, which assumes a decay, the weights are measured.

SelfTrainTarget
    Stage 8b pseudo-labelled unlabelled TRAINING rows and failed: they were 97.5%
    pseudo-licit and diluted the rare class. This labels the TARGET period's rows
    instead. The purpose is different: not more data, but adapting the decision
    boundary to the distribution it will be scored on.

Both take the target rows from a registry, so model parameters stay small and
printable, and both infer the target window from the training rows they are given:
target = steps after the last training step, up to target_window steps ahead. In
validation that is the block being predicted; at submission time it is the test period.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

SEED = 42
TARGETS = {}
FIT_LOG = []


def register_target(name, features, steps):
    TARGETS[name] = (features, pd.Series(np.asarray(steps), index=features.index))


def register_default_target(train, test, feature_cols, name="all_rows"):
    """Every row of train and test, by feature columns, with its time step.

    Indices follow the concatenation order, so a training row's index is unchanged.
    """
    both = pd.concat([train[feature_cols + ["time_step"]].reset_index(drop=True),
                      test[feature_cols + ["time_step"]].reset_index(drop=True)], ignore_index=True)
    register_target(name, both[feature_cols], both["time_step"])
    return name


def _target_rows(target, last_train_step, window, columns):
    feats, steps = TARGETS[target]
    mask = (steps > last_train_step) & (steps <= last_train_step + window)
    return feats.loc[mask.values, columns]


def _last_step(target, index):
    _, steps = TARGETS[target]
    return int(steps.loc[index].max())


class CovariateShiftWeighted(BaseEstimator, ClassifierMixin):
    """Weight training rows by how much they resemble the target period."""

    def __init__(self, base="hgb_balanced", target="all_rows", target_window=14, clip=10.0,
                 n_folds=2, random_state=SEED):
        self.base = base
        self.target = target
        self.target_window = target_window
        self.clip = clip
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        k = _last_step(self.target, X.index)
        tgt = _target_rows(self.target, k, self.target_window, list(X.columns))
        w = np.ones(len(X))
        auc_like = np.nan
        if len(tgt) > 50:
            Z = pd.concat([X, tgt])
            is_target = np.r_[np.zeros(len(X)), np.ones(len(tgt))]
            # Cross-fitted probabilities: a row's weight never comes from a model that
            # was fitted on that row, which otherwise inflates the ratios.
            p = np.zeros(len(Z))
            skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
            for tr_idx, te_idx in skf.split(Z, is_target):
                clf = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150,
                                                     early_stopping=False,
                                                     random_state=self.random_state)
                clf.fit(Z.iloc[tr_idx], is_target[tr_idx])
                p[te_idx] = clf.predict_proba(Z.iloc[te_idx])[:, 1]
            ps = np.clip(p[: len(X)], 1e-6, 1 - 1e-6)
            w = ps / (1 - ps)
            w = np.clip(w, 1 / self.clip, self.clip)
            w = w / w.mean()
            auc_like = float(p[len(X):].mean() - p[: len(X)].mean())
        FIT_LOG.append({"model": "covariate_shift", "last_train_step": k, "n_target": int(len(tgt)),
                        "weight_min": float(w.min()), "weight_max": float(w.max()),
                        "weight_p90": float(np.quantile(w, 0.9)), "target_minus_source_p": auc_like})
        self.model_ = get_model_factory(self.base)().fit(X, np.asarray(y), sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class SelfTrainTarget(BaseEstimator, ClassifierMixin):
    """Add confident predictions on the target period's rows, then refit."""

    def __init__(self, base="hgb_balanced", target="all_rows", target_window=14, high=0.99,
                 low=0.01, pseudo_weight=0.5, rounds=1, random_state=SEED):
        self.base = base
        self.target = target
        self.target_window = target_window
        self.high = high
        self.low = low
        self.pseudo_weight = pseudo_weight
        self.rounds = rounds
        self.random_state = random_state

    def fit(self, X, y):
        from src.models import get_model_factory
        k = _last_step(self.target, X.index)
        tgt = _target_rows(self.target, k, self.target_window, list(X.columns))
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
            rounds.append({"n_added": int(len(add_X)), "pseudo_positive": int(add_y.sum())})
        FIT_LOG.append({"model": "self_train_target", "last_train_step": k, "n_target": int(len(tgt)),
                        "rounds": rounds})
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
