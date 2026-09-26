"""Pseudo-labelling: use the 110,537 unlabelled training rows.

About 78% of train.csv has no label. A first model scores those rows; the most
confident predictions are added to the training set with their predicted labels, and
the model is refitted.

Keeping it leak-free
--------------------
The pool is restricted, inside every fit, to unlabelled rows whose time step is at or
before the last step of the training rows it was given. A split validating steps 26-30
therefore never sees unlabelled rows from those steps, in any form. Test rows are never
in the pool at all (their steps are later than any training window).

Known risk
----------
Stage 1 found unlabelled rows differ from labelled ones (median KS 0.14 across
features). Confident predictions on them may reflect the model's existing biases rather
than new information, which is exactly how pseudo-labelling degrades a model. That is
why the added rows can be down-weighted and why the usual acceptance rule still applies.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42

# Pools are registered by name so that a model's parameters stay small and printable
# (a DataFrame in get_params() would make every log entry unreadable).
POOLS = {}
ROW_STEPS = {}
FIT_LOG = []


def register_pool(name, features, pool_steps, row_steps):
    """features/pool_steps: the unlabelled rows and their time steps.
    row_steps: time step of every row of the training frame, indexed like it."""
    POOLS[name] = (features, pd.Series(np.asarray(pool_steps), index=features.index))
    ROW_STEPS[name] = pd.Series(np.asarray(row_steps.values), index=row_steps.index)


def register_default_pool(train, feature_cols, name="train_unlabelled"):
    """Register the unlabelled rows of train.csv, restricted to the given features."""
    pool = train[train["label"].isna()]
    register_pool(name, pool[feature_cols], pool["time_step"], train["time_step"])
    return name


class PseudoLabel(BaseEstimator, ClassifierMixin):
    """Fit, label the confident unlabelled rows, refit."""

    def __init__(self, base="hgb_balanced", pool="train_unlabelled", high=0.99, low=0.01,
                 pseudo_weight=1.0, rounds=1, max_add=None, random_state=SEED):
        self.base = base
        self.pool = pool
        self.high = high
        self.low = low
        self.pseudo_weight = pseudo_weight
        self.rounds = rounds
        self.max_add = max_add
        self.random_state = random_state

    def _new(self):
        from src.models import get_model_factory
        return get_model_factory(self.base)()

    def fit(self, X, y):
        if self.pool not in POOLS:
            raise ValueError(f"pool '{self.pool}' is not registered; call register_default_pool first")
        pool_X, pool_steps = POOLS[self.pool]
        steps = pool_steps.reindex(pool_X.index)
        # Only unlabelled rows from the training window: never the validation period.
        last_train_step = int(ROW_STEPS[self.pool].loc[X.index].max())
        pool_X = pool_X[steps <= last_train_step]
        pool_X = pool_X[X.columns]

        y = np.asarray(y)
        model = self._new().fit(X, y)
        added = []
        for _ in range(self.rounds):
            if len(pool_X) == 0:
                break
            p = model.predict_proba(pool_X)[:, list(model.classes_).index(1)]
            pos, neg = p >= self.high, p <= self.low
            take = pool_X[pos | neg]
            take_y = np.where(p[pos | neg] >= self.high, 1, 0)
            if self.max_add is not None and len(take) > self.max_add:
                order = np.argsort(-np.abs(p[pos | neg] - 0.5))[: self.max_add]
                take, take_y = take.iloc[order], take_y[order]
            if len(take) == 0:
                break
            X_aug = pd.concat([X, take])
            y_aug = np.r_[y, take_y]
            w = np.r_[np.ones(len(X)), np.full(len(take), self.pseudo_weight)]
            model = self._new().fit(X_aug, y_aug, sample_weight=w)
            added.append({"n_added": int(len(take)), "n_pseudo_positive": int(take_y.sum())})
        FIT_LOG.append({"n_train": int(len(X)), "last_train_step": last_train_step,
                        "pool_available": int(len(pool_X)), "rounds": added})
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
