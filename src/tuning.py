"""Nested hyperparameter selection: the search runs inside each split's training rows.

An earlier version chose one parameter set on fixed inner blocks (steps 11-15, 16-20,
21-25, 26-30) and then reported outer scores. Three of those blocks are outer fold
validation blocks, so the reported scores reused labels that had influenced the
selection. Evaluating the winner once does not undo that.

NestedTunedClassifier fixes it by doing the search inside fit(): whatever training
rows a split hands it, it carves the last inner blocks out of those rows, scores every
candidate on them, refits the winner on the whole training portion, and predicts. The
outer block is never seen during selection, so the outer score estimates the tuning
PROCEDURE, which is what a submission actually uses.

The row -> time_step mapping is passed in at construction. evaluate() slices frames by
row, so the index of X identifies the rows; time steps are not labels and are known for
every training row.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
BASELINE_PARAMS = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31,
                   "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0}
SPACE = {"learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
         "max_iter": [200, 300, 500, 800],
         "max_leaf_nodes": [15, 31, 63],
         "min_samples_leaf": [10, 20, 50, 100],
         "l2_regularization": [0.0, 0.1, 1.0, 10.0],
         "max_features": [0.5, 0.8, 1.0]}

# Every fit appends {"params": ..., "inner_scores": ..., "n_train": ...} here, so a
# script can report which settings each split selected.
SELECTION_LOG = []


def candidate_params(n_random, seed=SEED, include_baseline=True):
    """The baseline settings plus n_random distinct random draws.

    The baseline is a candidate, so a search can never select settings that scored
    worse than it on the inner blocks.
    """
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    if include_baseline:
        out.append(dict(BASELINE_PARAMS))
        seen.add(tuple(sorted(BASELINE_PARAMS.items())))
    while len(out) - len(seen) + len(seen) < n_random + (1 if include_baseline else 0):
        p = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _make(params, random_state=SEED):
    return HistGradientBoostingClassifier(early_stopping=False, class_weight="balanced",
                                          random_state=random_state, **params)


def inner_scores(params, X, y, steps, n_blocks=2, block_len=5, random_state=SEED):
    """Score one parameter set on the last n_blocks chronological blocks of the rows given."""
    last = int(steps.max())
    scores = []
    for b in range(n_blocks):
        hi = last - b * block_len
        lo = hi - block_len + 1
        tr = steps <= lo - 1
        va = (steps >= lo) & (steps <= hi)
        if tr.sum() == 0 or va.sum() == 0 or y[va.values].min() == y[va.values].max() or y[tr.values].min() == y[tr.values].max():
            continue
        m = _make(params, random_state).fit(X[tr.values], y[tr.values])
        scores.append(roc_auc_score(y[va.values], m.predict_proba(X[va.values])[:, 1]))
    return scores


class NestedTunedClassifier(BaseEstimator, ClassifierMixin):
    """Selects settings inside fit(), using only the rows it is given."""

    def __init__(self, step_lookup=None, candidates=None, n_blocks=2, block_len=5, random_state=SEED):
        self.step_lookup = step_lookup
        self.candidates = candidates
        self.n_blocks = n_blocks
        self.block_len = block_len
        self.random_state = random_state

    def fit(self, X, y):
        steps = self.step_lookup.loc[X.index]
        y = pd.Series(np.asarray(y), index=X.index)
        results = []
        for params in self.candidates:
            s = inner_scores(params, X, y, steps, self.n_blocks, self.block_len, self.random_state)
            results.append({"params": params, "mean_inner_auc": float(np.mean(s)) if s else float("nan"),
                            "inner_scores": s})
        table = pd.DataFrame(results).sort_values("mean_inner_auc", ascending=False)
        self.selection_ = table
        self.best_params_ = table.iloc[0]["params"]
        SELECTION_LOG.append({"n_train": len(X), "train_steps": [int(steps.min()), int(steps.max())],
                              "params": self.best_params_,
                              "mean_inner_auc": float(table.iloc[0]["mean_inner_auc"]),
                              "baseline_inner_auc": float(
                                  table[table["params"].apply(lambda p: p == BASELINE_PARAMS)]["mean_inner_auc"].max()),
                              "inner_scores": table.iloc[0]["inner_scores"]})
        self.model_ = _make(self.best_params_, self.random_state).fit(X, y.values)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
