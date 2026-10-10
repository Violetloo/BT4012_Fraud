"""A two-stage model: predict, spread the predictions over the graph, predict again.

Stage 8a smoothed FINAL predictions with one hop of neighbours and gained +0.0006. This
is the stronger version of the same idea: the first model's score and its propagated
value become FEATURES of a second model, so the model learns how much to trust the
neighbourhood rather than being forced to a fixed blend, and the propagation carries
over many hops with a decay.

Leakage control. Scores for training rows are out-of-fold: they come from models fitted
on strictly earlier time steps within the training window. Scores for the rows being
predicted come from a model fitted on the whole training window and never on those rows.
Because every edge lies inside one time step, training rows and validation rows are
never in the same component, so propagation cannot carry a score across the split.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42
CONTEXT = {}


def register_context(train, edges, name="default"):
    """Row index -> txId and time_step, plus the edge list, for the wrapper to use."""
    CONTEXT[name] = {"txId": pd.Series(train["txId"].values, index=train.index),
                     "time_step": pd.Series(train["time_step"].values, index=train.index),
                     "edges": edges}


def register_predict_context(frame, name="default_predict"):
    CONTEXT[name] = {"txId": pd.Series(frame["txId"].values, index=frame.index),
                     "time_step": pd.Series(frame["time_step"].values, index=frame.index),
                     "edges": CONTEXT["default"]["edges"]}


class TwoStagePropagated(BaseEstimator, ClassifierMixin):
    def __init__(self, base="hgb_balanced", context="default", alpha=0.15, iters=10,
                 inner_blocks=2, block_len=5, random_state=SEED):
        self.base = base
        self.context = context
        self.alpha = alpha
        self.iters = iters
        self.inner_blocks = inner_blocks
        self.block_len = block_len
        self.random_state = random_state

    def _augment(self, X, scores_by_tx, edges):
        from src.propagation import propagate_scores
        prop = propagate_scores(scores_by_tx, edges, self.alpha, self.iters)
        out = X.copy()
        out["stage1_score"] = scores_by_tx.to_numpy()
        out["stage1_score_propagated"] = prop.to_numpy()
        out["stage1_score_contrast"] = out["stage1_score"] - out["stage1_score_propagated"]
        return out

    def fit(self, X, y):
        from src.models import get_model_factory
        ctx = CONTEXT[self.context]
        tx = ctx["txId"].loc[X.index]
        steps = ctx["time_step"].loc[X.index]
        y = pd.Series(np.asarray(y), index=X.index)

        # out-of-fold first-stage scores for the training rows
        oof = pd.Series(np.nan, index=X.index)
        last = int(steps.max())
        for b in range(self.inner_blocks):
            hi = last - b * self.block_len
            lo = hi - self.block_len + 1
            tr = (steps <= lo - 1).values
            va = ((steps >= lo) & (steps <= hi)).values
            if tr.sum() == 0 or va.sum() == 0 or y[tr].nunique() < 2:
                continue
            m = get_model_factory(self.base)().fit(X[tr], y[tr].values)
            oof[va] = m.predict_proba(X[va])[:, list(m.classes_).index(1)]
        oof = oof.fillna(float(y.mean()))     # earliest rows have no earlier data

        scores = pd.Series(oof.to_numpy(), index=tx.to_numpy())
        X_aug = self._augment(X, scores, ctx["edges"])
        self.stage1_ = get_model_factory(self.base)().fit(X, y.values)
        self.model_ = get_model_factory(self.base)().fit(X_aug, y.values)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        ctx = CONTEXT.get("default_predict", CONTEXT[self.context])
        tx = ctx["txId"].loc[X.index]
        s = self.stage1_.predict_proba(X)[:, list(self.stage1_.classes_).index(1)]
        X_aug = self._augment(X, pd.Series(s, index=tx.to_numpy()), ctx["edges"])
        return self.model_.predict_proba(X_aug)
