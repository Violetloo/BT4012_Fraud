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
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.ensembles import Blend, FeatureSetBlend, SeedEnsemble
from src.adaptation import CovariateShiftWeighted, SelfTrainTarget
from src.pseudo import PseudoLabel, PUWeakNegatives
from src.stacking import StackedEnsemble
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


# --- ensembles (src/ensembles.py) -----------------------------------------
def hgb_balanced_seed10():
    """hgb_balanced fitted with 10 seeds, combined by average rank."""
    return SeedEnsemble(base="hgb_balanced", n_seeds=10, random_state=SEED)


def blend_hgb_lgbm():
    return Blend(members=("hgb_balanced", "lgbm_balanced"), weights=(0.5, 0.5), random_state=SEED)


def blend_hgb75_lgbm25():
    return Blend(members=("hgb_balanced", "lgbm_balanced"), weights=(0.75, 0.25), random_state=SEED)


def blend_hgb_xgb():
    return Blend(members=("hgb_balanced", "xgb"), weights=(0.5, 0.5), random_state=SEED)


def blend_hgb_lgbm_xgb():
    return Blend(members=("hgb_balanced", "lgbm_balanced", "xgb"), weights=(1, 1, 1), random_state=SEED)


def blend_seed10_lgbm():
    return Blend(members=("hgb_balanced_seed10", "lgbm_balanced"), weights=(0.75, 0.25), random_state=SEED)


def hgb_slow():
    """Slow-and-shallow: lr 0.02, 1500 iterations, 15 leaves, 50 samples per leaf.
    The only region of the search space that beat the defaults, and it improves all four
    long-horizon splits."""
    return HistGradientBoostingClassifier(learning_rate=0.02, max_iter=1500, max_leaf_nodes=15,
                                          min_samples_leaf=50, early_stopping=False,
                                          class_weight="balanced", random_state=SEED)


def hgb_bins127():
    """127 histogram bins instead of 255: fewer split points, less sensitivity to fine
    distinctions that shift between periods."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=20, max_bins=127, early_stopping=False,
                                          class_weight="balanced", random_state=SEED)


MODELS = {"logreg": logreg, "logreg_balanced": logreg_balanced,
          "hgb": hgb, "hgb_balanced": hgb_balanced, "hgb_bins127": hgb_bins127, "hgb_slow": hgb_slow,
          "rf": rf, "rf_balanced": rf_balanced, "extratrees": extratrees,
          "hgb_balanced_seed10": hgb_balanced_seed10}

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


# --- pseudo-labelling (src/pseudo.py) -------------------------------------
# The unlabelled pool is registered by name before use (register_default_pool), so a
# model's parameters stay small and printable.
def pseudo_hgb():
    """Confident pseudo-labels (p >= 0.99 or <= 0.01), full weight, one round."""
    return PseudoLabel(base="hgb_balanced", high=0.99, low=0.01, pseudo_weight=1.0, rounds=1,
                       random_state=SEED)


def pseudo_hgb_w05():
    """Same, but pseudo-labelled rows count half as much as real ones."""
    return PseudoLabel(base="hgb_balanced", high=0.99, low=0.01, pseudo_weight=0.5, rounds=1,
                       random_state=SEED)


def pseudo_hgb_loose():
    """A looser confidence band (0.95 / 0.05), half weight."""
    return PseudoLabel(base="hgb_balanced", high=0.95, low=0.05, pseudo_weight=0.5, rounds=1,
                       random_state=SEED)


def pseudo_hgb_2rounds():
    """Two rounds of confident pseudo-labelling at half weight."""
    return PseudoLabel(base="hgb_balanced", high=0.99, low=0.01, pseudo_weight=0.5, rounds=2,
                       random_state=SEED)


MODELS.update({"pseudo_hgb": pseudo_hgb, "pseudo_hgb_w05": pseudo_hgb_w05,
               "pseudo_hgb_loose": pseudo_hgb_loose, "pseudo_hgb_2rounds": pseudo_hgb_2rounds})

# --- adapting to the rows being scored (src/adaptation.py) ----------------
# These need a target: pass split_hook=split_target_hook(cols) to evaluate(), or call
# set_target(test_features) before fitting for a submission.
def shift_hgb():
    """Training rows weighted by resemblance to the rows being scored (clip 10)."""
    return CovariateShiftWeighted(base="hgb_balanced", clip=10.0, domain_model="flexible",
                                  random_state=SEED)


def shift_hgb_clip3():
    return CovariateShiftWeighted(base="hgb_balanced", clip=3.0, domain_model="flexible",
                                  random_state=SEED)


def shift_hgb_reg():
    """Same, with a deliberately smooth domain classifier so the ratios stay usable."""
    return CovariateShiftWeighted(base="hgb_balanced", clip=10.0, domain_model="regularised",
                                  random_state=SEED)


def shift_hgb_reg_clip3():
    return CovariateShiftWeighted(base="hgb_balanced", clip=3.0, domain_model="regularised",
                                  random_state=SEED)


def selftrain_target():
    return SelfTrainTarget(base="hgb_balanced", high=0.99, low=0.01, pseudo_weight=0.5, rounds=1,
                           random_state=SEED)


MODELS.update({"shift_hgb": shift_hgb, "shift_hgb_clip3": shift_hgb_clip3,
               "shift_hgb_reg": shift_hgb_reg, "shift_hgb_reg_clip3": shift_hgb_reg_clip3,
               "selftrain_target": selftrain_target})

# --- blends across feature sets (views registered by the running script) ---
def fsblend_cmp_avl():
    return FeatureSetBlend(base="hgb_balanced", views=("raw_cmp", "raw_avl"), random_state=SEED)


def fsblend_cmp_avl_all():
    return FeatureSetBlend(base="hgb_balanced", views=("raw_cmp", "raw_avl", "raw_all"),
                           random_state=SEED)


def fsblend_cmp_avl_notime():
    return FeatureSetBlend(base="hgb_balanced", views=("raw_cmp", "raw_avl", "raw_notime"),
                           random_state=SEED)


def fsblend_cmp2_avl1():
    return FeatureSetBlend(base="hgb_balanced", views=("raw_cmp", "raw_avl"), weights=(2, 1),
                           random_state=SEED)


def fsblend_cmp_avl_drop5():
    return FeatureSetBlend(base="hgb_balanced", views=("raw_cmp", "raw_avl", "raw_cmp_drop5"),
                           random_state=SEED)


MODELS.update({"fsblend_cmp_avl": fsblend_cmp_avl, "fsblend_cmp_avl_all": fsblend_cmp_avl_all,
               "fsblend_cmp_avl_notime": fsblend_cmp_avl_notime, "fsblend_cmp2_avl1": fsblend_cmp2_avl1,
               "fsblend_cmp_avl_drop5": fsblend_cmp_avl_drop5})

# --- stacking and positive-unlabelled learning (Stage 12) -----------------
def pu_w01():
    """Every unlabelled row added as a negative at weight 0.1."""
    return PUWeakNegatives(base="hgb_balanced", unlabelled_weight=0.1, random_state=SEED)


def pu_w005():
    return PUWeakNegatives(base="hgb_balanced", unlabelled_weight=0.05, random_state=SEED)


def pu_w03():
    return PUWeakNegatives(base="hgb_balanced", unlabelled_weight=0.3, random_state=SEED)


def stack_hgb_lgbm():
    return StackedEnsemble(bases=("hgb_balanced", "lgbm_balanced"), random_state=SEED)


def stack_three():
    return StackedEnsemble(bases=("hgb_balanced", "lgbm_balanced", "rf_balanced"), random_state=SEED)


def stack_four():
    return StackedEnsemble(bases=("hgb_balanced", "lgbm_balanced", "xgb", "rf_balanced"),
                           random_state=SEED)


MODELS.update({"pu_w01": pu_w01, "pu_w005": pu_w005, "pu_w03": pu_w03,
               "stack_hgb_lgbm": stack_hgb_lgbm, "stack_three": stack_three,
               "stack_four": stack_four})

# Blends are registered only when their members exist (LightGBM/XGBoost are optional).
for _name, _factory, _needs in [("blend_hgb_lgbm", blend_hgb_lgbm, ["lgbm_balanced"]),
                                ("blend_hgb75_lgbm25", blend_hgb75_lgbm25, ["lgbm_balanced"]),
                                ("blend_hgb_xgb", blend_hgb_xgb, ["xgb"]),
                                ("blend_hgb_lgbm_xgb", blend_hgb_lgbm_xgb, ["lgbm_balanced", "xgb"]),
                                ("blend_seed10_lgbm", blend_seed10_lgbm, ["lgbm_balanced"])]:
    if all(n in MODELS for n in _needs):
        MODELS[_name] = _factory


if "xgb" in MODELS:
    class RankXGB(BaseEstimator, ClassifierMixin):
        """XGBoost with a pairwise ranking objective instead of log loss.

        Every model so far optimises log loss and is then scored on AUC. A pairwise
        ranking objective optimises ordering directly. Scores are mapped to (0, 1) by
        rank, which leaves AUC unchanged and keeps the submission format valid.
        Groups: one group of all rows, or one group per time step.
        """

        def __init__(self, per_step_groups=False, n_estimators=300, learning_rate=0.05,
                     max_depth=6, step_lookup=None, random_state=SEED):
            self.per_step_groups = per_step_groups
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.step_lookup = step_lookup
            self.random_state = random_state

        def fit(self, X, y):
            import numpy as np
            from xgboost import XGBRanker
            y = np.asarray(y)
            if self.per_step_groups and self.step_lookup is not None:
                steps = self.step_lookup.loc[X.index].values
                order = np.argsort(steps, kind="stable")
                X, y, steps = X.iloc[order], y[order], steps[order]
                group = pd.Series(steps).value_counts(sort=False).reindex(pd.unique(steps)).values
            else:
                group = [len(X)]
            self.model_ = XGBRanker(objective="rank:pairwise", n_estimators=self.n_estimators,
                                    learning_rate=self.learning_rate, max_depth=self.max_depth,
                                    tree_method="hist", n_jobs=-1, random_state=self.random_state)
            self.model_.fit(X, y, group=group)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            import numpy as np
            from scipy.stats import rankdata
            r = rankdata(self.model_.predict(X)) / (len(X) + 1)
            return np.column_stack([1 - r, r])

    MODELS["rank_xgb"] = lambda: RankXGB(per_step_groups=False, random_state=SEED)


def get_model_factory(name):
    if name not in MODELS:
        extra = f" Missing optional libraries: {sorted(MISSING_LIBRARIES)}." if MISSING_LIBRARIES else ""
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}.{extra}")
    return MODELS[name]


# Pairs compared in Stage 4: (unweighted, weighted)
CLASS_WEIGHT_PAIRS = [("logreg", "logreg_balanced"), ("hgb", "hgb_balanced"), ("rf", "rf_balanced"),
                      ("lgbm", "lgbm_balanced"), ("xgb", "xgb_weighted")]


# --- Stage 18: two-stage propagated-score model ---------------------------
from src.two_stage import TwoStagePropagated  # noqa: E402


def two_stage_ppr():
    return TwoStagePropagated(base="hgb_balanced", alpha=0.15, iters=10, random_state=SEED)


def two_stage_ppr_local():
    return TwoStagePropagated(base="hgb_balanced", alpha=0.5, iters=10, random_state=SEED)


MODELS.update({"two_stage_ppr": two_stage_ppr, "two_stage_ppr_local": two_stage_ppr_local})
