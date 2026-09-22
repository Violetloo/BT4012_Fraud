"""Named model factories. Each call returns a NEW unfitted model with every
random_state set, as evaluate() requires.

Stage 3 baselines use fixed settings and no early stopping, so the final model
retrained on all labelled data uses exactly the configuration that was validated
(no number of boosting rounds chosen on the outer validation blocks).
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def logreg():
    """Standardised logistic regression: a simple linear reference."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def hgb():
    """Histogram gradient boosting (scikit-learn), fixed settings, no early stopping."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=20, l2_regularization=0.0,
                                          early_stopping=False, random_state=SEED)


MODELS = {"logreg": logreg, "hgb": hgb}


def get_model_factory(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name]
