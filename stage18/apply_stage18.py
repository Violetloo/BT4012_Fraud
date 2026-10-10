"""Stage 18 patch: neighbourhood dispersion features and a two-stage propagated-score model.

Run from the repo root:  python3 stage18/apply_stage18.py
Appends to src/propagation.py, adds src/two_stage.py, and registers both in
src/features.py, src/frames.py and src/models.py. Safe to run twice.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROP_CODE = '''

# ---------------------------------------------------------------------------
# Stage 18: neighbourhood DISPERSION, not just its average.
#
# Every propagated block so far is a weighted mean, so it describes where the
# neighbourhood sits. None describes how spread out it is, or how far the transaction
# itself lies from it in units of that spread. Both are natural fraud signals: mixing
# services sit among highly varied counterparties, and a transaction that is several
# neighbourhood standard deviations from its neighbours is unusual in a way a mean
# cannot express.
#
#   var  = P(x^2) - (P x)^2          dispersion of the neighbourhood
#   z    = (x - P x) / sqrt(var)     the node's distance from it, in those units
DISPERSION_CACHE_PATH = REPO_ROOT / "data" / "derived" / "dispersion_features.pkl"
DISPERSION_VERSION = 1
VAR_PREFIX = "nvar_"
Z_PREFIX = "nz_"
MIN_NEIGHBOURS_FOR_Z = 3     # below this the spread is not estimable


def build_dispersion_features(train, test, edges):
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    P, deg = _row_normalised(np.r_[si, di], np.r_[di, si], n)
    mean = P @ X
    var = np.maximum((P @ (X ** 2)) - mean ** 2, 0.0)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 1e-6, (X - mean) / sd, np.nan)
    z = np.clip(z, -20, 20)

    index = pd.Index(ids, name="txId")
    var_frame = pd.DataFrame(var, index=index, columns=[f"{VAR_PREFIX}{c}" for c in cols])
    z_frame = pd.DataFrame(z, index=index, columns=[f"{Z_PREFIX}{c}" for c in cols])
    var_frame[deg == 0] = np.nan
    z_frame[deg < MIN_NEIGHBOURS_FOR_Z] = np.nan
    return pd.concat([var_frame, z_frame], axis=1).astype("float32")


def build_or_load_dispersion(train, test, edges, rebuild=False, verbose=True):
    cache_path = Path(DISPERSION_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "version": DISPERSION_VERSION}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding dispersion features (different inputs).")
    feats = build_dispersion_features(train, test, edges)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def dispersion_columns(df, kinds=("var", "z")):
    prefixes = tuple({"var": VAR_PREFIX, "z": Z_PREFIX}[k] for k in kinds)
    return [c for c in df.columns if c.startswith(prefixes)]


def propagate_scores(scores, edges, alpha=0.15, iters=10):
    """Spread a vector of per-transaction scores over the graph (PageRank style).

    `scores` is indexed by txId. Only edges between the supplied transactions are used,
    so a validation block's scores never mix with the training rows' scores.
    """
    ids = list(scores.index)
    pos = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    e = edges.copy()
    e.columns = ["src", "dst"]
    e = e[e["src"].isin(pos.index) & e["dst"].isin(pos.index)]
    si, di = pos.loc[e["src"]].values, pos.loc[e["dst"]].values
    P, deg = _row_normalised(np.r_[si, di], np.r_[di, si], len(ids))
    x = scores.to_numpy("float32")[:, None]
    H = x.copy()
    for _ in range(iters):
        H = (1 - alpha) * (P @ H) + alpha * x
    out = pd.Series(H.ravel(), index=scores.index)
    out[deg == 0] = scores[deg == 0]
    return out
'''

TWO_STAGE = '''"""A two-stage model: predict, spread the predictions over the graph, predict again.

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
'''

FEATURES_BLOCK = '''# Stage 18: neighbourhood dispersion (how varied the neighbourhood is, and how far the
# transaction sits from it in those units).
NEEDS_DISPERSION = {"disp_cmp", "ppr_disp_cmp", "ppr_z_cmp"}
FEATURE_SETS["disp_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                            + dispersion_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_disp_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + dispersion_columns(df)
                                + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_z_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + ppr_columns(df) + dispersion_columns(df, kinds=("z",))
                             + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_disp_cmp", "ppr_z_cmp"}
'''

FRAMES_BLOCK = '''    if sets & NEEDS_DISPERSION:
        d2 = propagation.build_or_load_dispersion(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, d2), propagation.attach(test, d2)
        if verbose:
            print(f"Attached {d2.shape[1]} dispersion columns")
'''

MODELS_BLOCK = '''

# --- Stage 18: two-stage propagated-score model ---------------------------
from src.two_stage import TwoStagePropagated  # noqa: E402


def two_stage_ppr():
    return TwoStagePropagated(base="hgb_balanced", alpha=0.15, iters=10, random_state=SEED)


def two_stage_ppr_local():
    return TwoStagePropagated(base="hgb_balanced", alpha=0.5, iters=10, random_state=SEED)


MODELS.update({"two_stage_ppr": two_stage_ppr, "two_stage_ppr_local": two_stage_ppr_local})
'''


def patch(path, marker, fn, label):
    p = ROOT / path
    if not p.exists():
        print(f"SKIP {label}: {path} missing")
        return False
    s = p.read_text()
    if marker in s:
        print(f"ok   {label}: already present")
        return True
    new = fn(s)
    if new is None:
        print(f"FAIL {label}: anchor not found")
        return False
    p.write_text(new)
    print(f"done {label}")
    return True


ok = True
ok &= patch("src/propagation.py", "build_dispersion_features", lambda s: s + PROP_CODE,
            "propagation: dispersion + score propagation")

two_stage_path = ROOT / "src" / "two_stage.py"
if two_stage_path.exists():
    print("ok   two_stage.py: already present")
else:
    two_stage_path.write_text(TWO_STAGE)
    print("done two_stage.py written")


def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "dispersion_columns" not in s:
        if "from src.propagation import (" in s:
            s = s.replace("from src.propagation import (", "from src.propagation import (dispersion_columns, ", 1)
        else:
            s = s.replace("from src.propagation import ", "from src.propagation import dispersion_columns, ", 1)
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    return s.replace("NEEDS_GRAPH |= {", 'NEEDS_GRAPH |= {"disp_cmp", "ppr_disp_cmp", "ppr_z_cmp", ', 1)


ok &= patch("src/features.py", "NEEDS_DISPERSION", _features, "features: dispersion sets")


def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_DISPERSION, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)


ok &= patch("src/frames.py", "NEEDS_DISPERSION", _frames, "frames: dispersion loader")
ok &= patch("src/models.py", "two_stage_ppr", lambda s: s + MODELS_BLOCK, "models: two-stage model")
print("\nAll patches applied." if ok else "\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
