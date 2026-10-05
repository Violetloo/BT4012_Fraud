"""Multi-hop feature propagation over the transaction graph.

What this adds over Stage 5. Those neighbour aggregates were one hop and per direction,
computed independently per feature. This propagates the whole feature matrix through a
row-normalised adjacency matrix, and does it twice, so a transaction's representation
includes information from transactions two hops away that it never touches directly.
That multi-hop mixing is the part no model in this project can currently represent, and
it is the mechanism a graph neural network would use.

Coverage is matched as in Stage 5: only "usable" rows (labelled training rows and test
rows) contribute, so a training row's neighbourhood is populated to a similar degree as
a test row's (63% vs 61%).

No labels are involved at any point.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import REPO_ROOT, raw_feature_columns

CACHE_PATH = REPO_ROOT / "data" / "derived" / "propagated_features.pkl"
BUILDER_VERSION = 1
PREFIXES = {1: "p1_", 2: "p2_"}


def _fingerprint(*frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(str(f.shape).encode())
        h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
    return h.hexdigest()[:16]


def build_propagated_features(train, test, edges, hops=2):
    """Return 1-hop and 2-hop propagated versions of every supplied feature, by txId."""
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "label"] + cols],
                       test[["txId"] + cols].assign(label=1.0)], ignore_index=True)
    usable = nodes["label"].notna().values
    ids = nodes["txId"].values
    pos = pd.Series(np.arange(len(ids)), index=ids)

    e = edges.copy()
    e.columns = ["src", "dst"]
    e = e[e["src"].isin(pos.index) & e["dst"].isin(pos.index)]
    si, di = pos.loc[e["src"]].values, pos.loc[e["dst"]].values
    # undirected, and a usable neighbour only contributes if it has features
    keep = usable[si] & usable[di]
    si, di = si[keep], di[keep]
    rows = np.r_[si, di]
    cols_idx = np.r_[di, si]
    A = sparse.coo_matrix((np.ones(len(rows), dtype="float32"), (rows, cols_idx)),
                          shape=(len(ids), len(ids))).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    P = sparse.diags(inv.astype("float32")) @ A  # row-normalised: a mean over neighbours

    X = nodes[cols].to_numpy(dtype="float32")
    out = {}
    H = X
    for hop in range(1, hops + 1):
        H = P @ H
        frame = pd.DataFrame(H, index=pd.Index(ids, name="txId"),
                             columns=[f"{PREFIXES[hop]}{c}" for c in cols])
        # isolated nodes receive zeros; mark them as missing so the model can split on it
        frame[deg == 0] = np.nan
        out[hop] = frame.astype("float32")
    return pd.concat(out.values(), axis=1)


def build_or_load(train, test, edges, cache_path=CACHE_PATH, rebuild=False, hops=2, verbose=True):
    cache_path = Path(cache_path)
    meta = {"inputs": _fingerprint(train, test, edges), "builder_version": BUILDER_VERSION, "hops": hops}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding propagated features (different inputs or settings).")
    feats = build_propagated_features(train, test, edges, hops=hops)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def attach(frame, propagated):
    if not frame["txId"].is_unique:
        raise ValueError("frame has duplicate txId values")
    missing = (~frame["txId"].isin(propagated.index)).sum()
    if missing:
        raise ValueError(f"{missing} rows have no propagated features; rebuild the cache")
    return frame.join(propagated, on="txId")


def propagated_columns(df, hops=(1, 2)):
    prefixes = tuple(PREFIXES[h] for h in hops)
    return [c for c in df.columns if c.startswith(prefixes)]


# ---------------------------------------------------------------------------
# Directed and decayed propagation (Stage 14)
#
# The propagation above symmetrises the graph: a transaction's upstream and
# downstream neighbourhoods are averaged together. In a payment graph those are
# different things — where money came FROM and where it went TO — and laundering
# patterns are directional. These builders keep them apart, and add a
# personalised-PageRank variant that mixes all hop distances with a decay instead
# of using one hop distance at a time.
DIRECTED_CACHE_PATH = REPO_ROOT / "data" / "derived" / "directed_features.pkl"
DIRECTED_BUILDER_VERSION = 1
DIR_PREFIXES = {("in", 2): "din2_", ("in", 3): "din3_", ("in", 4): "din4_",
                ("out", 2): "dout2_", ("out", 3): "dout3_", ("out", 4): "dout4_"}
PPR_PREFIX = "ppr_"


def _usable_matrix(train, test, edges):
    """Shared setup: node table, usable mask, feature matrix and the edge endpoints."""
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "label"] + cols],
                       test[["txId"] + cols].assign(label=1.0)], ignore_index=True)
    usable = nodes["label"].notna().values
    ids = nodes["txId"].values
    pos = pd.Series(np.arange(len(ids)), index=ids)
    e = edges.copy()
    e.columns = ["src", "dst"]
    e = e[e["src"].isin(pos.index) & e["dst"].isin(pos.index)]
    si, di = pos.loc[e["src"]].values, pos.loc[e["dst"]].values
    keep = usable[si] & usable[di]
    X = nodes[cols].to_numpy(dtype="float32")
    return ids, cols, X, si[keep], di[keep], len(ids)


def _row_normalised(rows, cols_idx, n):
    A = sparse.coo_matrix((np.ones(len(rows), "float32"), (rows, cols_idx)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sparse.diags(inv.astype("float32")) @ A, deg


def build_directed_features(train, test, edges, hops=(2, 3, 4)):
    """Propagate separately over incoming and outgoing edges."""
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    out = {}
    for direction, (r, c) in [("in", (di, si)), ("out", (si, di))]:
        P, deg = _row_normalised(r, c, n)
        H = X
        for hop in range(1, max(hops) + 1):
            H = P @ H
            if hop in hops:
                frame = pd.DataFrame(H, index=pd.Index(ids, name="txId"),
                                     columns=[f"{DIR_PREFIXES[(direction, hop)]}{c_}" for c_ in cols])
                frame[deg == 0] = np.nan
                out[(direction, hop)] = frame.astype("float32")
    return pd.concat(out.values(), axis=1)


def build_ppr_features(train, test, edges, alpha=0.15, iters=10):
    """Personalised-PageRank propagation: H <- (1 - alpha) P H + alpha X.

    One block of columns mixes every hop distance with a geometric decay, instead of
    taking hop distances one at a time. alpha controls how fast influence falls off:
    0.15 keeps a long tail, 0.5 stays local.
    """
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    P, deg = _row_normalised(np.r_[si, di], np.r_[di, si], n)
    H = X.copy()
    for _ in range(iters):
        H = (1 - alpha) * (P @ H) + alpha * X
    frame = pd.DataFrame(H, index=pd.Index(ids, name="txId"),
                         columns=[f"{PPR_PREFIX}{c}" for c in cols])
    frame[deg == 0] = np.nan
    return frame.astype("float32")


def build_or_load_directed(train, test, edges, cache_path=DIRECTED_CACHE_PATH, rebuild=False,
                           hops=(2, 3, 4), alpha=0.15, verbose=True):
    """Directed hops and the PageRank block, cached together."""
    cache_path = Path(cache_path)
    meta = {"inputs": _fingerprint(train, test, edges), "builder_version": DIRECTED_BUILDER_VERSION,
            "hops": list(hops), "alpha": alpha}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding directed/PPR features (different inputs or settings).")
    feats = pd.concat([build_directed_features(train, test, edges, hops=hops),
                       build_ppr_features(train, test, edges, alpha=alpha)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def directed_columns(df, directions=("in", "out"), hops=(2, 3, 4)):
    prefixes = tuple(DIR_PREFIXES[(d, h)] for d in directions for h in hops if (d, h) in DIR_PREFIXES)
    return [c for c in df.columns if c.startswith(prefixes)]


def ppr_columns(df):
    return [c for c in df.columns if c.startswith(PPR_PREFIX)]


# ---------------------------------------------------------------------------
# Stage 15: max-propagation and symmetric normalisation
#
# Two things the mean-propagation above cannot express:
#
# 1. MEAN washes out a single extreme neighbour. For fraud the presence of one
#    suspicious transaction nearby matters more than the average of the
#    neighbourhood. Stage 8a already hinted at this: smoothing predictions with the
#    neighbour MAX beat the neighbour mean. These builders propagate the max.
# 2. ROW normalisation (D^-1 A) treats every neighbour of a node equally regardless
#    of how connected that neighbour is. Symmetric normalisation (D^-1/2 A D^-1/2),
#    the standard graph-convolution weighting, down-weights hub neighbours that
#    connect to everything. Hubs are exactly what an exchange looks like here.
MAXPROP_CACHE_PATH = REPO_ROOT / "data" / "derived" / "maxprop_features.pkl"
SYM_CACHE_PATH = REPO_ROOT / "data" / "derived" / "sym_features.pkl"
MAXPROP_VERSION = 1
SYM_VERSION = 1
MAX_PREFIXES = {1: "mx1_", 2: "mx2_"}
SYM_PREFIX = "sym_"


def build_max_features(train, test, edges, hops=(1, 2)):
    """Propagate the MAX over usable neighbours, one and two hops."""
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    pairs = pd.DataFrame({"node": np.r_[si, di], "other": np.r_[di, si]})
    H = pd.DataFrame(X, columns=cols)
    out = {}
    deg = np.bincount(pairs["node"].values, minlength=n)
    for hop in range(1, max(hops) + 1):
        joined = pairs.join(H, on="other")
        H_new = joined.groupby("node")[cols].max()
        H = H_new.reindex(range(n))          # nodes with no usable neighbour -> NaN
        if hop in hops:
            frame = H.copy()
            frame.columns = [f"{MAX_PREFIXES[hop]}{c}" for c in cols]
            frame.index = pd.Index(ids, name="txId")
            out[hop] = frame.astype("float32")
        H = H.fillna(pd.DataFrame(X, columns=cols))   # keep propagating from own values
    return pd.concat(out.values(), axis=1)


def build_symmetric_ppr(train, test, edges, alpha=0.15, iters=10):
    """Personalised PageRank with symmetric normalisation D^-1/2 A D^-1/2."""
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    rows, cols_idx = np.r_[si, di], np.r_[di, si]
    A = sparse.coo_matrix((np.ones(len(rows), "float32"), (rows, cols_idx)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    D = sparse.diags(inv_sqrt.astype("float32"))
    P = D @ A @ D
    H = X.copy()
    for _ in range(iters):
        H = (1 - alpha) * (P @ H) + alpha * X
    frame = pd.DataFrame(H, index=pd.Index(ids, name="txId"),
                         columns=[f"{SYM_PREFIX}{c}" for c in cols])
    frame[deg == 0] = np.nan
    return frame.astype("float32")


def build_or_load_stage15(train, test, edges, rebuild=False, alpha=0.15, verbose=True):
    """Max-propagated and symmetric-PageRank blocks, cached together."""
    cache_path = Path(MAXPROP_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "maxprop_version": MAXPROP_VERSION,
            "sym_version": SYM_VERSION, "alpha": alpha}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding max/symmetric features (different inputs or settings).")
    feats = pd.concat([build_max_features(train, test, edges),
                       build_symmetric_ppr(train, test, edges, alpha=alpha)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def max_columns(df, hops=(1, 2)):
    prefixes = tuple(MAX_PREFIXES[h] for h in hops)
    return [c for c in df.columns if c.startswith(prefixes)]


def symmetric_columns(df):
    return [c for c in df.columns if c.startswith(SYM_PREFIX)]
