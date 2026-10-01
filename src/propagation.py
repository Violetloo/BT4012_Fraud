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
PREFIXES = {1: "p1_", 2: "p2_", 3: "p3_"}


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


def build_or_load(train, test, edges, cache_path=CACHE_PATH, rebuild=False, hops=3, verbose=True):
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
