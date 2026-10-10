"""Smoothing predicted scores over the transaction graph, as post-processing.

Stage 5 tested neighbour FEATURES and they did not help. This is different: it blends
each transaction's predicted score with the scores of the transactions it is connected
to. If illicit activity clusters in the graph, a node's neighbours' predictions carry
information its own features do not, and the model never sees that because it scores
each row independently.

Nothing here uses labels, so it is valid at submission time.

Coverage matches between validation and test, which is what makes the analysis a fair
proxy. Only transactions that are being scored can contribute a score: in validation
that is the labelled rows of the block (about 63% of a row's neighbours); at test time
it is the test rows (about 61%). Neighbours with no score are simply skipped.
"""
import numpy as np
import pandas as pd

MODES = ("nbr_mean", "nbr_max", "comp_mean", "ppr")


def _undirected(edges, scored):
    """Edges with both endpoints among the scored transactions, both directions."""
    e = edges.copy()
    e.columns = ["src", "dst"]
    e = e[e["src"].isin(scored) & e["dst"].isin(scored)]
    return pd.concat([e.rename(columns={"src": "node", "dst": "other"}),
                      e.rename(columns={"dst": "node", "src": "other"})], ignore_index=True)


def _components(pairs, nodes):
    """Weakly connected component id per node, over the scored subgraph."""
    idx = {n: i for i, n in enumerate(nodes)}
    parent = np.arange(len(nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(pairs["node"].map(idx).values, pairs["other"].map(idx).values):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return pd.Series([find(i) for i in range(len(nodes))], index=nodes)


def smooth_scores(pred, tx_ids, edges, mode="nbr_mean", alpha=0.2, iters=1):
    """Blend each score with its neighbours' (or component's) scores.

    pred     : predicted probabilities
    tx_ids   : transaction ids, same order as pred
    edges    : the full edge list; only edges between scored transactions are used
    mode     : nbr_mean | nbr_max | comp_mean
    alpha    : weight on the neighbourhood score (0 = unchanged, 1 = neighbours only)
    iters    : repeat the blend, letting information travel more than one hop
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = pd.Series(np.asarray(pred, dtype=float), index=pd.Index(tx_ids, name="txId"))
    pairs = _undirected(edges, set(s.index))
    if pairs.empty:
        return s.values
    if mode == "ppr":
        p0 = s.copy()
        for _ in range(iters):
            joined = pairs.assign(score=pairs["other"].map(s))
            nbr = pd.Series(s.index.map(joined.groupby("node")["score"].mean()), index=s.index)
            s = alpha * p0 + (1 - alpha) * nbr.fillna(s)
        return s.values
    if mode == "comp_mean":
        comp = _components(pairs, list(s.index))
        for _ in range(iters):
            comp_mean = s.groupby(comp).transform("mean")
            s = (1 - alpha) * s + alpha * comp_mean
        return s.values
    agg = "mean" if mode == "nbr_mean" else "max"
    for _ in range(iters):
        joined = pairs.assign(score=pairs["other"].map(s))
        nbr = pd.Series(s.index.map(joined.groupby("node")["score"].agg(agg)), index=s.index)
        s = (1 - alpha) * s + alpha * nbr.fillna(s)
    return s.values
