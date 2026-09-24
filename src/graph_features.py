"""Graph features built from txs_edgelist.csv.

Rules followed here (from Stage 1):
- No neighbour LABEL VALUES are ever used. Only the graph structure, the supplied
  feature values, and whether a neighbour's feature row is usable.
- Test transactions are missing ~39% of their neighbours' feature rows, because the
  unlabelled transactions of the test period are absent from the files. To give
  train and test comparable coverage, aggregates are computed over "usable"
  neighbours only: labelled training rows and test rows (about 63% vs 61% of
  neighbours). Using a neighbour's label availability is allowed; using its class
  is not.
- Degrees come from the complete edge list, including neighbours with no feature
  row, so they are equally complete for train and test.

Column groups (prefixes):
  deg_  degree            in/out/total degree, unique neighbours, self-ratio
  avl_  availability      neighbours with usable feature rows, and their share
  nbm_  neighbour means   mean of each supplied feature over predecessors/successors
  nbs_  neighbour stats   max/min/std over the highest-variance features
  cmp_  component         weakly connected component size within the time step
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import REPO_ROOT, raw_feature_columns

CACHE_PATH = REPO_ROOT / "data" / "derived" / "graph_features.pkl"
GROUP_PREFIXES = {"deg": "deg_", "avail": "avl_", "nbrmean": "nbm_", "nbrstats": "nbs_", "comp": "cmp_"}
N_STAT_FEATURES = 20  # how many features get max/min/std aggregates


def _usable_ids(train, test):
    """Rows whose features may enter aggregates: labelled train rows and test rows."""
    return set(train.loc[train["label"].notna(), "txId"]) | set(test["txId"])


def _directional_aggregates(edges, values, usable, stat_cols):
    """Aggregate neighbour features in both directions.

    edges: columns src, dst. values: DataFrame of features indexed by txId.
    Predecessor features of a node come from edges pointing into it, successor
    features from edges leaving it. Only usable neighbours contribute.
    """
    out = []
    for direction, node_col, nbr_col in [("pred", "dst", "src"), ("succ", "src", "dst")]:
        e = edges[edges[nbr_col].isin(usable)]
        joined = e[[node_col, nbr_col]].join(values, on=nbr_col).drop(columns=nbr_col)
        g = joined.groupby(node_col)
        mean = g.mean()
        mean.columns = [f"nbm_{direction}_{c}" for c in mean.columns]
        stats = g[stat_cols].agg(["max", "min", "std"])
        stats.columns = [f"nbs_{direction}_{c}_{s}" for c, s in stats.columns]
        count = g.size().rename(f"avl_{direction}_n_usable")
        out += [mean, stats, count]
    return out


def _component_table(edges, usable):
    """Weakly connected components over the complete edge list.

    Every edge with two known endpoints lies inside one time step (Stage 1), so
    components do not cross steps. Nodes without a feature row still join their
    component, which is why component size can exceed the number of usable rows.
    Returns a DataFrame indexed by txId with the component size and the share of
    its nodes that have usable feature rows.
    """
    nodes = pd.unique(pd.concat([edges["src"], edges["dst"]]))
    idx = {n: i for i, n in enumerate(nodes)}
    parent = np.arange(len(nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(edges["src"].map(idx).values, edges["dst"].map(idx).values):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    roots = pd.Series([find(i) for i in range(len(nodes))], index=nodes, name="root")
    df = pd.DataFrame({"root": roots.values, "usable": roots.index.isin(usable)}, index=roots.index)
    size = df.groupby("root").size()
    share = df.groupby("root")["usable"].mean()
    return pd.DataFrame({"cmp_size": df["root"].map(size), "cmp_usable_share": df["root"].map(share)},
                        index=df.index)


def build_graph_features(train, test, edges, n_stat_features=N_STAT_FEATURES):
    """Return a DataFrame of graph features indexed by txId, for train and test rows."""
    edges = edges.copy()
    edges.columns = ["src", "dst"]
    feat_cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "time_step"] + feat_cols], test[["txId", "time_step"] + feat_cols]],
                      ignore_index=True)
    values = nodes.set_index("txId")[feat_cols].astype("float32")
    usable = _usable_ids(train, test)

    # the features that get max/min/std: highest variance across all supplied rows.
    # Variance uses feature values only, never labels.
    stat_cols = values.var().sort_values(ascending=False).head(n_stat_features).index.tolist()

    out = pd.DataFrame(index=pd.Index(nodes["txId"], name="txId"))

    # deg_: degree from the complete edge list
    in_deg = edges["dst"].value_counts()
    out_deg = edges["src"].value_counts()
    uniq_in = edges.drop_duplicates(["src", "dst"])["dst"].value_counts()
    uniq_out = edges.drop_duplicates(["src", "dst"])["src"].value_counts()
    out["deg_in"] = out.index.map(in_deg).fillna(0)
    out["deg_out"] = out.index.map(out_deg).fillna(0)
    out["deg_total"] = out["deg_in"] + out["deg_out"]
    out["deg_unique_in"] = out.index.map(uniq_in).fillna(0)
    out["deg_unique_out"] = out.index.map(uniq_out).fillna(0)
    out["deg_in_share"] = out["deg_in"] / out["deg_total"].replace(0, np.nan)

    # avl_: how much of the neighbourhood has usable feature rows
    out["avl_n_neighbours"] = out["deg_total"]

    aggs = _directional_aggregates(edges, values, usable, stat_cols)
    for a in aggs:
        out = out.join(a)
    for d, col in [("pred", "deg_in"), ("succ", "deg_out")]:
        n = out[f"avl_{d}_n_usable"].fillna(0)
        out[f"avl_{d}_n_usable"] = n
        out[f"avl_{d}_share_usable"] = n / out[col].replace(0, np.nan)
    out["avl_n_usable"] = out["avl_pred_n_usable"] + out["avl_succ_n_usable"]
    out["avl_share_usable"] = out["avl_n_usable"] / out["deg_total"].replace(0, np.nan)

    # cmp_: component size within the time step
    comp = _component_table(edges, usable)
    out["cmp_size"] = out.index.map(comp["cmp_size"]).fillna(1)
    out["cmp_usable_share"] = out.index.map(comp["cmp_usable_share"]).fillna(1.0)
    out["cmp_degree_share"] = out["deg_total"] / out["cmp_size"]

    return out.astype("float32")



def build_or_load(train, test, edges, cache_path=CACHE_PATH, rebuild=False):
    """Build the features once and cache them; later calls read the cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)
    feats = build_graph_features(train, test, edges)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_pickle(cache_path)
    return feats


def attach(frame, graph_features):
    """Join graph features onto a train/test frame by txId."""
    return frame.join(graph_features, on="txId")


def group_columns(df, groups):
    """Columns in df belonging to the named groups, in a stable order."""
    prefixes = tuple(GROUP_PREFIXES[g] for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]
