"""Graph features built from txs_edgelist.csv.

Rules followed here (from Stage 1):
- No neighbour LABEL VALUES are ever used. Only the graph structure, the supplied
  feature values, and whether a neighbour's feature row is usable.
- Test transactions are missing ~39% of their neighbours' feature rows. Those
  missing nodes appear only in test-period components; the hypothesis (Stage 1,
  not established, since their labels and time steps are unobservable) is that they
  are the unlabelled transactions of the test period. To give train and test
  comparable coverage, aggregates are computed over "usable" neighbours only:
  labelled training rows and test rows. This matches mean coverage (about 63% vs
  61% of neighbours, 47% vs 49% missing aggregates among rows used for fitting and
  scoring); matching mean coverage does not prove the missingness patterns match.
  Using a neighbour's label availability is allowed; using its class is not.
- The features that receive max/min/std aggregates are chosen by variance over an
  early window of time steps only (STAT_SELECTION_STEPS), never the whole data set,
  so no later period influences features used in an earlier validation block.
- Degrees come from the complete edge list, including neighbours with no feature
  row, so they are equally complete for train and test.

Column groups (prefixes):
  deg_  degree            in/out/total degree, unique neighbours, self-ratio
  avl_  availability      neighbours with usable feature rows, and their share
  nbm_  neighbour means   mean of each supplied feature over predecessors/successors
  nbs_  neighbour stats   max/min/std over the highest-variance features
  cmp_  component         weakly connected component size within the time step
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import REPO_ROOT, raw_feature_columns

CACHE_PATH = REPO_ROOT / "data" / "derived" / "graph_features.pkl"
BUILDER_VERSION = 2  # bump whenever the feature definitions change
# Variance for choosing the max/min/std features is measured on these steps only.
# Every validation split trains on step 1..k with k >= 15, so this window is always
# historical relative to any split's validation block.
STAT_SELECTION_STEPS = (1, 15)
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


def _frame_fingerprint(*frames):
    """Content hash of the inputs, so a stale cache can be detected."""
    h = hashlib.sha256()
    for f in frames:
        h.update(str(f.shape).encode())
        h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _config(n_stat_features):
    return {"builder_version": BUILDER_VERSION, "n_stat_features": n_stat_features,
            "stat_selection_steps": list(STAT_SELECTION_STEPS)}


def build_graph_features(train, test, edges, n_stat_features=N_STAT_FEATURES):
    """Return a DataFrame of graph features indexed by txId, for train and test rows."""
    edges = edges.copy()
    edges.columns = ["src", "dst"]
    feat_cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "time_step"] + feat_cols], test[["txId", "time_step"] + feat_cols]],
                      ignore_index=True)
    values = nodes.set_index("txId")[feat_cols].astype("float32")
    usable = _usable_ids(train, test)

    # The features that get max/min/std: highest variance within the early window
    # only (see STAT_SELECTION_STEPS). Uses feature values, never labels, and never
    # periods that lie in the future of a validation block.
    early = nodes["time_step"].between(*STAT_SELECTION_STEPS).values
    stat_cols = values[early].var().sort_values(ascending=False).head(n_stat_features).index.tolist()

    out = pd.DataFrame(index=pd.Index(nodes["txId"], name="txId"))

    # deg_: degree from the complete edge list
    in_deg = edges["dst"].value_counts()
    out_deg = edges["src"].value_counts()
    out["deg_in"] = out.index.map(in_deg).fillna(0)
    out["deg_out"] = out.index.map(out_deg).fillna(0)
    out["deg_total"] = out["deg_in"] + out["deg_out"]
    out["deg_in_share"] = out["deg_in"] / out["deg_total"].replace(0, np.nan)
    # (unique-neighbour counts were dropped: the supplied edge list has no duplicate
    #  pairs, so they equalled deg_in and deg_out exactly)

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
    # degree relative to the size of the node's component (not a share within the step)
    out["cmp_degree_over_size"] = out["deg_total"] / out["cmp_size"]

    return out.astype("float32")



def build_or_load(train, test, edges, cache_path=CACHE_PATH, rebuild=False,
                  n_stat_features=N_STAT_FEATURES, verbose=True):
    """Build the features once and cache them, with the cache checked before reuse.

    The cache stores content hashes of train, test and the edge list plus the
    builder version and settings. If any of them differ (different data directory,
    edited edge list, changed builder), the cache is rebuilt instead of silently
    reused: a stale cache cannot be detected by the training-frame hash recorded at
    submission time, because that hash sees the cached values as if they were current.
    """
    cache_path = Path(cache_path)
    meta = {"inputs": _frame_fingerprint(train, test, edges), **_config(n_stat_features)}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        reason = "different inputs or settings" if isinstance(cached, dict) else "older cache format"
        if verbose:
            print(f"Rebuilding graph features ({reason}).")
    feats = build_graph_features(train, test, edges, n_stat_features=n_stat_features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def attach(frame, graph_features):
    """Join graph features onto a train/test frame by txId, checking the join."""
    if not frame["txId"].is_unique:
        raise ValueError("frame has duplicate txId values")
    if not graph_features.index.is_unique:
        raise ValueError("graph features have duplicate txId values")
    missing = (~frame["txId"].isin(graph_features.index)).sum()
    if missing:
        raise ValueError(f"{missing} rows have no graph features; rebuild the cache with --rebuild-features")
    out = frame.join(graph_features, on="txId")
    if len(out) != len(frame):
        raise ValueError("join changed the number of rows")
    return out


def group_columns(df, groups):
    """Columns in df belonging to the named groups, in a stable order."""
    prefixes = tuple(GROUP_PREFIXES[g] for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]
