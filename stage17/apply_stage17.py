"""Stage 17 patch: propagate over more of the graph than before.

Run from the repo root:  python3 stage17/apply_stage17.py
Appends to src/propagation.py and inserts into src/features.py and src/frames.py.
Safe to run twice; every change is checked first.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROP_CODE = '''

# ---------------------------------------------------------------------------
# Stage 17: which nodes take part in propagation.
#
# Everything so far propagated over "usable" rows only: labelled training rows and test
# rows. That rule came from Stage 5, where it matched neighbourhood COVERAGE between
# periods for one-hop aggregates. For multi-hop propagation it is much more costly than
# it looks, and the cost was never measured:
#
#   * 110,537 unlabelled training rows have full features and are simply dropped. They
#     are not just missing from the averages, they BREAK PATHS: a two-hop route through
#     an unlabelled transaction disappears entirely.
#   * 46,668 graph nodes have no feature row at all, and 24,245 test-incident edges lead
#     to them. Dropping them severs exactly the paths that connect test transactions.
#
# Two settings are added here:
#   population="all_features"  unlabelled training rows take part (they have features)
#   relay=True                 featureless nodes stay in the graph as pass-through: they
#                              hold no features of their own and never restart, but
#                              feature mass flows through them, reconnecting paths.
#
# Neither uses labels. The coverage asymmetry Stage 5 worried about is real, so this is
# a measurement, not an assumption: compare against ppr_cmp on the usual splits.
POPULATION_CACHE_PATH = REPO_ROOT / "data" / "derived" / "population_features.pkl"
POPULATION_VERSION = 1
POP_PREFIXES = {("all_features", False): "pa_", ("usable", True): "pr_",
                ("all_features", True): "par_"}


def _population_ppr(train, test, edges, population, relay, alpha=0.15, iters=10):
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "label"] + cols],
                       test[["txId"] + cols].assign(label=1.0)], ignore_index=True)
    if population == "usable":
        takes_part = nodes["label"].notna().values
    elif population == "all_features":
        takes_part = np.ones(len(nodes), bool)      # every row that has features
    else:
        raise ValueError(population)

    ids = list(nodes["txId"].values)
    X_known = nodes[cols].to_numpy("float32")
    e = edges.copy()
    e.columns = ["src", "dst"]
    if relay:
        extra = sorted((set(e["src"]) | set(e["dst"])) - set(ids))
        ids = ids + extra
        X = np.vstack([X_known, np.zeros((len(extra), len(cols)), "float32")])
        has_features = np.r_[takes_part, np.zeros(len(extra), bool)]
    else:
        X = X_known
        has_features = takes_part
    pos = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
    e = e[e["src"].isin(pos.index) & e["dst"].isin(pos.index)]
    si, di = pos.loc[e["src"]].values, pos.loc[e["dst"]].values
    # an edge is usable if both ends are in the graph we are building; nodes that do not
    # take part contribute nothing but, when relay=True, still pass values along
    keep = (has_features[si] | relay) & (has_features[di] | relay)
    si, di = si[keep], di[keep]

    n = len(ids)
    rows, cols_idx = np.r_[si, di], np.r_[di, si]
    A = sparse.coo_matrix((np.ones(len(rows), "float32"), (rows, cols_idx)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    P = sparse.diags(inv.astype("float32")) @ A
    a = np.where(has_features, alpha, 0.0).astype("float32")[:, None]
    H = X.copy()
    for _ in range(iters):
        H = (1 - a) * (P @ H) + a * X

    prefix = POP_PREFIXES[(population, relay)]
    frame = pd.DataFrame(H[: len(nodes)], index=pd.Index(nodes["txId"].values, name="txId"),
                         columns=[f"{prefix}{c}" for c in cols])
    frame[deg[: len(nodes)] == 0] = np.nan
    return frame.astype("float32")


def build_population_features(train, test, edges, alpha=0.15, iters=10):
    blocks = [_population_ppr(train, test, edges, pop, relay, alpha, iters)
              for (pop, relay) in POP_PREFIXES]
    return pd.concat(blocks, axis=1)


def build_or_load_population(train, test, edges, rebuild=False, alpha=0.15, verbose=True):
    cache_path = Path(POPULATION_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "version": POPULATION_VERSION, "alpha": alpha}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding population features (different inputs or settings).")
    feats = build_population_features(train, test, edges, alpha=alpha)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def population_columns(df, kind):
    """kind: 'all' (unlabelled rows included), 'relay' (featureless pass-through),
    'all_relay' (both)."""
    prefix = {"all": "pa_", "relay": "pr_", "all_relay": "par_"}[kind]
    return [c for c in df.columns if c.startswith(prefix)]
'''

FEATURES_BLOCK = '''# Stage 17: propagation populations (who takes part in the graph).
NEEDS_POPULATION = {"pprall_cmp", "pprrelay_cmp", "pprallrelay_cmp", "ppr_pprallrelay_cmp"}
FEATURE_SETS["pprall_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                              + population_columns(df, "all") + group_columns(df, ["comp"]))
FEATURE_SETS["pprrelay_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + population_columns(df, "relay") + group_columns(df, ["comp"]))
FEATURE_SETS["pprallrelay_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                   + population_columns(df, "all_relay") + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_pprallrelay_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                       + ppr_columns(df) + population_columns(df, "all_relay")
                                       + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_pprallrelay_cmp"}
'''

FRAMES_BLOCK = '''    if sets & NEEDS_POPULATION:
        q = propagation.build_or_load_population(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, q), propagation.attach(test, q)
        if verbose:
            print(f"Attached {q.shape[1]} population-variant columns")
'''


def patch(path, marker, fn, label):
    p = ROOT / path
    if not p.exists():
        print(f"SKIP {label}: {path} not found")
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
ok &= patch("src/propagation.py", "build_population_features", lambda s: s + PROP_CODE,
            "propagation: population builders")


def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "population_columns" not in s:
        if "from src.propagation import (" in s:
            s = s.replace("from src.propagation import (", "from src.propagation import (population_columns, ", 1)
        else:
            s = s.replace("from src.propagation import ", "from src.propagation import population_columns, ", 1)
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    return s.replace("NEEDS_GRAPH |= {",
                     'NEEDS_GRAPH |= {"pprall_cmp", "pprrelay_cmp", "pprallrelay_cmp", "ppr_pprallrelay_cmp", ', 1)


ok &= patch("src/features.py", "NEEDS_POPULATION", _features, "features: population sets")


def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_POPULATION, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)


ok &= patch("src/frames.py", "NEEDS_POPULATION", _frames, "frames: population loader")
print("\nAll patches applied." if ok else "\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
