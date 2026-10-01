"""Load the competition data with any derived features the chosen feature sets need."""
from src.data import DATA_DIR, load_data
from src.features import NEEDS_GRAPH, NEEDS_MATCHED_RANKS, NEEDS_PROPAGATED, NEEDS_RANKS
from src import graph_features, propagation, transforms


def load_prepared(data_dir=DATA_DIR, feature_sets=(), rebuild=False, verbose=True):
    """Return (train, test) with graph and/or rank features attached as required."""
    train, test, edges = load_data(data_dir)
    sets = set(feature_sets)
    if sets & NEEDS_GRAPH:
        g = graph_features.build_or_load(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = graph_features.attach(train, g), graph_features.attach(test, g)
        if verbose:
            print(f"Attached {g.shape[1]} graph feature columns")
    if sets & NEEDS_RANKS:
        r = transforms.build_or_load(train, test, rebuild=rebuild, verbose=verbose)
        train, test = transforms.attach(train, r), transforms.attach(test, r)
        if verbose:
            print(f"Attached {r.shape[1]} per-step rank columns")
    if sets & NEEDS_PROPAGATED:
        pr = propagation.build_or_load(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, pr), propagation.attach(test, pr)
        if verbose:
            print(f"Attached {pr.shape[1]} propagated (multi-hop) columns")
    if sets & NEEDS_MATCHED_RANKS:
        m = transforms.build_or_load_matched(train, test, rebuild=rebuild, verbose=verbose)
        train, test = transforms.attach(train, m), transforms.attach(test, m)
        if verbose:
            print(f"Attached {m.shape[1]} matched-population rank columns")
    return train, test
