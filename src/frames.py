"""Load the competition data with any derived features the chosen feature sets need."""
from src.data import DATA_DIR, load_data
from src.features import NEEDS_GRAPH, NEEDS_RANKS
from src import graph_features, transforms


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
    return train, test
