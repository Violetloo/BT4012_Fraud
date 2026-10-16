"""Load the competition data with any derived features the chosen feature sets need."""
from src.data import DATA_DIR, load_data
from src.features import (NEEDS_DIRECTED, NEEDS_GRAPH, NEEDS_MATCHED_RANKS, NEEDS_PROPAGATED,
                          NEEDS_CONTRAST, NEEDS_POPULATION, NEEDS_DISPERSION, NEEDS_KNN, NEEDS_FINE, NEEDS_ITERS, NEEDS_RANKS, NEEDS_STAGE15)
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
    if sets & NEEDS_DIRECTED:
        d = propagation.build_or_load_directed(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, d), propagation.attach(test, d)
        if verbose:
            print(f"Attached {d.shape[1]} directed/PageRank columns")
    if sets & NEEDS_STAGE15:
        m = propagation.build_or_load_stage15(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, m), propagation.attach(test, m)
        if verbose:
            print(f"Attached {m.shape[1]} max/symmetric columns")
    if sets & NEEDS_CONTRAST:
        c = propagation.build_or_load_contrast(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, c), propagation.attach(test, c)
        if verbose:
            print(f"Attached {c.shape[1]} contrast columns")
    if sets & NEEDS_POPULATION:
        q = propagation.build_or_load_population(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, q), propagation.attach(test, q)
        if verbose:
            print(f"Attached {q.shape[1]} population-variant columns")
    if sets & NEEDS_DISPERSION:
        d2 = propagation.build_or_load_dispersion(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, d2), propagation.attach(test, d2)
        if verbose:
            print(f"Attached {d2.shape[1]} dispersion columns")
    if sets & NEEDS_KNN:
        from src import knn_features
        kf = knn_features.build_or_load_knn(train, test, rebuild=rebuild, verbose=verbose)
        train, test = graph_features.attach(train, kf), graph_features.attach(test, kf)
        if verbose:
            print(f"Attached {kf.shape[1]} similarity-neighbour columns")
    if sets & NEEDS_FINE:
        fine = propagation.build_or_load_fine(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, fine), propagation.attach(test, fine)
        if verbose:
            print(f"Attached {fine.shape[1]} fine-decay PageRank columns")
    if sets & NEEDS_ITERS:
        it = propagation.build_or_load_iters(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, it), propagation.attach(test, it)
        if verbose:
            print(f"Attached {it.shape[1]} iteration-count columns")
    if sets & NEEDS_MATCHED_RANKS:
        m = transforms.build_or_load_matched(train, test, rebuild=rebuild, verbose=verbose)
        train, test = transforms.attach(train, m), transforms.attach(test, m)
        if verbose:
            print(f"Attached {m.shape[1]} matched-population rank columns")
    return train, test
