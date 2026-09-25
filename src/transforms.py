"""Per-time-step rank normalisation.

Each supplied feature is replaced by its percentile rank within its own time step,
computed over every row of that step (train, unknown and test alike). Ranks use
feature values only, never labels, and never information from another period: a
row's rank depends on its own step's rows. Level shifts between periods therefore
cannot reach the model, which is the point — Stage 1 found features whose
distributions move over time, and Stage 5 found degree features that helped short
horizons while hurting long ones.
"""
import hashlib
from pathlib import Path

import pandas as pd

from src.data import REPO_ROOT, raw_feature_columns

CACHE_PATH = REPO_ROOT / "data" / "derived" / "rank_features.pkl"
BUILDER_VERSION = 1
PREFIX = "rnk_"


def _fingerprint(*frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(str(f.shape).encode())
        h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
    return h.hexdigest()[:16]


def build_rank_features(train, test):
    """Percentile rank of every supplied feature within its time step, indexed by txId."""
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    both = pd.concat([train[["txId", "time_step"] + cols], test[["txId", "time_step"] + cols]],
                     ignore_index=True).set_index("txId")
    ranked = both.groupby("time_step")[cols].rank(pct=True, method="average")
    ranked.columns = [f"{PREFIX}{c}" for c in cols]
    return ranked.astype("float32")


def build_or_load(train, test, cache_path=CACHE_PATH, rebuild=False, verbose=True):
    """Build once and cache, rebuilding whenever the inputs or builder change."""
    cache_path = Path(cache_path)
    meta = {"inputs": _fingerprint(train, test), "builder_version": BUILDER_VERSION}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding rank features (different inputs or builder version).")
    feats = build_rank_features(train, test)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def attach(frame, rank_features):
    if not frame["txId"].is_unique:
        raise ValueError("frame has duplicate txId values")
    missing = (~frame["txId"].isin(rank_features.index)).sum()
    if missing:
        raise ValueError(f"{missing} rows have no rank features; rebuild the cache")
    return frame.join(rank_features, on="txId")


def rank_columns(df):
    return [c for c in df.columns if c.startswith(PREFIX)]


# --- matched-population ranks ------------------------------------------------
# The ranks above use every row of a time step as the reference population. Training
# steps hold ~4,000 rows, mostly unlabelled; test steps hold ~1,095, all of the
# labelled kind. A training row is therefore ranked against a different mix than a
# test row, which introduces a shift instead of removing one. These ranks use the
# same reference population in both: labelled training rows and test rows ("usable"),
# the choice Stage 5 made for neighbour aggregates.
MATCHED_CACHE_PATH = REPO_ROOT / "data" / "derived" / "rank_matched_features.pkl"
MATCHED_PREFIX = "rnm_"
MATCHED_BUILDER_VERSION = 1


def build_matched_rank_features(train, test):
    """Percentile of each value within its step, against the usable rows of that step."""
    import numpy as np

    cols = raw_feature_columns(train, exclude_time_candidates=True)
    both = pd.concat([train[["txId", "time_step", "label"] + cols],
                      test[["txId", "time_step"] + cols].assign(label=1.0)], ignore_index=True)
    usable = both["label"].notna().values  # labelled train rows and every test row
    values = both[cols].to_numpy(dtype="float64")
    out = np.empty(values.shape, dtype="float32")
    for step, idx in both.groupby("time_step").indices.items():
        ref_idx = idx[usable[idx]]
        for j in range(values.shape[1]):
            ref = np.sort(values[ref_idx, j])
            v = values[idx, j]
            pos = (np.searchsorted(ref, v, side="left") + np.searchsorted(ref, v, side="right")) / 2
            out[idx, j] = pos / max(len(ref), 1)
    return pd.DataFrame(out, index=pd.Index(both["txId"].values, name="txId"),
                        columns=[f"{MATCHED_PREFIX}{c}" for c in cols])


def build_or_load_matched(train, test, cache_path=MATCHED_CACHE_PATH, rebuild=False, verbose=True):
    cache_path = Path(cache_path)
    meta = {"inputs": _fingerprint(train, test), "builder_version": MATCHED_BUILDER_VERSION}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding matched-rank features (different inputs or builder version).")
    feats = build_matched_rank_features(train, test)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def matched_rank_columns(df):
    return [c for c in df.columns if c.startswith(MATCHED_PREFIX)]
