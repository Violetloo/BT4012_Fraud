"""Data loading and shared constants for the BT4012 illicit-transaction project.

Only the four supplied competition files are read. Nothing here uses label
values to build features; labels are only returned for supervised fitting and
scoring.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Stage 1: six features whose per-step means rise almost perfectly with
# time_step and which barely overlap between train and test. Excluded in the
# starting configuration; Stage 3 compares keeping them on identical splits.
TIME_ENCODING_CANDIDATES = ["feat_100", "feat_101", "feat_103", "feat_136", "feat_137", "feat_139"]

# Never used as predictors (identifiers, split variable, target).
NON_FEATURE_COLUMNS = ["txId", "time_step", "label", "index"]

# Raw label text accepted as "unknown" (compared after stripping and lower-casing).
UNKNOWN_LABEL_STRINGS = {"", "none", "unknown", "nan"}
LABEL_VALUES = {"0": 0.0, "0.0": 0.0, "1": 1.0, "1.0": 1.0}


def parse_label(raw):
    """Map raw label text to 1.0 (illicit), 0.0 (licit) or NaN (unknown).

    Any other text raises ValueError instead of being silently treated as unknown.
    """
    text = str(raw).strip().lower()
    if text in UNKNOWN_LABEL_STRINGS:
        return np.nan
    if text in LABEL_VALUES:
        return LABEL_VALUES[text]
    raise ValueError(f"Unexpected label value: {raw!r}")


def load_data(data_dir=DATA_DIR):
    """Return train, test, edges as DataFrames.

    train["label"] is float: 1.0 illicit, 0.0 licit, NaN unknown.
    """
    data_dir = Path(data_dir)
    # The converter receives the raw text of each label cell, so invalid strings
    # are caught before any numeric conversion.
    train = pd.read_csv(data_dir / "train.csv", converters={"label": parse_label})
    test = pd.read_csv(data_dir / "test.csv")
    edges = pd.read_csv(data_dir / "txs_edgelist.csv")
    train["label"] = train["label"].astype(float)
    return train.copy(), test.copy(), edges


def raw_feature_columns(df, exclude_time_candidates=True):
    """Supplied feature columns (feat_*), optionally without the six time-encoding candidates."""
    cols = [c for c in df.columns if c.startswith("feat_")]
    if exclude_time_candidates:
        cols = [c for c in cols if c not in TIME_ENCODING_CANDIDATES]
    return cols


def labelled(train):
    """Rows with a known label (unknowns are never used for fitting or scoring)."""
    return train[train["label"].notna()].copy()


def set_seed(seed=42):
    np.random.seed(seed)
    return seed
