"""Named feature sets. Later stages add graph features here.

Rule for every feature builder: never use label values. Use only supplied
feature columns, time_step for grouping, and the edge list.
"""
from src.data import raw_feature_columns

FEATURE_SETS = {
    # 159 supplied features: the six time-encoding candidates from Stage 1 removed
    "raw_notime": lambda df: raw_feature_columns(df, exclude_time_candidates=True),
    # all 165 supplied features
    "raw_all": lambda df: raw_feature_columns(df, exclude_time_candidates=False),
}


def get_feature_columns(name, df):
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
