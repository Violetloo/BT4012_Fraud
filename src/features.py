"""Named feature sets. Later stages add graph features here.

Rule for every feature builder: never use label values. Use only supplied
feature columns, time_step for grouping, and the edge list.
"""
from src.data import raw_feature_columns
from src.graph_features import group_columns

# Cumulative graph feature sets, one group added at a time (Stage 5).
GRAPH_STAGES = {
    "raw_g1": ["deg"],
    "raw_g2": ["deg", "avail"],
    "raw_g3": ["deg", "avail", "nbrmean"],
    "raw_g4": ["deg", "avail", "nbrmean", "nbrstats"],
    "raw_g5": ["deg", "avail", "nbrmean", "nbrstats", "comp"],
}
NEEDS_GRAPH = set(GRAPH_STAGES)

FEATURE_SETS = {
    # 159 supplied features: the six time-encoding candidates from Stage 1 removed
    "raw_notime": lambda df: raw_feature_columns(df, exclude_time_candidates=True),
    # all 165 supplied features
    "raw_all": lambda df: raw_feature_columns(df, exclude_time_candidates=False),
}

for _name, _groups in GRAPH_STAGES.items():
    FEATURE_SETS[_name] = (lambda groups: lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + group_columns(df, groups))(_groups)


def get_feature_columns(name, df):
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
