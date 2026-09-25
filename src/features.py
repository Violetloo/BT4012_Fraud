"""Named feature sets. Later stages add graph features here.

Rule for every feature builder: never use label values. Use only supplied
feature columns, time_step for grouping, and the edge list.
"""
import json
from pathlib import Path

from src.data import REPO_ROOT, raw_feature_columns
from src.graph_features import group_columns
from src.transforms import rank_columns

# Features to drop, written by stage6/adversarial_validation.py. Absent until then.
DROP_LIST_PATH = REPO_ROOT / "stage6" / "drift_drop_lists.json"


def _drop_list(n):
    if not Path(DROP_LIST_PATH).exists():
        raise FileNotFoundError(f"{DROP_LIST_PATH} not found; run stage6/adversarial_validation.py first")
    return json.loads(Path(DROP_LIST_PATH).read_text())[str(n)]

# Cumulative graph feature sets, one group added at a time (Stage 5). Because they
# are cumulative, a poor result for a later set does not isolate that group: it still
# contains the earlier ones. The isolated sets below test single groups on their own.
GRAPH_STAGES = {
    "raw_g1": ["deg"],
    "raw_g2": ["deg", "avail"],
    "raw_g3": ["deg", "avail", "nbrmean"],
    "raw_g4": ["deg", "avail", "nbrmean", "nbrstats"],
    "raw_g5": ["deg", "avail", "nbrmean", "nbrstats", "comp"],
    # isolated: raw features plus exactly one group
    "raw_deg": ["deg"],
    "raw_avl": ["avail"],
    "raw_nbm": ["nbrmean"],
    "raw_nbs": ["nbrstats"],
    "raw_cmp": ["comp"],
    "raw_avl_cmp": ["avail", "comp"],
}
NEEDS_GRAPH = set(GRAPH_STAGES)

FEATURE_SETS = {
    # 159 supplied features: the six time-encoding candidates from Stage 1 removed
    "raw_notime": lambda df: raw_feature_columns(df, exclude_time_candidates=True),
    # all 165 supplied features
    "raw_all": lambda df: raw_feature_columns(df, exclude_time_candidates=False),
}

# Stage 6 sets. "rank" columns are per-time-step percentile ranks (src/transforms.py);
# "drop" sets remove the most drift-prone raw features found by adversarial validation.
NEEDS_RANKS = {"rank_cmp", "rankraw_cmp"}
FEATURE_SETS["rank_cmp"] = lambda df: rank_columns(df) + group_columns(df, ["comp"])
FEATURE_SETS["rankraw_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + rank_columns(df) + group_columns(df, ["comp"]))
for _n in (5, 10, 20):
    FEATURE_SETS[f"raw_cmp_drop{_n}"] = (lambda n: lambda df: (
        [c for c in raw_feature_columns(df, exclude_time_candidates=True) if c not in set(_drop_list(n))]
        + group_columns(df, ["comp"])))(_n)
NEEDS_GRAPH |= {"rank_cmp", "rankraw_cmp", "raw_cmp_drop5", "raw_cmp_drop10", "raw_cmp_drop20"}

for _name, _groups in GRAPH_STAGES.items():
    FEATURE_SETS[_name] = (lambda groups: lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + group_columns(df, groups))(_groups)


def get_feature_columns(name, df):
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
