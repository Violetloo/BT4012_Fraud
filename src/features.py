from src.knn_features import knn_columns
"""Named feature sets. Later stages add graph features here.

Rule for every feature builder: never use label values. Use only supplied
feature columns, time_step for grouping, and the edge list.
"""
import json
from pathlib import Path

from src.data import REPO_ROOT, raw_feature_columns
from src.graph_features import group_columns
from src.propagation import (fine_columns, dispersion_columns, population_columns, contrast_columns, directed_columns, max_columns, ppr_columns, ppr_columns_alpha,
                             propagated_columns, symmetric_columns)
from src.transforms import matched_rank_columns, rank_columns

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
NEEDS_MATCHED_RANKS = {"rankm_cmp", "rankmraw_cmp"}
# Stage 13: multi-hop propagated features (src/propagation.py)
NEEDS_PROPAGATED = {"prop1_cmp", "prop2_cmp", "prop2_avl_cmp", "prop2_only"}
FEATURE_SETS["prop1_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + propagated_columns(df, hops=(1,)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop2_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + propagated_columns(df, hops=(1, 2)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop2_avl_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                 + propagated_columns(df, hops=(1, 2))
                                 + group_columns(df, ["avail", "comp"]))
# Stage 14: directed propagation (upstream vs downstream kept apart) and
# personalised-PageRank propagation (all hop distances mixed with a decay).
NEEDS_DIRECTED = {"ppr05_cmp", "ppr30_cmp", "ppr_multi_cmp", "dir34_cmp", "dir34_prop34_cmp", "ppr_cmp", "dir34_ppr_cmp", "dir34_avl_cmp"}
FEATURE_SETS["dir34_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + directed_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["dir34_avl_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                 + directed_columns(df) + group_columns(df, ["avail", "comp"]))
FEATURE_SETS["dir34_prop34_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                    + directed_columns(df) + propagated_columns(df, hops=(2, 3, 4))
                                    + group_columns(df, ["comp"]))
FEATURE_SETS["ppr05_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + ppr_columns_alpha(df, 0.05) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr30_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + ppr_columns_alpha(df, 0.3) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_multi_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                 + ppr_columns(df) + ppr_columns_alpha(df, 0.05)
                                 + ppr_columns_alpha(df, 0.3) + group_columns(df, ["comp"]))
NEEDS_STAGE15 = {"max_cmp", "ppr_max_cmp", "sym_cmp", "ppr_sym_cmp"}
NEEDS_DIRECTED |= {"ppr_max_cmp", "ppr_sym_cmp"}
FEATURE_SETS["max_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + max_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_max_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + ppr_columns(df) + max_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["sym_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + symmetric_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_sym_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + ppr_columns(df) + symmetric_columns(df) + group_columns(df, ["comp"]))
NEEDS_CONTRAST = {"contrast_cmp", "ppr_contrast_cmp", "ppr_cppr_cmp"}
FEATURE_SETS["contrast_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + contrast_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_contrast_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                    + ppr_columns(df) + contrast_columns(df)
                                    + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_cppr_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + contrast_columns(df, kinds=("ppr",))
                                + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_contrast_cmp", "ppr_cppr_cmp"}
# Stage 17: propagation populations (who takes part in the graph).
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
# Stage 18: neighbourhood dispersion (how varied the neighbourhood is, and how far the
# transaction sits from it in those units).
NEEDS_DISPERSION = {"disp_cmp", "ppr_disp_cmp", "ppr_z_cmp"}
FEATURE_SETS["disp_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                            + dispersion_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_disp_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + dispersion_columns(df)
                                + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_z_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + ppr_columns(df) + dispersion_columns(df, kinds=("z",))
                             + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_disp_cmp", "ppr_z_cmp"}
# Stage 19: similarity-neighbour label features (src/knn_features.py). These are the
# only features that carry LABEL information to the test rows: the payment graph cannot,
# because every edge lies inside one time step.
NEEDS_KNN = {"knn_cmp", "ppr_knn_cmp", "prop34_knn_cmp", "ppr_multi_knn_cmp"}
FEATURE_SETS["knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + knn_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + ppr_columns(df) + knn_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["prop34_knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                  + propagated_columns(df, hops=(2, 3, 4)) + knn_columns(df)
                                  + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_knn_cmp"}
NEEDS_PROPAGATED |= {"prop34_knn_cmp"}
# Stage 20. Both sets below extend something the LEADERBOARD rewarded, not something
# validation liked: the best single model mixes PageRank decays, and the best submission
# blends the PageRank and discrete-hop representations.
NEEDS_FINE = {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp"}
FEATURE_SETS["ppr_fine_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + fine_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_fine_multi_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                      + ppr_columns(df) + ppr_columns_alpha(df, 0.05)
                                      + ppr_columns_alpha(df, 0.3) + fine_columns(df)
                                      + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_multi_prop34_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                        + ppr_columns(df) + ppr_columns_alpha(df, 0.05)
                                        + ppr_columns_alpha(df, 0.3)
                                        + propagated_columns(df, hops=(2, 3, 4))
                                        + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp"}
NEEDS_PROPAGATED |= {"ppr_multi_prop34_cmp"}
FEATURE_SETS["ppr_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + ppr_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["dir34_ppr_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                 + directed_columns(df) + ppr_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["prop3_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + propagated_columns(df, hops=(1, 2, 3)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop23_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                              + propagated_columns(df, hops=(2, 3)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop4_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                             + propagated_columns(df, hops=(1, 2, 3, 4)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop34_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                              + propagated_columns(df, hops=(2, 3, 4)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop4only_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                 + propagated_columns(df, hops=(3, 4)) + group_columns(df, ["comp"]))
FEATURE_SETS["prop2_only"] = (lambda df: propagated_columns(df, hops=(1, 2))
                              + group_columns(df, ["comp"]))
FEATURE_SETS["rankm_cmp"] = lambda df: matched_rank_columns(df) + group_columns(df, ["comp"])
FEATURE_SETS["rankmraw_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + matched_rank_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["rank_cmp"] = lambda df: rank_columns(df) + group_columns(df, ["comp"])
FEATURE_SETS["rankraw_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + rank_columns(df) + group_columns(df, ["comp"]))
for _n in (5, 10, 20):
    FEATURE_SETS[f"raw_cmp_drop{_n}"] = (lambda n: lambda df: (
        [c for c in raw_feature_columns(df, exclude_time_candidates=True) if c not in set(_drop_list(n))]
        + group_columns(df, ["comp"])))(_n)
NEEDS_GRAPH |= {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp", "knn_cmp", "ppr_knn_cmp", "prop34_knn_cmp", "disp_cmp", "ppr_disp_cmp", "ppr_z_cmp", "pprall_cmp", "pprrelay_cmp", "pprallrelay_cmp", "ppr_pprallrelay_cmp", "contrast_cmp", "ppr_contrast_cmp", "ppr_cppr_cmp", "max_cmp", "ppr_max_cmp", "sym_cmp", "ppr_sym_cmp", "ppr05_cmp", "ppr30_cmp", "ppr_multi_cmp", "dir34_cmp", "dir34_avl_cmp", "dir34_prop34_cmp", "ppr_cmp", "dir34_ppr_cmp",
                "prop1_cmp", "prop2_cmp", "prop2_avl_cmp", "prop2_only", "prop3_cmp", "prop23_cmp",
                "prop4_cmp", "prop34_cmp", "prop4only_cmp",
                "rank_cmp", "rankraw_cmp", "rankm_cmp", "rankmraw_cmp",
                "raw_cmp_drop5", "raw_cmp_drop10", "raw_cmp_drop20"}

for _name, _groups in GRAPH_STAGES.items():
    FEATURE_SETS[_name] = (lambda groups: lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + group_columns(df, groups))(_groups)


def get_feature_columns(name, df):
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
