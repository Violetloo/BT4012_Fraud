# Stage 5: graph features

Stage 5 adds features built from `txs_edgelist.csv`, one group at a time, and tests each across several long-horizon splits rather than one.

## Why this stage matters

The Stage 3/4 models score about 0.98 on 5-step folds but 0.90 on a 14-step horizon and 0.944 on the leaderboard. Raw features alone degrade over long ranges. Graph structure may describe behaviour that survives when the individual transaction features shift.

## Acceptance rule (fixed before running)

Stage 4 taught the key lesson: `hgb_balanced` gained +0.0164 on one horizon split and nothing on the leaderboard. So a cumulative feature set replaces the previous one only if **all three** hold, judged against the previous set:

1. fold-mean AUC is not worse by more than **0.002**,
2. the mean AUC over the **four** long-horizon splits improves, and
3. at least **3 of those 4** splits improve.

The long-horizon splits train on steps 1–k and validate the next 14 steps (or up to step 35) for k = 15, 18, 21, 24. `h21` is the same split as the usual horizon. Weak-step AUC is reported as a diagnostic only, never as an acceptance condition.

## Feature groups

| Set | Adds | Columns |
|---|---|---|
| `raw_g1` | `deg_` degree: in, out, total, unique neighbours, in-share | 6 |
| `raw_g2` | `avl_` usable-neighbour counts and shares, per direction | ~6 |
| `raw_g3` | `nbm_` mean of every supplied feature over predecessors and successors | 318 |
| `raw_g4` | `nbs_` max/min/std over the 20 highest-variance features, per direction | 120 |
| `raw_g5` | `cmp_` component size, usable share, degree share within the time step | 3 |

## Leakage review, per group

The guards in `evaluate()` check how rows are separated; they cannot certify a feature. Each group was reviewed by hand:

- **`deg_`**: counts from the complete edge list, including neighbours with no feature row. No labels involved. Equally complete for train and test.
- **`avl_`**: counts and shares of neighbours whose feature rows are *usable*. This uses label **availability**, never a label's value. Stage 1 established the reason: test nodes are missing about 39% of their neighbours' rows, so aggregating over labelled train neighbours (63%) matches test coverage (61%).
- **`nbm_`, `nbs_`**: aggregates of supplied feature values over usable neighbours. No labels. The 20 features chosen for max/min/std are the highest-variance ones across all supplied rows, a label-free criterion.
- **`cmp_`**: component sizes from the edge list. No labels.

Every group is built once over train and test together. That is transductive (test feature values inform test-row features) but uses no labels, and the competition provides the full edge list and test features, so it is within the rules. Nothing here uses a neighbour's class.

## Missing values

Aggregates are NaN when a node has no usable neighbour. They are **not filled**: the model is `hgb_balanced`, and histogram gradient boosting handles missing values natively, learning which side of a split they belong on. The script prints each group's missing rate.

## How to run

```bash
# full sweep: builds the cache, then 6 feature sets (expect roughly 30-90 minutes)
python3 stage5/run_graph_experiments.py

# a quicker first look
python3 stage5/run_graph_experiments.py --sets raw_notime raw_g1 raw_g3 --version s5quick
```

The features are cached in `data/derived/graph_features.pkl` (git-ignored) and rebuilt with `--rebuild-features`. `make_submission.py` attaches them automatically for any `raw_g*` feature set, so a submission reproduces exactly what was validated.

## Outputs

- `stage5/graph_feature_results_<version>.csv`: every set's fold, horizon, weak-step and multi-horizon scores.
- `stage5/multi_horizon_<version>.csv`: the four long-horizon splits per set.
- `stage5/graph_feature_decisions_<version>.csv`: each group's deltas and the keep/reject decision.

## What to send back

1. The graph feature summary (column counts and missing rates).
2. The results table and the long-horizon table.
3. The per-group decisions and the recommendation.

## Notes

- Each set is logged twice: the standard run (logged, named `<model>_<set>_<version>`) and the multi-horizon run (not logged, since the log's columns are fixed to the four folds).
- If the sweep is too slow, run `--sets` in batches; the cache is shared.
- A group that fails the rule is still worth reporting. "Graph aggregates did not improve long-horizon generalisation" is a real finding, and the review above explains what was tried.
