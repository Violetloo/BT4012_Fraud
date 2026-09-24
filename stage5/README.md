# Stage 5: graph features

Stage 5 adds features built from `txs_edgelist.csv`, one group at a time, and tests each across several long-horizon splits rather than one.

## Why this stage matters

The Stage 3/4 models score about 0.98 on 5-step folds but 0.90 on a 14-step horizon and 0.944 on the leaderboard. Raw features alone degrade over long ranges. Graph structure may describe behaviour that survives when the individual transaction features shift.

## Acceptance rule (fixed before running)

Stage 4 taught the key lesson: `hgb_balanced` gained +0.0164 on one horizon split and nothing on the leaderboard. So a cumulative feature set replaces the previous one only if **all three** hold, judged against the previous set:

1. fold-mean AUC is not worse by more than **0.002**,
2. the mean AUC over the **four** long-horizon splits improves, and
3. at least **3 of those 4** splits improve.

The long-horizon splits train on steps 1–k and validate the following steps, for k = 15, 18, 21, 24: three 14-step windows (h15, h18, h21) and one 11-step window (h24: steps 25–35, the end of the training data). `h21` is the same split as the usual horizon. They share training periods and their validation windows overlap heavily, so "3 of 4 improve" is a **consistency check, not four independent confirmations**. Weak-step AUC is reported as a diagnostic only, never as an acceptance condition.

## Feature groups

| Set | Adds | Columns |
|---|---|---|
| `raw_g1` | `deg_` degree: in, out, total, in-share | 4 |
| `raw_g2` | `avl_` usable-neighbour counts and shares, per direction | ~6 |
| `raw_g3` | `nbm_` mean of every supplied feature over predecessors and successors | 318 |
| `raw_g4` | `nbs_` max/min/std over the 20 highest-variance features, per direction | 120 |
| `raw_g5` | `cmp_` component size, usable share, and degree divided by component size | 3 |

Isolated sets test one group on its own: `raw_deg`, `raw_avl`, `raw_nbm`, `raw_nbs`, `raw_cmp`. The cumulative sets cannot isolate a group, because each still contains the groups added before it.

## Leakage review, per group

The guards in `evaluate()` check how rows are separated; they cannot certify a feature. Each group was reviewed by hand:

- **`deg_`**: counts from the complete edge list, including neighbours with no feature row. No labels involved. Equally complete for train and test.
- **`avl_`**: counts and shares of neighbours whose feature rows are *usable*. This uses label **availability**, never a label's value. The reason: test nodes are missing about 39% of their neighbours' rows, so aggregating over labelled train neighbours (63%) matches test coverage (61%) — 47.1% vs 49.4% missing aggregates among the rows actually used. That those missing nodes are unlabelled test-period transactions remains a Stage 1 **hypothesis**; their labels and time steps are unobservable. Matching mean coverage does not prove the missingness patterns match, and aggregating over *all* available training neighbours has not been compared against it.
- **`nbm_`, `nbs_`**: aggregates of supplied feature values over usable neighbours. No labels. The 20 features chosen for max/min/std are the highest-variance ones **within steps 1–15 only**. Every split trains on step 1 through k ≥ 15, so that window is historical for all of them. An earlier version chose them using all periods including test: no labels were involved, but later distributions would have informed features used in earlier validation blocks. "No labels used" is not the same as "no look-ahead".
- **`cmp_`**: component sizes from the edge list. No labels.

Every group is built once over train and test together. That is transductive: a test row's aggregates use its test-period neighbours' feature values. It uses no labels, the competition supplies the full edge list and test features, and nothing here uses a neighbour's class. The one place where look-ahead could have leaked into the *choice* of features (the max/min/std selection) is restricted to steps 1–15.

## Missing values

Aggregates are NaN when a node has no usable neighbour. They are **not filled**: the model is `hgb_balanced`, and histogram gradient boosting handles missing values natively, learning which side of a split they belong on. The script prints each group's missing rate.

## How to run

```bash
# full sweep: builds the cache, then 6 feature sets (expect roughly 30-90 minutes)
python3 stage5/run_graph_experiments.py

# a quicker first look
python3 stage5/run_graph_experiments.py --sets raw_notime raw_g1 raw_g3 --version s5quick
```

The features are cached in `data/derived/graph_features.pkl` (git-ignored). The cache stores content hashes of train, test and the edge list plus the builder version and settings, and is rebuilt automatically when any of them differ, so a changed `--data-dir` or edge list cannot silently reuse old features. The submission-time data hash cannot catch that on its own, because it sees cached values as current. `--rebuild-features` forces a rebuild, and `attach()` checks unique ids and full coverage. `make_submission.py` attaches them automatically for any `raw_g*` feature set, so a submission reproduces exactly what was validated.

## Outputs

- `stage5/graph_feature_results_<version>.csv`: every set's fold, horizon, weak-step and multi-horizon scores.
- `stage5/multi_horizon_<version>.csv`: the four long-horizon splits per set.
- `stage5/graph_feature_decisions_<version>.csv`: each group's deltas and the keep/reject decision.

## What to send back

1. The graph feature summary (column counts and missing rates).
2. The results table and the long-horizon table.
3. The per-group decisions and the recommendation.

## Notes

- If the sweep is too slow, run `--sets` in batches; the cache is shared.
- State results at the level the experiments support: **none of the tested graph-feature configurations improved the long-horizon criterion**. A cumulative set that fails does not show that its newest group fails on its own; use the isolated sets for that.
- Each set is evaluated twice: the standard-fold run (a row in `experiment_log.csv` plus a JSON record) and the long-horizon run (a JSON record and saved predictions, but no CSV row, since the log's columns are fixed to the four folds). Both are therefore auditable.
