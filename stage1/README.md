# Stage 1: supplied-data inspection

Completed 21 September 2026; updated the same day with additional graph and feature checks. Only the four user-provided competition CSVs were used. Source files were not modified. No model has been trained or submitted.

## Dataset inventory

| File | Rows | Columns | Time steps |
|---|---:|---:|---|
| train.csv | 141,772 | 168 | 1–35 |
| test.csv | 15,329 | 168 | 36–49 |
| txs_edgelist.csv | 234,355 | 2 | Not supplied directly |
| sample_submission.csv | 15,329 | 2 | Not applicable |

The actual feature names are `feat_1` through `feat_165`, not `feature_1` through `feature_165`. Both transaction tables have `txId` and `time_step`. Train additionally has `label`; test instead has `index`. Do not use `txId` or submission `index` as baseline predictors. Use time_step for splitting; evaluate its inclusion as a predictor separately.

## Labels and quality checks

- Licit: 27,591. Illicit: 3,644. Known labels: 31,235.
- Unknown: 110,537 (about 78% of training rows). These are empty CSV fields, read as missing values, rather than literal `none` strings.
- Illicit share among known labels: 11.67%. This does not establish the illicit rate among unknown or test transactions.
- No missing feature values, infinite feature values, or constant feature columns in either transaction table.
- No duplicate transaction IDs, no train/test ID overlap, and no duplicate complete rows in any input file.
- There are 31 repeated feature vectors beyond their first occurrence in train, involving 44 rows in total. No group has conflicting known labels. Distinct transaction IDs can share numeric features; do not automatically remove these rows.

## Time coverage

Training contains every step from 1 through 35; test contains every step from 36 through 49. There is no temporal overlap. Every training step has both known classes, but step 6 has only five illicit examples.

Illicit share among labeled transactions ranges from 0.43% in step 5 to 35.97% in step 13. This describes observed labels, not confirmed changes in population prevalence. Forward-chaining validation over blocks is preferable to relying on individual small periods. The test horizon spans 14 steps, so short validation blocks alone may understate long-horizon uncertainty.

See time_step_overview.png, train_by_time_step.csv and test_by_time_step.csv.

## Graph coverage and limitations

All 141,772 train and 15,329 test transactions occur in the supplied graph. The graph has 203,769 distinct endpoint IDs, including 46,668 with no feature rows in train or test. Of 234,355 edges, 175,918 have both endpoints available; 58,437 have at least one unavailable endpoint. There are no duplicate edges or self-loops.

All 175,918 edges whose endpoints both have known time steps connect transactions within the same step. No cross-step edges occur among those observable endpoints. Missing-endpoint time steps cannot be checked from these files. The inspection does not establish that the entire graph is acyclic.

There are no direct train-to-test or test-to-train edges. No train-incident edges have missing neighbors, whereas 24,245 test-incident edges do. This is an important asymmetry: graph feature aggregates are more incomplete for test transactions. Validation should examine sensitivity to incomplete neighbor features rather than assuming equal coverage.

Degrees can use the supplied edge structure even when a neighbor lacks features. Feature aggregates must use neighbors whose feature rows are available and should retain available-neighbor counts or coverage indicators. Do not invent missing-node features, labels, or time steps. Unknown-labeled training nodes still have usable feature rows and are distinct from nodes absent from both transaction files.

## Missing graph nodes (hypothesis)

The 46,668 unavailable endpoint nodes are not all directly attached to test transactions: 17,897 are directly adjacent to a test transaction, and 34,192 edges connect two unavailable nodes. However, every unavailable node lies in a weakly connected component that contains test transactions, none lies in a component containing training transactions, and no such component spans more than one test time step.

Test also averages about 1,095 rows per step, close to the ~892 labeled rows per training step and far below the ~4,051 total rows per training step. Together these facts are consistent with the hypothesis that the unavailable nodes are unlabeled test-period transactions removed from the files while their edges were kept. The files cannot confirm this: the labels and time steps of unavailable nodes are not observable.

## Neighbour availability

Labeled training transactions have feature rows for 100% of their neighbours; on average 63.1% of their neighbours are labeled. Test transactions have feature rows for 60.9% of their neighbours on average. Aggregating training neighbours over labeled rows only is therefore a candidate way to mimic test coverage, but similar mean coverage does not show that the missingness patterns match. Treat it as an experiment to compare with all-available-neighbour aggregation on identical chronological splits. Such features may use a neighbour's availability (has a label / has a feature row), never its class value.

Test transactions also have a heavier-tailed in-degree than training transactions (99th percentile 42 vs 9, mean 2.02 vs 1.15). Degree features may therefore shift between periods.

## Feature diagnostics

See feature_diagnostics.csv. KS statistics compare labeled training rows with test rows.

- No duplicate feature columns across train and test.
- Six features (`feat_100`, `feat_101`, `feat_103`, `feat_136`, `feat_137`, `feat_139`) have KS 0.92–1.00 and per-step means that are almost perfectly monotonic in time_step (Spearman ≥ 0.995). For `feat_100`, `feat_101` and `feat_103`, over 99% of test values exceed the training maximum; the other three shift strongly but stay within the training range. These are candidates for exclusion. The Stage 2 starting configuration excludes them, and retaining them will be compared on identical chronological splits: correlation with time alone does not prove a feature is harmful.
- 21 features have KS above 0.3 (including the six above); revisit these in Stage 6.
- Labeled and unknown training rows differ moderately (median KS 0.14 across features), which is relevant before any pseudo-labelling.

## Submission mapping

Test `index` exactly matches the sample submission, in the same order, from 0 through 15,328. Preserve this explicit column and align predictions to it after any joins or sorting. Export exactly `index,target` with no extra dataframe index. `target` must contain the predicted probability of label 1.

## Reproduction and next stage

Run `python3 stage1/inspect_data.py --data-dir /path/to/competition/files`. The script needs Python, pandas, NumPy, SciPy and Matplotlib. `--data-dir` defaults to `../data` relative to the script. Outputs default to the script directory. diagnostics.json records input SHA-256 hashes and detailed checks; feature_diagnostics.csv records per-feature drift statistics; run_log.txt records the run summary.

Stage 2 should define fixed chronological splits, check their labeled class counts, and establish a reusable evaluation function and experiment log. Fit any learned preprocessing only on each split's training data. Reserve unknown labels from supervised fitting and evaluation.
