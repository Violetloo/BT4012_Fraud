# Stage 3: baselines, time-feature comparison and first submission

Stage 3 trains the first real models through `evaluate()`, decides whether to keep the six time-encoding features excluded, and produces the first Kaggle submission.

## Files

| File | Purpose |
|---|---|
| `src/features.py` | Named feature sets: `raw_notime` (159 features) and `raw_all` (165). Later stages add graph features here |
| `src/models.py` | Named model factories: `logreg` and `hgb`, every `random_state` fixed |
| `src/reporting.py` | Compares two `evaluate()` results; reads and updates the experiment log |
| `stage3/run_baselines.py` | Runs 2 models × 2 feature sets, prints the time-feature comparison, a reference ranking and the recommended submission. Outputs are suffixed by version (`_v1`, `_v2`, …) |
| `stage3/investigate_lr_warnings.py` | Tests whether the runtime warnings affected logistic regression (arithmetic and optimisation checks) |
| `stage3/make_submission.py` | Retrains the configuration of one named validation run on all labelled rows and writes a checked submission file |
| `stage3/record_lb.py` | Writes the Kaggle public score back into the experiment log |

## Models

- **`logreg`**: standardised logistic regression. A simple linear reference point.
- **`hgb`**: scikit-learn's `HistGradientBoostingClassifier` (learning rate 0.05, 300 iterations, 31 leaves). It is the same family as LightGBM but needs no extra installation. LightGBM and XGBoost are compared in Stage 4.

Both use fixed settings with no early stopping and no class weighting. The model retrained for submission therefore uses exactly the validated configuration, and class weighting is tested in Stage 4 rather than assumed.

## Decision order

Decisions are made in this order, and the earlier one takes precedence:

1. **Feature set, by the conservative time-feature rule.** The Stage 1 starting hypothesis is to exclude `feat_100`, `feat_101`, `feat_103`, `feat_136`, `feat_137` and `feat_139`. They are included only if, for `hgb`, all three hold: the mean fold AUC improves, at least 3 of the 4 folds improve, and the horizon AUC does not get worse. The rule is conservative on purpose: these features are almost perfectly ordered by time, and test steps (36–49) lie beyond all training steps, so a small gain on 5-step blocks may not carry over. The logreg comparison is reported for reference.
2. **Model, within the chosen feature set.** The best mean fold AUC within that feature set is recommended, unless a close runner-up (within about 0.005) has a clearly better minimum fold, horizon AUC and weak-step AUC.

The ranking across all four configurations is for reference only. In v1 it placed `hgb_raw_all` first by 0.0007, a difference far smaller than the variation between folds, but the feature rule (2 of 4 folds improved) keeps the features excluded. Describe `raw_notime` as the conservative rule-based choice, not as the highest validation AUC.

## Horizon diagnostics

Every comparison reports the horizon AUC and the **weak-step AUC**: the horizon split restricted to steps 24, 29, 31 and 32, where the v1 `hgb` baseline scored 0.74–0.82 while most other steps scored above 0.95. These steps were identified from a validation result, so use them to diagnose, not as a target to tune against. The cause of the weakness is not established; one consistent explanation is illicit behaviour that appears after the training window.

## Logistic regression: provisional

The logreg runs recorded about 3,700 runtime warnings (overflow, invalid value, divide by zero in matrix multiplication). Finite predictions alone do not show that these are harmless, so logreg results stay provisional until `investigate_lr_warnings.py` passes. It checks that:

- **A. Arithmetic:** decision values recomputed without the BLAS routine match to floating-point precision.
- **B. Optimisation:** lbfgs and a second solver (newton-cholesky), both run to a tight tolerance, reach the same coefficients and AUC.

It also records where the warnings arise. The `hgb` runs recorded no warnings.

## Submissions reproduce one named run

`make_submission.py` requires `--run-id` and refuses to build unless the model's full parameters (from `get_params`, not the printed representation), the ordered feature list, and a hash of the training data values all match that run's record. The data hash covers identifiers, labels and every feature value, so it changes if the input files, the feature-building logic or the column order change. The submission record also stores SHA-256 hashes of the input files and the categories and messages of any warnings.

Runs logged before this version (the v1 baselines) have no data hash or ordered feature list. They are refused unless you pass `--allow-legacy-run`, which still checks parameters and features. Prefer rerunning with a new version suffix.

## How to run

From the repo root, with the competition CSVs in `data/`:

```bash
# 1. Baselines, time-feature comparison and recommendation (a few minutes)
python3 stage3/run_baselines.py --version v2

# 2. Investigate the logistic-regression warnings
python3 stage3/investigate_lr_warnings.py

# 3. Build the submission; run_baselines.py prints the exact command with the run id, e.g.
python3 stage3/make_submission.py --model hgb --features raw_notime --run-id <run_id>

# 4. Upload the CSV on the competition's Submit page, then record the public score
python3 stage3/record_lb.py --submission submissions/<file>.csv --score 0.XXXXX
```

Reused experiment names are rejected, so each rerun needs a new `--version`.

## What each script checks

- **`make_submission.py`** refuses to build unless parameters, ordered features and data hash match the named run (see above). It trains on all labelled rows, steps 1–35, excluding unknowns. The file must have 15,329 rows and exactly the columns `index,target`, with `index` identical to `sample_submission.csv` in value and order, and every prediction finite and within [0, 1]. It re-reads the written file to confirm, and saves a JSON record next to it with the model parameters, features, code and git state, and a prediction summary.
- **`record_lb.py`** attaches the public score to the validated run through that JSON record, so each leaderboard score is traceable to its validation result.

## What to send back

1. The full output of `run_baselines.py`: comparison tables, reference ranking and recommendation.
2. The output of `investigate_lr_warnings.py`, including its A and B verdicts.
3. The output of `make_submission.py`.
4. The public leaderboard score, after recording it.

## Notes

- Commit code before building a submission. The JSON record stores the commit and a `git_dirty` flag, and the script warns if code has uncommitted changes.
- `submissions/` is git-ignored. Keep the JSON records locally; they document which file produced which score.
- The public leaderboard uses 30% of test. Treat it as a sanity check, not as the basis for choosing models.
