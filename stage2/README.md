# Stage 2: chronological validation, evaluate() and experiment log

Stage 2 builds the evaluation framework that every later experiment uses. It does not develop a model.

## Files

| File | Purpose |
|---|---|
| `src/README.md` | One-line description of each shared module |
| `src/data.py` | Loads the supplied CSVs, parses labels strictly (1.0 / 0.0 / NaN; unexpected text raises an error), lists feature columns, and holds the six time-encoding candidates from Stage 1 |
| `src/validation.py` | Split definitions, `check_splits()`, `evaluate()`, run records and the experiment log |
| `stage2/check_splits.py` | Verifies the splits and saves labelled class counts to `stage2/split_summary.csv` |
| `stage2/sanity_checks.py` | Required pipeline checks (exits with status 1 on failure) plus informational diagnostics; saves `stage2/permutation_check.csv` |
| `experiments/experiment_log.csv` | One summary row per logged experiment |
| `experiments/runs/` | One full JSON record per logged experiment |
| `experiments/oof/` | Validation predictions per run (git-ignored) |
| `experiments/feature_sets/` | Feature lists, keyed by the `feature_hash` in the log |

## Split design

Training covers steps 1–35 and test covers steps 36–49, so validation always predicts later steps from earlier ones.

| Split | Train steps | Validation steps | Role |
|---|---|---|---|
| fold1 | 1–15 | 16–20 | Averaged |
| fold2 | 1–20 | 21–25 | Averaged |
| fold3 | 1–25 | 26–30 | Averaged |
| fold4 | 1–30 | 31–35 | Averaged |
| horizon | 1–21 | 22–35 | Reported separately |

- **Expanding windows.** Each fold trains on all earlier steps, as the final model will. Whether older steps help (steps 1–6 are almost all licit) is tested later, not assumed.
- **Five-step validation blocks.** Single steps are too small; several have fewer than 60 illicit rows. Each 5-step block has 566–690 illicit labelled rows (see `split_summary.csv`).
- **Horizon split.** Test spans 14 steps, but folds validate only 5 steps ahead. The horizon split trains on 1–21 and validates on 22–35. It overlaps the folds, so it is reported beside the fold mean, not averaged into it.
- **No gap.** Every edge between known transactions lies within one time step, so time-step splits never separate the two ends of an observed edge.
- **Unknown labels** are excluded from fitting and scoring. Their feature rows may still be used to build graph features later, as long as no label values are used.

## Metrics

- **Primary:** mean ROC AUC over the four folds (the competition metric).
- **Stability:** population standard deviation and minimum fold AUC. The standard deviation describes variation between folds; it is not a confidence interval, because training windows overlap.
- **Secondary:** mean average precision (PR AUC).
- **Horizon AUC:** longer-range generalisation check.
- **Per-step AUC:** in the returned result and the run record, for diagnosis. Steps with one class get NaN.

Public leaderboard scores go in the `public_lb` column by hand. Choose models by validation, not the public leaderboard (30% of test).

## Using evaluate()

```python
from src.data import load_data, raw_feature_columns
from src.validation import evaluate

train, test, edges = load_data()
cols = raw_feature_columns(train)            # 159 features (time-encoding candidates excluded)

result = evaluate(
    name="lgbm_raw_v1",                                  # unique; reused names are rejected
    make_model=lambda: LGBMClassifier(random_state=42),  # must return a NEW model, random_state set
    frame=train,                                         # unknown rows are dropped inside
    feature_cols=cols,
    description="LightGBM defaults on raw features",
)
result["summary"], result["folds"], result["per_step"], result["predictions"], result["warnings"]
```

Rules and built-in guards:
- `make_model` must build a fresh model on each call. Put scalers and other learned preprocessing inside a sklearn `Pipeline` so they are fit on each split's training rows only.
- Every `random_state` parameter, including nested ones inside pipelines, must be set explicitly; otherwise `evaluate()` raises an error.
- `evaluate()` rejects `txId`, `time_step`, `label` and `index` as predictors; requires non-empty training and validation sets containing both classes; requires validation rows strictly later than training rows with no shared transactions; takes the class-1 probability via `model.classes_`; and rejects non-finite probabilities or values outside [0, 1].
- Warnings raised while fitting are captured and counted (convergence, runtime, other), printed as a one-line summary, and stored in the log and run record. They are never silenced.
- Feature builders must never use label values. The guards check how rows are separated and that outputs are well formed; they cannot certify that a feature is free of leakage, so review every new feature for this.
- Early stopping or tuning must not use the outer validation block. Use an inner chronological holdout inside each training window.

## Experiment records

Each logged run writes, only after every split has finished successfully:
- **A row in `experiment_log.csv`:** metrics per fold and summary, warning counts, the full model description, all `random_state` values, the split definitions, feature count and hash, git commit, `git_dirty` (whether project code had uncommitted changes), a `code_hash` of `src/*.py` and `stage*/*.py`, and an empty `public_lb` column.
- **A JSON record in `experiments/runs/`:** everything above plus all estimator parameters, per-split and per-step metrics, warning messages and library versions.

Together, the commit, dirty flag and code hash identify the exact code, even for runs made before committing. If the log's columns change in a future version, `evaluate()` refuses to append to the old file; archive it first.

## Using saved predictions

Horizon predictions (steps 22–35) overlap fold predictions for the same transactions. Always filter by `split` and join on (`split`, `txId`), never on `txId` alone. For ensembling, use the four fold splits; keep the horizon split as a separate check.

## How to run

From the repo root, with the competition CSVs in `data/`:

```bash
python3 stage2/check_splits.py
python3 stage2/sanity_checks.py
```

## What to check

1. `check_splits.py` prints "All checks passed". Every training and validation part has both classes.
2. `sanity_checks.py` ends with "All required checks passed" (exit status 0):
   - **Random scores:** AUCs within 0.45–0.55.
   - **Positive control:** a canary feature equal to the label scores above 0.99, so the pipeline can learn target information it is given.
   - **Guards:** forbidden columns, overlapping splits, unset `random_state`, one-class validation blocks and invalid label text are all rejected.
   - **Reproducibility:** identical AUCs on a repeated run.
   - **Finite outputs and convergence:** logistic regression on real labels gives finite probabilities on every split with no convergence warnings. Runtime-warning counts are reported, not hidden.
3. The informational sections:
   - **Matmul diagnostic:** whether a plain NumPy matrix multiplication on finite data raises RuntimeWarnings here, and which BLAS library is in use. If it does, runtime warnings seen during fits are likely environment-related, but they stay visible and counted.
   - **Label-permutation observation:** permuted-label models do not centre exactly on AUC 0.5 here, and their AUC varies with how they score extreme rows. This is an observation about the data, not evidence for or against leakage.
4. `experiments/experiment_log.csv` has one `sanity_random` row, and `experiments/runs/` has its JSON record.
