# Stage 4: model families and class weighting

Stage 4 asks two questions on identical splits: which model family generalises best, and does class weighting help? The feature set stays at `raw_notime` (159 features) from Stage 3.

## The selection rule (fixed before running)

Stage 3 showed why the fold mean alone is not enough: fold mean 0.983, public leaderboard 0.944, horizon 0.899. The test period spans 14 steps, so longer-range behaviour matters.

A candidate replaces the baseline (`hgb`) only if **both** hold:

1. fold-mean AUC is not worse by more than **0.002**, and
2. horizon AUC is **strictly better**.

Accepted candidates are ordered by an equal blend of fold mean and horizon AUC. That blend is an internal selection score, not an estimate of leaderboard AUC. If nothing qualifies, the Stage 3 baseline stays.

**Weak-step AUC (horizon steps 24, 29, 31, 32) is reported but is not an acceptance condition.** Those steps were chosen after inspecting a validation result, so they are not independent evidence; fixing a rule in advance does not make previously inspected periods independent. The same caution applies in weaker form to the horizon split itself, which has now been examined several times: treat repeated selection on it as a source of optimism, and prefer changes that also hold up on the folds.

## Models compared

| Family | Unweighted | Weighted |
|---|---|---|
| Linear | `logreg` | `logreg_balanced` |
| Histogram gradient boosting (scikit-learn) | `hgb` (baseline) | `hgb_balanced` |
| Random forest | `rf` | `rf_balanced` |
| Extremely randomised trees | `extratrees` | — |
| LightGBM (optional) | `lgbm` | `lgbm_balanced` |
| XGBoost (optional) | `xgb` | `xgb_weighted` (`scale_pos_weight` recomputed inside each fit) |

All use fixed settings with no early stopping, so any of them can be retrained for submission exactly as validated. Hyperparameter tuning comes in Stage 6, using an inner chronological holdout inside each training window.

LightGBM and XGBoost are optional. If they aren't installed, the script says so and skips them. On macOS:

```bash
brew install libomp && pip install lightgbm xgboost
```

After installing, pin the versions:

```bash
python3 -c "import lightgbm, xgboost; print(f'lightgbm=={lightgbm.__version__}'); print(f'xgboost=={xgboost.__version__}')" >> requirements.txt
```

## Class weighting

Weighting the rare class changes the fitted model, not just the probability scale: the training objective changes, so splits and the resulting ranking can change too. The script reports each family's weighted-minus-unweighted deltas so the decision rests on evidence rather than on the assumption that imbalance always needs weighting.

**Compare within a family only.** Families differ in their settings, in how weights enter the objective and the regularisation, and in how the ratio is derived: scikit-learn's `balanced` recomputes it per fit, `balanced_subsample` recomputes it per bootstrap sample, and XGBoost's `scale_pos_weight` is a single ratio. A difference between two libraries is therefore not evidence about how each applies weighting.

`AutoScalePosWeight` in `src/models.py` sets XGBoost's `scale_pos_weight` inside every fit, from that split's training labels only. An earlier version used a single ratio computed over all labelled rows, which included each validation block's labels.

## How to run

From the repo root (expect roughly 10–25 minutes, mostly the forests):

```bash
python3 stage4/run_model_comparison.py
```

Useful options:

```bash
# only some models
python3 stage4/run_model_comparison.py --models hgb lgbm xgb hgb_balanced
# rerun after changes (experiment names must be unique)
python3 stage4/run_model_comparison.py --version s4v2
```

The script ends by printing the exact `make_submission.py` command for the winner, including its run id. Submitting is optional here: only submit if a candidate beats the baseline, and remember the 3-per-day limit.

## Outputs

- `stage4/model_comparison_<version>.csv`: every model's fold mean, standard deviation, minimum fold, horizon, weak-step and AP scores, plus deltas against the baseline and the accept/reject decision.
- `stage4/class_weight_comparison_<version>.csv`: the weighted-minus-unweighted deltas per family.
- One log row and one JSON record per model, as usual.

## What to send back

1. The three printed tables: all models, the comparison against the baseline, and class weighting.
2. The recommendation line.
3. Whether LightGBM and XGBoost were installed, so we know if those rows are missing.

## Notes

- Tree ensembles with `n_jobs=-1` can vary slightly between machines; LightGBM is configured with deterministic settings for this reason.
- Runtime warnings are counted per run, as in earlier stages. Logistic regression's warnings were investigated in Stage 3 and had no effect on its results.
- The weak steps were identified from a validation result, so treat them as a diagnostic, not a tuning target.
