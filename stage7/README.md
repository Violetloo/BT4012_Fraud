# Stage 7: prediction normalisation and ensembling

Stage 6 found no improvement anywhere: drift is real (early-vs-late AUC 0.9610) but dropping drifting features, ranking within steps, shrinking the training window and (properly nested) tuning all failed. Stage 7 tries the two remaining cheap ideas, both of which change how predictions are combined rather than what the model learns.

## a. Per-step prediction normalisation

The competition's AUC pools all 15,329 test rows across 14 time steps. A row in step 36 is ranked against rows in step 49. If the model's scores sit systematically higher in some periods than others, that pooled ranking suffers **even when the ordering inside each step is good**. Converting predictions to percentile ranks within each time step removes those level differences.

`stage7/per_step_normalisation.py` needs no retraining: it reads the saved validation predictions in `experiments/oof/` and recomputes each split's AUC with and without normalisation. The long-horizon splits are the ones to read, because they span many steps as the test set does; the 5-step folds span fewer and will show less.

Normalisation uses predictions and time steps only, never labels, so it is valid at submission time: `stage3/make_submission.py --per-step-normalise`.

## b. Seed ensembles and blends

`stage7/run_ensembles.py` evaluates, all on `raw_cmp` and against `hgb_balanced`:

| Model | What it is |
|---|---|
| `hgb_balanced_seed10` | the same model fitted 10 times with different randomness |
| `blend_hgb_lgbm`, `blend_hgb75_lgbm25` | hgb + LightGBM at 50/50 and 75/25 |
| `blend_hgb_xgb` | hgb + XGBoost, 50/50 |
| `blend_hgb_lgbm_xgb` | all three, equal weights |
| `blend_seed10_lgbm` | the seed ensemble + LightGBM, 75/25 |

**Members are combined by average rank, not average probability.** AUC depends only on ordering, and these models are calibrated differently (their mean predictions differ), so rank averaging stops one model's scale dominating.

**Why the seed ensemble needs more than seeds.** Histogram gradient boosting with no row or feature subsampling is deterministic given the data, so ten seeds would give ten identical models. `SeedEnsemble` therefore injects diversity two ways: each member considers a random 80% of features per split, and each is fitted with Poisson(1) sample weights, the standard online-bagging equivalent of resampling rows with replacement. Neither changes the data or the labels.

Only a handful of fixed weights are tried. Searching many weights on the same splits would fit the validation data rather than reveal a real effect.

## How to run

```bash
python3 stage7/per_step_normalisation.py --contains raw_cmp   # seconds; no retraining
python3 stage7/run_ensembles.py                               # ~30-50 minutes
```

The first script is worth running first: if per-step normalisation helps on the long splits, it is free and applies to whatever model you end up submitting.

## Judging the results

The usual rule applies to the ensembles: fold mean not worse by more than 0.002, multi-horizon mean improved, at least 3 of 4 splits improved.

For normalisation, look at the mean delta on splits covering 10+ steps. A clearly positive mean across many runs is evidence; a mean near zero means the pooled ranking is not the problem.

Keep expectations calibrated. Across Stages 4-6, about 25 configurations were tested and one improved the leaderboard (component features, +0.0046). Ensembles typically give 0.002-0.005, which is inside this leaderboard's noise of roughly 0.008. The strongest argument for them is variance reduction rather than a higher expected score.

## Submitting

```bash
python3 stage3/make_submission.py --model <name> --features raw_cmp --run-id <id> [--per-step-normalise]
```

`--per-step-normalise` is post-processing, recorded in the submission's JSON. Use it only if part (a) supports it.
