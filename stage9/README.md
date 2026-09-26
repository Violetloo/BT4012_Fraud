# Stage 9 (corrected): adapting to the rows being scored

Stage 6 measured how far the feature distributions move between periods (early-vs-late adversarial AUC 0.961) and tried to cope by throwing information away: dropping drifting features, ranking within steps, shrinking the training window. Stage 9 instead reweights or extends the training set using the **features of the rows being scored**. Their labels are never used.

## What was wrong in the first version, and what changed

1. **The target was the wrong set of rows.** It was inferred as "the next 14 steps after the last training step", so fold1 adapted to steps 16–29 while scoring 16–20, and fold4's target reached into the test period. It also included unlabelled training rows, a population the test file does not contain. Now `evaluate(split_hook=...)` hands each split **exactly the rows it will score**, labels withheld, which is the same situation as submission time (target = the test rows).
2. **The density ratio was mis-scaled.** It is `p / (1 - p) × n_source / n_target`. The sample-size factor cancels under mean normalisation alone, but clipping happens first, so it changes which rows hit the clip and by how much.
3. **A collapse to constant weights was invisible.** Every fit now records the source probability quantiles before clipping, the share of weights at each clip boundary, the weight spread, and the effective sample size `(Σw)² / Σw²`. The script prints them and says plainly when weights have collapsed, in which case the configuration cannot differ from the baseline.
4. **Diagnostics could be lost.** The fit log is now written after every configuration, so an interrupted run keeps the evidence it has already produced.

A second domain classifier is also available. `shift_hgb`/`shift_hgb_clip3` use a flexible one; `shift_hgb_reg`/`shift_hgb_reg_clip3` use a deliberately smooth one (8 leaves, strong L2, 100 minimum samples per leaf). A flexible classifier that separates the periods almost perfectly pushes every source probability toward zero, which makes the ratios useless; a smoother one keeps them usable. Which behaviour actually occurs is now visible in the diagnostics rather than assumed.

## Configurations

| Name | What it does |
|---|---|
| `shift_hgb`, `shift_hgb_clip3` | importance weighting, flexible domain classifier, clip 10 / 3 |
| `shift_hgb_reg`, `shift_hgb_reg_clip3` | the same with the regularised domain classifier |
| `selftrain_target` | confident predictions on the scored rows added back as training rows |
| `per_step_balanced` | every time step contributes equal weight |
| `rank_xgb` | pairwise ranking objective instead of log loss (needs XGBoost) |

## How to run

```bash
# the corrected comparison first: one shift configuration against the baseline
python3 stage9/run_adaptation.py --models shift_hgb --skip-per-step --version s9fix

# then the rest, if the diagnostics show the weighting is doing something
python3 stage9/run_adaptation.py --models shift_hgb_clip3 shift_hgb_reg shift_hgb_reg_clip3 --version s9fix2
```

Leave `selftrain_target` aside until the simpler experiment behaves as intended.

## Reading the diagnostics

- **Source P(target) quantiles.** If the 99th percentile is still near zero, no training row looks like the rows being scored, and importance weighting has nothing to work with.
- **Share at the clip floor.** A large share means the clip, not the data, is setting most weights.
- **Effective sample size.** The fraction of rows the weighting effectively keeps. A very low value means the model is training on a small subset, which trades bias against variance and may hurt.
- **Weight spread.** Near zero means the configuration is the baseline in disguise.

## Limits on what these results can show

- A failed reweighting does **not** establish concept drift. Poor probability estimates, clipping, a population mismatch or insufficient overlap all produce the same outcome.
- This stage is not the first use of test features: the graph features (Stage 5) and rank transformations (Stage 6) were transductive too. What is new here is using the target's distribution to change **how training rows are weighted**.
- Stage 8b's pseudo-labelling failure was reported with a class-dilution explanation. That remains unsupported: the base model rebalances classes internally, so the mechanism is not established.

## Submitting

```bash
python3 stage3/make_submission.py --model shift_hgb --features raw_cmp --run-id <id>
python3 stage3/make_submission.py --model hgb_balanced --features raw_cmp --run-id <id> --sample-weight per_step_balanced
```

`make_submission.py` sets the target to the test rows for model names starting with `shift` or `selftrain`, which is the same code path validation used.
