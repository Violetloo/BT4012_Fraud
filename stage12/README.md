# Stage 12: stacking and positive-unlabelled learning

The leader is at 0.96362 and you are at 0.95060: a gap of 0.013, well outside the ~0.008 leaderboard noise. That is structural, so this stage tries two techniques that are different in kind rather than further refinements.

## Stacking: learn the combination

Every blend so far used weights chosen by hand. Stacking learns them. Inside `fit()`:

1. chronological blocks are carved out of the **training** rows;
2. each base model is fitted on the earlier rows and predicts the held-out block;
3. a meta-model (logistic regression) is fitted on those out-of-fold predictions;
4. the bases are refitted on all the training rows.

At scoring time the bases predict and the meta-model combines them. The outer validation block is never involved. Base predictions are rank-transformed first, so no base's probability scale dominates, and the meta-model's coefficients are logged, which shows which bases actually carry weight.

| Model | Bases |
|---|---|
| `stack_hgb_lgbm` | hgb_balanced, lgbm_balanced |
| `stack_three` | + rf_balanced |
| `stack_four` | + xgb |

Unavailable bases (if LightGBM or XGBoost is missing) are dropped automatically.

## Positive-unlabelled learning: use all 110,537 unlabelled rows

Stage 8b added only **confident** pseudo-labels and lost 0.04–0.06: thresholding selected rows the model already agreed with, reinforcing its own boundary. The positive-unlabelled treatment is different. "Unlabelled" here means "mostly negative but unverified", so **every** unlabelled row is added as a negative with a small weight. No threshold, no selection, and ten times more rows shaping the boundary. Stage 8b's own diagnostic supports the premise: 97.5% of unlabelled rows were predicted licit.

| Model | Weight per unlabelled row |
|---|---|
| `pu_w005` | 0.05 |
| `pu_w01` | 0.1 |
| `pu_w03` | 0.3 |

The pool is restricted inside each fit to unlabelled rows at or before the last training step, so a validation block's rows are never used. The script prints the effective extra weight against the number of real labelled rows, which is the number that decides whether the unlabelled rows are a nudge or a takeover.

## How to run

```bash
python3 stage12/run_stack_pu.py                 # ~60-90 minutes for all six
python3 stage12/run_stack_pu.py --models pu_w01 stack_three --version s12a   # a quicker subset
```

## Judging

The usual rule applies, but with one adjustment that the evidence now supports: across nine probes, validation rank correlated 0.00 with the leaderboard. So treat a configuration that is merely **close** to the baseline as a candidate worth a submission slot, not a failure — especially stacking, whose benefit (learned combination weights) is the kind that shows up on test rather than on splits the meta-model was not tuned for.

## Submitting

```bash
python3 stage3/make_submission.py --model stack_three --features raw_cmp --run-id <run_id>
python3 stage3/make_submission.py --model pu_w01 --features raw_cmp --run-id <run_id>
```

`make_submission.py` registers the unlabelled pool for `pu_*` models and the row time steps for `stack_*` models, so the submission reproduces exactly what was validated.

## Honest expectation

Stacking commonly adds 0.005–0.01 over hand-weighted blends when the bases are genuinely different; yours are moderately different. Positive-unlabelled learning is the higher-variance bet: it either sharpens the boundary with far more data, or drags it toward the unlabelled population. Neither is likely on its own to close a 0.013 gap, but they are the two largest untried levers.
