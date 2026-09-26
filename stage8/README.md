# Stage 8a: graph score smoothing

Stage 5 tested neighbour **features** and they failed. This tests something different: blending each transaction's **predicted score** with the scores of the transactions it is connected to.

## Why this is a distinct idea

The model scores every row independently. It never sees that a transaction sits among others it scored highly. If illicit activity clusters in the graph — which is what laundering patterns look like — then neighbours' predictions carry information the model cannot use, no matter how good its features are.

This also explains why it is not the same as Stage 5. Neighbour features summarise neighbours' raw inputs; neighbour scores summarise the model's whole judgement about them, which is a far more compressed and relevant signal.

It is post-processing: no labels, no retraining, valid at submission time.

## Why validation transfers here

Only transactions that are being scored can contribute a score. In validation that means the labelled rows of the block, about 63% of a row's neighbours; at test time it means the test rows, about 61%. Those coverages match closely (the same match Stage 5 engineered for neighbour aggregates), so a gain measured on saved validation predictions should carry over.

## What is tested

`stage8/score_smoothing.py` rescores every saved prediction file under each combination of:

| Setting | Values |
|---|---|
| mode | `nbr_mean` (mean of neighbours), `nbr_max` (strongest neighbour), `comp_mean` (whole component within the step) |
| alpha | 0.1, 0.2, 0.3, 0.5 — weight on the neighbourhood score |
| iterations | 1, 2 — repeating lets information travel further than one hop |

Long-horizon splits (10+ steps) are the ones to read; they span many periods as the test set does. The folds are reported too, as a consistency check.

## How to run

```bash
python3 stage8/score_smoothing.py --contains raw_cmp
```

Minutes, not hours: no model is trained.

## Judging it

Accept a setting only if its mean delta on long splits is clearly positive **and** most cases improve. A setting that helps on average through one split is the pattern that failed to transfer in Stage 4.

Then submit it as post-processing:

```bash
python3 stage3/make_submission.py --model hgb_balanced --features raw_cmp --run-id <id> \
    --smooth-mode nbr_mean --smooth-alpha 0.2 --smooth-iters 1
```

The settings are recorded in the submission's JSON, alongside the validated run.

## If it fails

That is a clean result for the report: score-level graph information adds nothing beyond the supplied features, which already include neighbourhood aggregates. Combined with Stage 5, it would say the graph in this dataset has been thoroughly exploited by the features already provided.

---

# Stage 8b: pseudo-labelling

78% of `train.csv` (110,537 rows) has no label and has never been used for fitting. A first model scores those rows, the most confident predictions are added to the training set with their predicted labels, and the model is refitted.

## Keeping it leak-free

Inside **every** fit, the pool is restricted to unlabelled rows whose time step is at or before the last step of the training rows that fit received. A split validating steps 26–30 therefore never sees unlabelled rows from those steps, in any form. Test rows are never in the pool: their steps are later than any training window.

The fit log (`stage8/pseudo_fit_log_<version>.json`) records each fit's last training step and how many pool rows were available, so the restriction is auditable.

## Variants

| Model | Confidence band | Weight | Rounds |
|---|---|---|---|
| `pseudo_hgb` | p ≥ 0.99 or ≤ 0.01 | 1.0 | 1 |
| `pseudo_hgb_w05` | p ≥ 0.99 or ≤ 0.01 | 0.5 | 1 |
| `pseudo_hgb_loose` | p ≥ 0.95 or ≤ 0.05 | 0.5 | 1 |
| `pseudo_hgb_2rounds` | p ≥ 0.99 or ≤ 0.01 | 0.5 | 2 |

Down-weighting matters because pseudo-labels are guesses: at weight 1.0 they can outvote the real labels, since the pool is far larger than the labelled set.

## The known risk

Stage 1 found unlabelled rows differ from labelled ones (median KS 0.14 across features). Confident predictions on them may reflect the model's existing biases rather than new information, which is exactly how pseudo-labelling degrades a model. The script reports, per variant, how many rows each fit added and what share were pseudo-positive. Compare that share with the 11.7% illicit rate of the labelled data: a much lower share means the pool is being labelled almost entirely licit, which adds little.

## How to run

```bash
python3 stage8/run_pseudo_labelling.py        # ~40-60 minutes (each fit trains twice)
```

## Judging it

The usual rule, and with more scepticism than usual: this is the highest-variance change in the project, and validation has predicted the leaderboard correctly once in four attempts. A result inside ±0.002 on the multi-horizon mean is not worth a submission.
