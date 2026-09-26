# Stage 9: adapting to the period being scored

Every stage so far learned from the training periods and hoped the result carried forward. Stage 6 measured how badly that assumption holds: a classifier separates early from late periods with AUC 0.961. The responses tried then — dropping drifting features, ranking within steps, shrinking the window — all threw information away. Stage 9 does the opposite: it uses the **feature distribution of the period being predicted**.

Labels are never used. The competition supplies `test.csv`, so this is transductive learning, and it is the first thing in the project to use the test features at all.

## Configurations

| Name | What it does |
|---|---|
| `shift_hgb`, `shift_hgb_clip5`, `shift_hgb_clip3` | Importance weighting for covariate shift: a classifier separates training rows from target-period rows, and each training row is weighted by its density ratio `p / (1 - p)`, clipped at 10 / 5 / 3. Rows that resemble the period being predicted count more. Weights are **cross-fitted**, so a row's weight never comes from a model fitted on that row. |
| `selftrain_target`, `selftrain_target_2rounds` | Confident predictions on the **target period's** rows are added back as training rows. Stage 8b did this with unlabelled *training* rows and failed (97.5% pseudo-licit, diluting the rare class); here the aim is adapting the decision boundary to the distribution being scored, not adding volume. |
| `per_step_balanced` | Training rows weighted so each time step contributes equally, instead of the largest steps dominating. |
| `rank_xgb` | A pairwise ranking objective instead of log loss, since the metric is AUC. Scores are mapped to (0, 1) by rank, which leaves AUC unchanged. Requires XGBoost. |

## How the target window is chosen

Each fit infers it from the rows it was given: the target is the steps **after** the last training step, up to 14 ahead. In validation that is the block being predicted; at submission time it is the test period (steps 36–49). The same code therefore runs in validation and at submission, which is what makes the validation estimate meaningful.

## Why this differs from what failed before

- Recency weighting (Stage 6) assumed a decay; these weights are **measured** against the actual target distribution.
- Dropping drifting features (Stage 6) removed signal; reweighting keeps every feature and changes only which rows the model listens to.
- Pseudo-labelling training rows (Stage 8b) added mostly-licit volume; self-training on the target period changes where the boundary sits in the region being scored.

## How to run

```bash
python3 stage9/run_adaptation.py            # ~45-70 minutes
# or a subset
python3 stage9/run_adaptation.py --models shift_hgb shift_hgb_clip5 --version s9a
```

The script reports, per configuration, the weight range the shift models produced and how many target rows self-training added, with the pseudo-positive share to compare against the 11.7% illicit rate.

## Judging and submitting

The usual rule: fold mean not worse by more than 0.002, multi-horizon mean improved, at least 3 of 4 splits improved.

```bash
# a model
python3 stage3/make_submission.py --model shift_hgb --features raw_cmp --run-id <id>
# weights rather than a model
python3 stage3/make_submission.py --model hgb_balanced --features raw_cmp --run-id <id> \
    --sample-weight per_step_balanced
```

`make_submission.py` registers the target rows automatically for model names starting with `shift` or `selftrain`, and checks named weights against the validated run.

## Interpreting a failure

If reweighting toward the target period does not help either, that is a strong statement for the report: the drift is not a covariate shift that can be corrected by reweighting, which points to the relationship between features and labels changing over time (concept drift) rather than just the inputs moving. No amount of reweighting fixes that; only labels from the newer period would.
