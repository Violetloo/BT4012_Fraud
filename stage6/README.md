# Stage 6: drift, normalisation, training window and tuning

Stage 5 left a gap: about 0.983 on 5-step folds, 0.936 on the long-horizon mean, 0.949 on the leaderboard. Stage 6 attacks the drift behind that gap from four directions, all judged by the Stage 5 rule against the current best configuration (`hgb_balanced` on `raw_cmp`).

## Acceptance rule (unchanged)

A configuration replaces the baseline only if **all three** hold:

1. fold-mean AUC is not worse by more than 0.002,
2. the mean AUC over the four long-horizon splits improves, and
3. at least 3 of those 4 splits improve (a consistency check: the splits overlap and three cover 14 steps, one 11).

Stage 5 gave the first evidence this rule tracks the leaderboard: `raw_cmp` improved the multi-horizon mean by 0.0083 and the public score by 0.0046. One agreement is not proof, so treat a passing result as promising, not certain.

## a. Adversarial validation — which features drift

`stage6/adversarial_validation.py` trains a classifier to tell **early rows from late rows** using features alone. Its AUC says how far the distributions move; permutation importance ranks the features it leans on.

The comparison uses **steps 1–7 vs 8–15**, inside the earliest training window, so the ranking never sees a period that lies in the future of any validation block. No labels are used, and all rows take part, including unknown ones.

It writes `stage6/drift_drop_lists.json`, which defines `raw_cmp_drop5`, `raw_cmp_drop10` and `raw_cmp_drop20` (the current best set minus the most drift-prone raw features), then evaluates them against `raw_cmp`.

## b. Rank normalisation, training window, recency weighting

`stage6/run_experiments.py` tests six configurations against the baseline:

| Configuration | What changes |
|---|---|
| `rank_cmp` | features replaced by their percentile rank **within each time step**, plus component features |
| `rankraw_cmp` | raw features *and* their per-step ranks, plus component features |
| `minstep7`, `minstep12` | training rows before step 7 / 12 dropped (steps 1–6 are almost all licit); validation blocks untouched |
| `halflife5`, `halflife10` | recency weights `0.5 ** ((last_train_step - step) / h)` on training rows |

Ranks are computed within a step over all of that step's rows (train, unknown and test), using feature values only, so a row's rank never depends on another period. That is what removes level shifts between periods, and it is the transformation most likely to rescue signals like degree, which helped folds but hurt horizons in Stage 5.

## c. Tuning with an inner chronological holdout

`stage6/tune_hgb.py` runs a random search whose scores come only from **inner** splits carved out of training windows: for k = 15, 20, 25, 30 it trains on steps 1..k−5 and scores steps k−4..k. The outer folds and long-horizon blocks are never touched during the search, so the winning settings can be evaluated on them once, honestly. The number of iterations is a searched parameter rather than an early-stopping decision, so the submitted model matches the validated one exactly.

## How to run

```bash
python3 stage6/adversarial_validation.py     # ~15 min; writes the drop lists, then evaluates them
python3 stage6/run_experiments.py            # ~35 min; 7 configurations
python3 stage6/tune_hgb.py --n-trials 20     # ~30-60 min; search, then one honest evaluation
```

Each script has its own default `--version`; pass a new one (for example `--version s6exp2`) to rerun, since experiment names must be unique.

## Submitting a winner

Feature-set winners submit as usual. Configurations that change **training** rather than features need the matching argument, which `make_submission.py` now checks against the run record:

```bash
python3 stage3/make_submission.py --model hgb_balanced --features raw_cmp --run-id <id> --min-train-step 12
python3 stage3/make_submission.py --model hgb_balanced --features raw_cmp --run-id <id> --recency-half-life 10
```

For tuned settings, add the parameters to `src/models.py` as a named factory, rerun its evaluation under that name, then submit that name.

## What to send back

1. The adversarial AUC, the top drifting features, and the drop-set results.
2. The seven-configuration table from `run_experiments.py`.
3. The tuning summary: baseline inner AUC, best inner AUC, best parameters, and the outer comparison.

## Notes

- Nothing here changes the acceptance rule or the splits, so Stage 6 results are directly comparable with Stages 3–5.
- If several configurations pass, prefer the simplest, and consider combining only the ones that are conceptually independent — Stage 5 showed that combining two individually good feature groups can be worse than either.
- Keep submissions for configurations that pass the rule; you have 3 per day and the private leaderboard is what counts.
