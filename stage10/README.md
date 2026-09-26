# Stage 10: feature-set blending and a fixed probe list

Two ideas, both aimed at the 0.0066 gap between your best (0.94864) and the leader (0.955).

## a. Blending across feature sets

Stage 7 blended hgb, LightGBM and XGBoost **on the same features** and gained nothing: the members made nearly the same errors. Feature diversity is a different axis, and the views have measurably different strengths:

| View | Best at |
|---|---|
| `raw_avl` | h15 (0.9319), best multi-horizon mean of any set (0.9331) |
| `raw_cmp` | h18 (0.9096) and h21 (0.9172) |
| `raw_all` | best fold mean in Stage 3 |

`FeatureSetBlend` fits the same model on each view and combines by average rank. `evaluate()` gets the union of the views' columns; each member sees only its own view. Blends tested: `fsblend_cmp_avl`, `fsblend_cmp2_avl1` (2:1), `fsblend_cmp_avl_all`, `fsblend_cmp_avl_notime`, `fsblend_cmp_avl_drop5`.

```bash
python3 stage10/run_feature_blends.py      # ~40-60 minutes
```

## b. A fixed probe list

Local validation has agreed with the leaderboard once in four attempts, and the differences now in play (0.002–0.005) are smaller than both its noise and the leaderboard's (~0.008). With 3 submissions a day and weeks left, the leaderboard is itself a measurement instrument.

The discipline that keeps this from becoming public-leaderboard overfitting is that **the candidate list is fixed in advance** and does not grow around whichever result looks best:

| Candidate | Why it is worth a probe |
|---|---|
| `avl` | best multi-horizon feature set, never submitted |
| `avl_cmp` | the two useful graph groups together |
| `all` | best fold mean in Stage 3; keeps the time-encoding features |
| `per_step` | best fold mean (0.9860) and best worst-fold (0.9593) of anything run |
| `lgbm` | a different library on the best feature set |
| `blend_lgbm` | library blend, near-identical validation |
| `fsblend_cmp_avl`, `fsblend_cmp_avl_all` | blends across feature sets |
| `drop5` | drift-prone features removed |

```bash
python3 stage10/build_probes.py                      # validates and builds all of them
python3 stage10/build_probes.py --only avl per_step  # or a subset
```

Each candidate is validated first (a logged run with the usual guards), then built through the normal `make_submission` checks. Nothing is uploaded for you. The plan is saved to `stage10/probe_plan_<version>.csv`.

## Rules for the probing phase

1. Upload at most 3 per day and record each with `stage3/record_lb.py`.
2. Do not add candidates once probing has started.
3. Treat differences under about 0.008 as unresolved: a candidate scoring 0.951 is not reliably better than one scoring 0.947.
4. For the two final selections, keep one on the best public score and one on the best-validated model. The private leaderboard uses the other 70% of the test data, and a candidate chosen purely on public score can fall back.

## What to expect

Honestly: a feature-set blend might add 0.002–0.005 if the views are diverse enough, and probing might find a candidate a few thousandths luckier on this particular test split. Together that could plausibly reach 0.952–0.955, but a good share of any gain will be the sampling noise of the public split rather than a better model.
