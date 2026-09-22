# Stage 3: baselines, time-feature comparison and first submission

Stage 3 trains the first real models through `evaluate()`, decides whether to keep the six time-encoding features excluded, and produces the first Kaggle submission.

## Files

| File | Purpose |
|---|---|
| `src/features.py` | Named feature sets: `raw_notime` (159 features) and `raw_all` (165). Later stages add graph features here |
| `src/models.py` | Named model factories: `logreg` and `hgb`, every `random_state` fixed |
| `src/reporting.py` | Compares two `evaluate()` results; reads and updates the experiment log |
| `stage3/run_baselines.py` | Runs 2 models × 2 feature sets, prints the time-feature comparison and a ranking |
| `stage3/make_submission.py` | Retrains a validated configuration on all labelled rows and writes a checked submission file |
| `stage3/record_lb.py` | Writes the Kaggle public score back into the experiment log |

## Models

- **`logreg`**: standardised logistic regression. A simple linear reference point.
- **`hgb`**: scikit-learn's `HistGradientBoostingClassifier` (learning rate 0.05, 300 iterations, 31 leaves). It is the same family as LightGBM but needs no extra installation. LightGBM and XGBoost are compared in Stage 4.

Both use fixed settings with no early stopping and no class weighting. The model retrained for submission therefore uses exactly the validated configuration, and class weighting is tested in Stage 4 rather than assumed.

## Time-feature decision rule

The Stage 1 starting hypothesis is to exclude `feat_100`, `feat_101`, `feat_103`, `feat_136`, `feat_137` and `feat_139`. `run_baselines.py` compares `raw_all` with `raw_notime` for each model on identical splits and suggests including them only if all three hold:

1. the mean fold AUC improves,
2. at least 3 of the 4 folds improve, and
3. the horizon AUC does not get worse.

Otherwise they stay excluded. The rule is deliberately conservative: these features are almost perfectly ordered by time, and test steps (36–49) lie beyond all training steps, so a gain on 5-step validation blocks may not carry over. If the two models disagree, prefer the `hgb` result, since tree ensembles are the main model family from here on, and note the disagreement in your report.

## Choosing the configuration to submit

`run_baselines.py` ranks the four configurations by mean fold AUC. Submit the top one unless a close runner-up (within about 0.005 in mean fold AUC) has a clearly better minimum fold and horizon AUC. In that case prefer the more stable one. Record the reason either way.

## How to run

From the repo root, with the competition CSVs in `data/`:

```bash
# 1. Baselines and time-feature comparison (a few minutes)
python3 stage3/run_baselines.py

# 2. Build the submission for your chosen configuration, e.g.
python3 stage3/make_submission.py --model hgb --features raw_notime

# 3. Upload the CSV on the competition's Submit page, then record the public score, e.g.
python3 stage3/record_lb.py --submission submissions/<file>.csv --score 0.XXXXX
```

To rerun baselines after a code change, use a new suffix: `python3 stage3/run_baselines.py --version v2`. Reused experiment names are rejected.

## What each script checks

- **`make_submission.py`** refuses a configuration that was never evaluated (the same model parameters and feature set must appear in the log). It trains on all labelled rows, steps 1–35, excluding unknowns. The file must have 15,329 rows and exactly the columns `index,target`, with `index` identical to `sample_submission.csv` in value and order, and every prediction finite and within [0, 1]. It re-reads the written file to confirm, and saves a JSON record next to it with the model parameters, features, code and git state, and a prediction summary.
- **`record_lb.py`** attaches the public score to the validated run through that JSON record, so each leaderboard score is traceable to its validation result.

## What to send back

1. The full output of `run_baselines.py`: both comparison tables with their suggestions, and the ranking.
2. The output of `make_submission.py`.
3. The public leaderboard score, after recording it.

## Notes

- Commit code before building a submission. The JSON record stores the commit and a `git_dirty` flag, and the script warns if code has uncommitted changes.
- `submissions/` is git-ignored. Keep the JSON records locally; they document which file produced which score.
- The public leaderboard uses 30% of test. Treat it as a sanity check, not as the basis for choosing models.
