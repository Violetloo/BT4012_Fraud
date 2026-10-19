# Notebook: `e1156454.ipynb`

The graded notebook for BT4012. It rebuilds the final Kaggle submission (public AUC 0.96137)
from the four competition files, and documents the preprocessing, feature engineering,
validation and submission steps along the way.

It imports nothing from `src/`, so it runs on a fresh clone with only the data present.

## Requirements

- Python 3.11 or later
- Packages from the repository's `requirements.txt`: numpy, pandas, scipy, scikit-learn,
  matplotlib
- The four competition files in `../data/`:
  `train.csv`, `test.csv`, `txs_edgelist.csv`, `sample_submission.csv`

## Running

From this folder, in Jupyter or VS Code, run all cells. Or headless:

```bash
jupyter nbconvert --to notebook --execute e1156454.ipynb \
  --output e1156454_executed.ipynb --ExecutePreprocessor.timeout=7200
```

Runtime is about an hour: 40 validation fits (five configurations across eight splits) and five
final fits.

The paths in the setup cell assume this notebook sits in `notebooks/` with the data in `data/`.
If you launch Jupyter from the repository root instead, change `DATA_DIR` to `Path("data")` and
`OUT_DIR` to `Path("submissions")`.

## Outputs

| File | Contents |
|---|---|
| `../submissions/e1156454_submission.csv` | The submission: `index`, `target`, 15,329 rows |
| `../submissions/e1156454_validation_predictions.csv` | Row-level validation predictions for every model and the ensemble, by split |

## Sections

| Section | What it does |
|---|---|
| Setup | Imports, seed, paths, library versions |
| 1. Loading the data | Parses labels explicitly, prints file hashes and dataset sizes |
| 2. Inspection | Asserts structural properties (unique IDs, period separation, finite values, same-step edges, no conflicting duplicate labels); drops six time-encoding features; plots the illicit rate by time step |
| 3. Graph features | Component features, PageRank propagation at six restart values, and walks of 2–4 steps, with coverage matching |
| 4. Validation | Four five-step folds and four long horizons; fits all five models per split and scores both members and the ensemble |
| 5. Results | Validation and leaderboard results, and what they show |
| 6. Final model | Refits on all labelled rows, applies the staged rank blend, checks the format, writes the submission |
| Optional checks | Compares the output against the originally uploaded files; skipped automatically when those files are absent |
| 7. Conclusion | Findings, limitations and what would change |

## Expected results

Validation (ROC AUC):

| | Fold mean | Long-horizon mean | Worst long horizon |
|---|---|---|---|
| Best single member | 0.9865 | 0.9401 | 0.9056 |
| Ensemble | 0.9864 | 0.9385 | 0.9035 |

The ensemble does not lead its members locally; it was selected on public score.

The written submission scored **0.96137** on the public leaderboard. Each of the five member
models matches its originally uploaded predictions within numerical precision; the blended file
differs from the original on a small number of predictions because rank averaging orders exact
ties differently, and both files scored the same.

## Optional comparison cells

The last two code cells compare this run against submission files that were uploaded during the
competition. They are not in the repository, so on a fresh clone these cells print a message and
skip. Nothing else depends on them.
