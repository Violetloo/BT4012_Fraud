# BT4012 — Detecting Illicit Bitcoin Transactions

Kaggle competition entry for BT4012 (Fraud Analytics). The task is to rank 15,329 Bitcoin
transactions from time steps 36–49 by their likelihood of being illicit, scored by ROC AUC.

| | Public AUC |
|---|---|
| Competition baseline | 0.92119 |
| First model in this pipeline (raw features) | 0.94351 |
| **Final submission (rank blend of five models)** | **0.96137** |

**Student number:** e1156454

---

## Quick start

The graded notebook reproduces the final submission from the four competition files and does
not depend on anything else in this repository.

```bash
git clone https://github.com/Violetloo/BT4012_Fraud.git
cd BT4012_Fraud
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
```

Place the four competition files in `data/` (they are not committed):

```
data/
├── train.csv
├── test.csv
├── txs_edgelist.csv
└── sample_submission.csv
```

Then open `notebooks/e1156454.ipynb` and run all cells, or run it headless:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute e1156454.ipynb \
  --output e1156454_executed.ipynb --ExecutePreprocessor.timeout=7200
```

Runtime is about an hour. The submission is written to `submissions/e1156454_submission.csv`.
See [`notebooks/README.md`](notebooks/README.md) for what each section of the notebook does.

---

## Approach in brief

The supplied features already contain statistics over each transaction's immediate neighbours.
One-step aggregates built on top of them did not improve performance; features summarising the
graph further out did.

- **Component features**: connected-component size, the share of the component with usable
  features, and degree relative to component size.
- **Multi-step walks**: P²X, P³X and P⁴X, with P the row-normalised symmetrised adjacency matrix.
- **PageRank propagation**: H ← (1 − α)PH + αX for ten iterations, at restart values from 0.05 to 0.30.
- **Coverage matching**: only labelled training rows and test rows contribute features to
  propagation, so neighbourhood coverage is comparable between the two periods.

Every model is scikit-learn's `HistGradientBoostingClassifier` with balanced class weights and
fixed settings. The final submission is a staged rank average of five such models that differ in
propagation features and, for one member, in histogram binning.

Validation is strictly chronological: four five-step folds and four long horizons, with
unlabelled rows excluded from fitting and scoring.

---

## Repository layout

```
BT4012_Fraud/
├── notebooks/
│   ├── e1156454.ipynb          graded notebook: rebuilds the final submission end to end
│   └── README.md
├── src/                        shared pipeline used by the stage scripts
├── stage1/ … stage25/          experiments, in the order they were run
├── report_check/               scripts that verify the report's figures against the logs
├── experiments/
│   ├── experiment_log.csv      one row per validated run (181 runs)
│   ├── leaderboard_scores.csv  public score for every submission
│   ├── runs/                   full JSON record per run: parameters, features, data hash, commit
│   └── oof/                    saved out-of-fold predictions per run
├── submissions/                generated submission files (not committed)
├── data/                       competition files (not committed)
└── requirements.txt            pinned dependencies
```

### `src/`

| Module | Purpose |
|---|---|
| `data.py` | Loading, label parsing, raw feature selection |
| `validation.py` | Chronological splits, the guarded `evaluate()` function, run logging |
| `experiments.py` | `run_config()` wrapper and result summaries |
| `features.py` | Named feature sets and their dependencies |
| `frames.py` | `load_prepared()`: attaches the requested feature blocks with caching |
| `graph_features.py` | Degree, availability, neighbour aggregates, component features |
| `propagation.py` | Multi-step walks, PageRank, directed, dispersion and contrast blocks |
| `knn_features.py` | Similarity-neighbour label features (Stage 19) |
| `models.py` | Named model factories |
| `ensembles.py`, `stacking.py` | Blends, seed ensembles, stacking |
| `pseudo.py`, `adaptation.py`, `two_stage.py` | Pseudo-labelling, covariate-shift weighting, two-stage scores |
| `drift.py`, `transforms.py`, `smoothing.py`, `tuning.py` | Drift analysis, rank transforms, score smoothing, nested tuning |

---

## Stages

Each stage folder contains the script for one line of investigation, and most write their
results to a CSV alongside it. Results below are the long-horizon mean AUC unless stated, against
the baseline current at the time.

| Stage | Question | Outcome |
|---|---|---|
| 1 | What does the data look like? | Six features encode time and were dropped; all known edges lie within one time step; test rows miss ~39% of neighbours |
| 2 | How should models be validated? | Chronological splits with guards against leakage; sanity checks all passed |
| 3 | Baselines and first submission | `hgb` on raw features, public 0.94406 |
| 4 | Which model family, and does class weighting help? | Histogram gradient boosting with balanced weights selected; weighting helped hgb and random forests, hurt XGBoost; a `scale_pos_weight` leak was found and fixed |
| 5 | Do graph features help? | One-hop aggregates did not; component features did, public 0.94864 |
| 6 | Can drift be corrected? | Dropping drift-prone features, rank normalisation, windows and recency weighting all rejected; a first tuning search leaked and was replaced by a properly nested one, which also did not help |
| 7 | Per-step normalisation and ensembles | Both harmful or neutral |
| 8 | Score smoothing and pseudo-labelling | Smoothing marginal; pseudo-labelling lost 0.04–0.06 |
| 9 | Adapting to the scored period | Covariate-shift weights collapsed to the clipping floor; self-training and a rank objective rejected |
| 10 | Does validation predict the leaderboard? | Nine pre-committed probes: rank correlation 0.00 |
| 11 | Blending scored submissions | Blends of near-equal models helped slightly |
| 12 | Stacking and positive-unlabelled learning | Stacking learned near-equal weights; PU learning lost heavily |
| 13 | Multi-step propagation and GraphSAGE | Walks of 2–4 steps reached public 0.95937; dropping the 1-step block helped; GraphSAGE scored 0.88213 |
| 14 | Directed and PageRank propagation | PageRank at α = 0.15 reached public 0.96034; mixing decays 0.96057; directed propagation rejected |
| 15 | Max-propagation and symmetric normalisation | Both rejected |
| 16 | Contrast features | Improved validation, lost publicly |
| 17 | Who contributes to propagation? | Admitting unlabelled rows or relaying through featureless nodes both scored lower |
| 18 | Dispersion and two-stage propagated scores | Two-stage improved validation, lost publicly |
| 19 | Similarity-neighbour label features | A leak was found and fixed; the corrected version lost publicly |
| 20 | Finer decays and tuning on the winning features | Six decays scored 0.96066; tuning on fixed inner blocks found nothing |
| 21–22 | CatBoost, bagging, slow-shallow settings | All rejected; slow-shallow improved every validation metric and lost publicly |
| 23 | Score smoothing in feature space | Rejected |
| 24 | Propagation iterations and histogram binning | More iterations worse; 127 bins scored 0.96073 and joined the final blend |
| 25 | Sharpening per-step score levels | Rejected in both directions |

Stages 21 and 22 were run as short inline scripts; their results are recorded in
`experiments/experiment_log.csv`.

---

## Final submission

Five models, each using the 159 raw features, the three component features and its own
propagation blocks:

| Member | Propagation | Public AUC |
|---|---|---|
| PPR | PageRank, α = 0.15 | 0.96034 |
| Walks 2–4 | P²X, P³X, P⁴X | 0.95937 |
| Three decays | α = 0.05, 0.15, 0.30 | 0.96057 |
| Six decays | adds α = 0.10, 0.20, 0.25 | 0.96066 |
| Three decays, 127 bins | as above, `max_bins=127` | 0.96073 |
| **Blend** | | **0.96137** |

The blend is built in four stages, each re-ranking the previous result, with r the rank scaled
to (0, 1) and ties averaged:

```
b1 = ½·r(PPR)  + ½·r(Walks 2–4)
b2 = ⅔·r(b1)   + ⅓·r(Three decays)
b3 = ¾·r(b2)   + ¼·r(Six decays)
s  = ¾·r(b3)   + ¼·r(Three decays, 127 bins)
```

The blend does not lead its members on any local validation summary; it was chosen on its
public score. The private-leaderboard score may differ.

---

## Reproducibility

- `notebooks/e1156454.ipynb` reproduces the submission from the four data files; its output
  scored 0.96137 on Kaggle.
- Each of the five members reproduces its originally uploaded predictions within numerical
  precision. The blended file differs on 1,125 of 15,329 predictions because rank averaging
  orders exact ties differently; this does not change the public score.
- `report_check/diagnose_reproduction.py` compares each member and each blending stage against
  the uploaded files.
- `report_check/rebuild_from_submissions.py` recomputes the report's results table from each
  submission's recorded validation run.

### Running a stage script

Stage scripts import from `src/` and expect the data in `data/`. Run them from the repository root:

```bash
python3 stage13/run_propagation.py
python3 stage3/make_submission.py --model hgb_balanced --features ppr_cmp --run-id <run_id>
```

`make_submission.py` refuses to build a file unless the model parameters, ordered feature list
and data hash match a logged validation run.

---

## Notes and limitations

- **Validation versus leaderboard.** Local validation identified clearly poor models but did not
  rank close candidates. Choices among the final candidates used public-leaderboard feedback,
  which introduces selection bias.
- **Transductive and retrospective features.** Propagation uses the test period's features, and
  a transaction's features depend on transactions that follow it. A live system would need
  features computable at prediction time.
- **Three leaks were caught and fixed:** a class-weight ratio computed over all labelled rows
  (Stage 4), a tuning search whose inner blocks overlapped the outer validation blocks (Stage 6),
  and a similarity feature that let validation rows see labels from their own block (Stage 19).
