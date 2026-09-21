# BT4012 Fraud Detection

Predict the probability that each test Bitcoin transaction is illicit. The competition evaluates ROC AUC.

## Current status

Stage 1 checks and findings are in [stage1/README.md](stage1/README.md). Additional time-feature and drift checks discussed separately have not yet been incorporated into the inspection script. No model has been trained or submitted.

## Layout

```text
BT4012_Fraud/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # Local competition CSVs; ignored by Git
├── stage1/                       # Inspection script, findings and outputs
├── src/                          # Future validation and feature code
├── experiments/
│   └── experiment_log.csv
├── notebooks/                    # Future <student_number>.ipynb
├── submissions/                  # Generated predictions; ignored by Git
└── report/                       # Future <student_number>.pdf
```

## Setup

Run these commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python stage1/inspect_data.py
```

The inspection script reads `data/` by default and regenerates outputs in `stage1/`. For another source folder, use `--data-dir /path/to/files`. The dependencies are pinned to the versions used for the Stage 1 run; extend them and verify a fresh environment as modeling progresses.

Required data files: `train.csv`, `test.csv`, `txs_edgelist.csv`, and `sample_submission.csv`. Local copies are included in `data/`; original Downloads files have been retained. Input hashes are recorded in `stage1/diagnostics.json`. Data and prediction CSVs are excluded from Git.

## Modeling and deliverables

Use only supplied competition data. Unknown labels are excluded from supervised fitting and scoring. Validation will use chronological splits. Do not supplement the data with outside labels or reconstructed dataset-specific solutions.

The final student-number notebook must reproduce preprocessing, feature engineering, validation and submission generation. The final report must be at most four pages and contain the GitHub repository link.

According to the provided instructions, Kaggle submissions close on 24 October 2026 at 11:55 PM Singapore time. The report is due on 25 October 2026 at 11:59 PM Singapore time.
