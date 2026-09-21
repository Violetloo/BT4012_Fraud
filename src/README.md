# src: shared project code

- `data.py`: loads the supplied CSVs, parses labels strictly (1.0 / 0.0 / NaN; unexpected text raises an error), lists feature columns, and defines the six time-encoding candidates from Stage 1.
- `validation.py`: chronological splits (four 5-step folds plus a 14-step horizon split), `check_splits()`, `evaluate()` with built-in guards, full run records and the experiment log.

Scripts and notebooks import these from the repository root, e.g. `from src.validation import evaluate`.
