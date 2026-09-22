"""Stage 3: retrain a validated configuration on all labelled data and write a Kaggle submission.

Run from the repo root, e.g.:
    python3 stage3/make_submission.py --model hgb --features raw_notime

Steps
1. Confirms this exact configuration (same model parameters and feature set) was
   evaluated and logged. Refuses otherwise, unless --force.
2. Fits the model once on every labelled training row (steps 1-35); unknown rows
   are excluded, as in validation. No settings are changed after validation.
3. Predicts test, then validates the file: 15,329 rows, columns exactly
   index,target, index identical to sample_submission.csv in the same order,
   probabilities finite and within [0, 1].
4. Writes submissions/<date>_<model>_<features>.csv plus a JSON record
   (model parameters, features, code/git state, training rows, prediction summary).
Upload the CSV to Kaggle yourself, then record the public score with
stage3/record_lb.py.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.reporting import read_log  # noqa: E402
from src.validation import _feature_hash, _git_state, _params, _jsonable, code_hash  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--features", required=True)
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--force", action="store_true", help="allow a configuration that was never evaluated")
args = p.parse_args()

train, test, _ = load_data(args.data_dir)
sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
cols = get_feature_columns(args.features, train)
make_model = get_model_factory(args.model)
model = make_model()
fhash = _feature_hash(cols)
model_repr = " ".join(repr(model).split())

# 1. Was this exact configuration validated?
log = read_log()
match = log[(log["feature_hash"] == fhash) & (log["model"] == model_repr)]
if match.empty and not args.force:
    sys.exit(f"No logged evaluation with this model and feature set (feature_hash {fhash}).\n"
             f"Run stage3/run_baselines.py first, or pass --force to override.")
validated = match.iloc[-1] if not match.empty else None
if validated is not None:
    print(f"Validated as {validated['run_id']}: fold AUC mean {validated['auc_mean']:.4f}, "
          f"min {validated['auc_min']:.4f}, horizon {validated['auc_horizon']:.4f}")

missing = [c for c in cols if c not in test.columns]
if missing:
    sys.exit(f"Test is missing feature columns: {missing[:5]}")

# 2. Fit on all labelled rows
lab = train[train["label"].notna()]
y = lab["label"].astype(int)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.fit(lab[cols], y)
    proba = model.predict_proba(test[cols])
classes = list(model.classes_)
pred = np.asarray(proba)[:, classes.index(1)]

# 3. Validate the submission
sub = pd.DataFrame({"index": test["index"].values, "target": pred})
problems = []
if len(sub) != len(sample):
    problems.append(f"row count {len(sub)} != sample {len(sample)}")
if not sub["index"].equals(sample["index"]):
    problems.append("index does not match sample_submission.csv in value and order")
if sub["index"].duplicated().any():
    problems.append("duplicate index values")
if not np.isfinite(sub["target"]).all():
    problems.append("non-finite predictions")
if ((sub["target"] < 0) | (sub["target"] > 1)).any():
    problems.append("predictions outside [0, 1]")
if list(sub.columns) != ["index", "target"]:
    problems.append(f"columns {list(sub.columns)} != ['index', 'target']")
if problems:
    sys.exit("Submission failed validation: " + "; ".join(problems))

# 4. Write file and record
now = datetime.now(timezone.utc)
out_dir = ROOT / "submissions"
out_dir.mkdir(exist_ok=True)
stem = f"{now:%Y%m%d-%H%M%S}_{args.model}_{args.features}"
csv_path = out_dir / f"{stem}.csv"
sub.to_csv(csv_path, index=False)

reread = pd.read_csv(csv_path)
assert list(reread.columns) == ["index", "target"] and len(reread) == len(sample), "re-read check failed"

commit, dirty = _git_state()
q = np.quantile(pred, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
record = {
    "file": csv_path.name, "created_utc": now.isoformat(timespec="seconds"),
    "model_name": args.model, "feature_set": args.features, "n_features": len(cols), "feature_hash": fhash,
    "model": model_repr, "params": {k: _jsonable(v) for k, v in _params(model).items()},
    "validated_run_id": None if validated is None else validated["run_id"],
    "training_rows": {"labelled": int(len(lab)), "illicit": int(y.sum()),
                      "time_steps": [int(lab["time_step"].min()), int(lab["time_step"].max())]},
    "prediction_summary": {"mean": float(pred.mean()),
                           "quantiles_0_25_50_75_90_99_100": [float(x) for x in q]},
    "n_warnings_during_fit": len(caught),
    "git_commit": commit, "git_dirty": dirty, "code_hash": code_hash(),
}
csv_path.with_suffix(".json").write_text(json.dumps(record, indent=2))

print(f"\nTrained on {len(lab):,} labelled rows ({int(y.sum()):,} illicit), {len(cols)} features.")
print(f"Test predictions: mean {pred.mean():.4f}, median {q[2]:.4f}, 99th pct {q[5]:.4f}, max {q[6]:.4f}")
if caught:
    print(f"Warnings captured during fit/predict: {len(caught)} (see categories in the JSON record count)")
print(f"All submission checks passed.\nWrote {csv_path}\nRecord {csv_path.with_suffix('.json')}")
if dirty == "True":
    print("Note: project code has uncommitted changes; commit before submitting so the record is traceable.")
