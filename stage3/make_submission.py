"""Stage 3: retrain a validated configuration on all labelled data and write a Kaggle submission.

Run from the repo root, e.g.:
    python3 stage3/make_submission.py --model hgb --features raw_notime --run-id 20261001-101500-hgb_raw_notime_v2

Steps
1. Loads the run record experiments/runs/<run-id>.json and requires an exact match
   with what is about to be trained:
     - every estimator parameter (full get_params, not the printed representation),
     - the ordered feature list, and
     - the data hash (identifier, label and feature values of the training frame),
       which changes if the input files, feature-building logic or column order change.
   Refuses on any mismatch. Runs logged before data hashes existed are refused
   unless --allow-legacy-run (then only parameters and features are compared).
2. Fits once on every labelled training row (steps 1-35); unknown rows are excluded.
3. Validates the file: 15,329 rows, columns exactly index,target, index identical to
   sample_submission.csv in value and order, probabilities finite and within [0, 1].
4. Writes submissions/<date>_<model>_<features>.csv plus a JSON record (parameters,
   features, input hashes, code/git state, prediction summary, warning categories and
   messages).
Upload the CSV to Kaggle yourself, then record the public score with stage3/record_lb.py.
"""
import argparse
import hashlib
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
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.validation import (RUN_DIR, FEATURE_SET_DIR, _feature_hash, _git_state, _jsonable,  # noqa: E402
                            _params, code_hash, frame_hash, warning_summary)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--features", required=True)
p.add_argument("--run-id", required=True, help="run_id of the validation run this submission reproduces")
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--allow-legacy-run", action="store_true",
               help="accept a run recorded before data hashes existed (parameters and features still checked)")
p.add_argument("--min-train-step", type=int, default=None,
               help="drop training rows before this time step (must match the validated run)")
p.add_argument("--train-window", type=int, default=None,
               help="keep only the last N time steps of training data (must match the validated run)")
p.add_argument("--per-step-normalise", action="store_true",
               help="convert test predictions to percentile ranks within each time step "
                    "(post-processing; justify it with stage7/per_step_normalisation.py)")
p.add_argument("--smooth-mode", default=None, choices=["nbr_mean", "nbr_max", "comp_mean"],
               help="blend test predictions with neighbours'/component scores (post-processing; "
                    "justify with stage8/score_smoothing.py)")
p.add_argument("--smooth-alpha", type=float, default=0.2)
p.add_argument("--smooth-iters", type=int, default=1)
p.add_argument("--sample-weight", default=None, choices=["per_step_balanced"],
               help="named training weights (must match the validated run)")
p.add_argument("--recency-half-life", type=float, default=None,
               help="recency weighting half-life in time steps (must match the validated run)")
args = p.parse_args()

train, test = load_prepared(args.data_dir, feature_sets=[args.features])
sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
cols = get_feature_columns(args.features, train)
if args.model.startswith("stack"):
    from src.stacking import register_steps
    register_steps(train["time_step"])
    print("Registered row time steps for the stacking inner blocks")
if args.model.startswith("fsblend"):
    from src.ensembles import register_views, union_columns
    from src.features import FEATURE_SETS
    views = {}
    for n in ["raw_cmp", "raw_avl", "raw_all", "raw_notime", "raw_cmp_drop5"]:
        if n in FEATURE_SETS:
            try:
                views[n] = get_feature_columns(n, train)
            except FileNotFoundError:
                pass
    register_views(views)
    cols = union_columns(get_model_factory(args.model)().views)
    print(f"Registered {len(views)} feature views; union has {len(cols)} columns")
if args.model.startswith(("shift", "selftrain")):
    from src.adaptation import set_target
    set_target(test[cols])
    print(f"Target set to the {len(test):,} test rows (features only)")
if args.model.startswith(("pseudo", "pu_")):
    from src.pseudo import register_default_pool
    register_default_pool(train, cols)
    print(f"Registered the unlabelled pool ({int(train['label'].isna().sum()):,} rows)")
model = get_model_factory(args.model)()

# 1. Exact match with the validated run
rec_path = RUN_DIR / f"{args.run_id}.json"
if not rec_path.exists():
    sys.exit(f"No run record at {rec_path}")
run = json.loads(rec_path.read_text())
problems = []

params_now = {k: _jsonable(v) for k, v in _params(model).items()}
params_run = run.get("params", {})
diff = sorted(k for k in set(params_now) | set(params_run) if params_now.get(k) != params_run.get(k))
if diff:
    problems.append(f"model parameters differ: {diff[:10]}")

run_cols = run.get("feature_columns")
if run_cols is None:
    legacy_file = FEATURE_SET_DIR / f"{run.get('feature_hash')}.json"
    if args.allow_legacy_run and legacy_file.exists():
        run_cols = json.loads(legacy_file.read_text())
    else:
        problems.append("run record has no ordered feature list (legacy run; use --allow-legacy-run)")
if run_cols is not None and list(run_cols) != list(cols):
    problems.append("feature list differs from the validated run (content or order)")

weight_name = args.sample_weight
if args.recency_half_life is not None:
    weight_name = f"recency_halflife_{args.recency_half_life:g}"
if run.get("train_window") != args.train_window:
    problems.append(f"train_window differs: run {run.get('train_window')} vs {args.train_window}")
if run.get("min_train_step") != args.min_train_step:
    problems.append(f"min_train_step differs: run {run.get('min_train_step')} vs {args.min_train_step}")
if run.get("sample_weight_fn") != weight_name:
    problems.append(f"sample weighting differs: run {run.get('sample_weight_fn')} vs {weight_name}")

dhash = frame_hash(train, cols)
if "data_hash" in run:
    if run["data_hash"] != dhash:
        problems.append("data hash differs: input data or feature-building logic changed since validation")
elif not args.allow_legacy_run:
    problems.append("run record has no data hash (legacy run; use --allow-legacy-run)")

if problems:
    sys.exit("Refusing to build submission:\n  - " + "\n  - ".join(problems))
legacy = "data_hash" not in run
print(f"Matches validated run {args.run_id}"
      f"{' (legacy: data hash not checked)' if legacy else ' (parameters, ordered features and data hash)'}")
s = run["summary"]
print(f"Validation: fold AUC mean {s['auc_mean']:.4f}, min {s['auc_min']:.4f}, horizon {s['auc_horizon']:.4f}")

missing = [c for c in cols if c not in test.columns]
if missing:
    sys.exit(f"Test is missing feature columns: {missing[:5]}")

# 2. Fit on all labelled rows
lab = train[train["label"].notna()]
if args.min_train_step is not None:
    lab = lab[lab["time_step"] >= args.min_train_step]
if args.train_window is not None:
    lab = lab[lab["time_step"] > lab["time_step"].max() - args.train_window]
y = lab["label"].astype(int)
fit_kwargs = {}
if args.recency_half_life is not None:
    last = lab["time_step"].max()
    fit_kwargs["sample_weight"] = 0.5 ** ((last - lab["time_step"].values) / args.recency_half_life)
elif args.sample_weight == "per_step_balanced":
    counts = lab["time_step"].map(lab["time_step"].value_counts()).values
    w = 1.0 / counts
    fit_kwargs["sample_weight"] = w / w.mean()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.fit(lab[cols], y, **fit_kwargs)
    proba = model.predict_proba(test[cols])
classes = list(model.classes_)
pred = np.asarray(proba)[:, classes.index(1)]

if args.smooth_mode:
    from src.smoothing import smooth_scores
    _, _, edges_all = load_data(args.data_dir)
    pred = smooth_scores(pred, test["txId"].values, edges_all, args.smooth_mode,
                         args.smooth_alpha, args.smooth_iters)
    print(f"Smoothed predictions over the graph: mode={args.smooth_mode}, "
          f"alpha={args.smooth_alpha}, iters={args.smooth_iters}")

if args.per_step_normalise:
    from src.ensembles import per_step_rank
    pred = per_step_rank(pred, test["time_step"].values)
    print("Applied per-step rank normalisation to the test predictions")

# 3. Validate the submission
sub = pd.DataFrame({"index": test["index"].values, "target": pred})
checks = []
if len(sub) != len(sample):
    checks.append(f"row count {len(sub)} != sample {len(sample)}")
if not sub["index"].equals(sample["index"]):
    checks.append("index does not match sample_submission.csv in value and order")
if sub["index"].duplicated().any():
    checks.append("duplicate index values")
if not np.isfinite(sub["target"]).all():
    checks.append("non-finite predictions")
if ((sub["target"] < 0) | (sub["target"] > 1)).any():
    checks.append("predictions outside [0, 1]")
if list(sub.columns) != ["index", "target"]:
    checks.append(f"columns {list(sub.columns)} != ['index', 'target']")
if checks:
    sys.exit("Submission failed validation: " + "; ".join(checks))

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
wsum = warning_summary(caught)
data_dir = Path(args.data_dir)
record = {
    "file": csv_path.name, "created_utc": now.isoformat(timespec="seconds"),
    "model_name": args.model, "feature_set": args.features, "n_features": len(cols),
    "feature_hash": _feature_hash(cols), "feature_columns": cols, "data_hash": dhash,
    "input_file_sha256": {n: file_sha256(data_dir / n) for n in ["train.csv", "test.csv", "sample_submission.csv"]},
    "model": " ".join(repr(model).split()), "params": params_now,
    "validated_run_id": args.run_id, "legacy_run": legacy,
    "min_train_step": args.min_train_step, "train_window": args.train_window,
    "recency_half_life": args.recency_half_life, "sample_weight": args.sample_weight, "per_step_normalised": bool(args.per_step_normalise),
    "smoothing": None if not args.smooth_mode else {"mode": args.smooth_mode, "alpha": args.smooth_alpha,
                                                    "iters": args.smooth_iters},
    "training_rows": {"labelled": int(len(lab)), "illicit": int(y.sum()),
                      "time_steps": [int(lab["time_step"].min()), int(lab["time_step"].max())]},
    "prediction_summary": {"mean": float(pred.mean()), "n_unique": int(np.unique(pred).size),
                           "quantiles_0_25_50_75_90_99_100": [float(x) for x in q]},
    "warnings": wsum,
    "git_commit": commit, "git_dirty": dirty, "code_hash": code_hash(),
}
csv_path.with_suffix(".json").write_text(json.dumps(record, indent=2))

print(f"\nTrained on {len(lab):,} labelled rows ({int(y.sum()):,} illicit), {len(cols)} features.")
print(f"Test predictions: mean {pred.mean():.4f}, median {q[2]:.4f}, 99th pct {q[5]:.4f}, max {q[6]:.4f}, "
      f"distinct values {np.unique(pred).size:,}")
if caught:
    print(f"Warnings during fit/predict by category: {wsum['counts']} (messages saved in the JSON record)")
print(f"All submission checks passed.\nWrote {csv_path}\nRecord {csv_path.with_suffix('.json')}")
if dirty == "True":
    print("Note: project code has uncommitted changes; commit before submitting so the record is traceable.")
