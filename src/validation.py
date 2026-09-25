"""Chronological validation, a reusable evaluate() function and the experiment log.

Design (see stage2/README.md for the reasoning):
- Four expanding-window forward-chaining folds with 5-step validation blocks.
- One separate 14-step "horizon" split that mimics the test horizon. It is
  reported separately and is not averaged into the fold score.
- Unknown labels are excluded from fitting and scoring in every split.

What the built-in guards establish: rows are separated correctly (validation
strictly later than training, no shared transactions, no identifier/label
columns as predictors) and outputs are well formed. They cannot certify that a
feature itself is free of leakage; feature builders must never use label values.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import REPO_ROOT


@dataclass(frozen=True)
class Split:
    name: str
    train_steps: tuple  # inclusive (first, last)
    val_steps: tuple    # inclusive (first, last)

    def train_mask(self, steps):
        return steps.between(*self.train_steps)

    def val_mask(self, steps):
        return steps.between(*self.val_steps)


FOLDS = [
    Split("fold1", (1, 15), (16, 20)),
    Split("fold2", (1, 20), (21, 25)),
    Split("fold3", (1, 25), (26, 30)),
    Split("fold4", (1, 30), (31, 35)),
]
HORIZON = Split("horizon", (1, 21), (22, 35))

# Long-horizon splits, for checking that a change holds up over long ranges rather
# than on the single horizon split. Three validate 14 steps (h15, h18, h21) and one
# validates 11 steps (h24: steps 25-35, the end of the training data). They overlap
# heavily and share training periods, so "3 of 4 improve" is a consistency check, not
# four independent confirmations. h21 is the same split as HORIZON. Stage 4 showed a
# gain on one horizon split need not transfer to the test set.
MULTI_HORIZONS = [Split(f"h{k}", (1, k), (k + 1, min(k + 14, 35))) for k in (15, 18, 21, 24)]

EXPERIMENT_DIR = REPO_ROOT / "experiments"
LOG_PATH = EXPERIMENT_DIR / "experiment_log.csv"
PRED_DIR = EXPERIMENT_DIR / "oof"
RUN_DIR = EXPERIMENT_DIR / "runs"
FEATURE_SET_DIR = EXPERIMENT_DIR / "feature_sets"
CODE_GLOBS = ["src/*.py", "stage*/*.py"]

LOG_COLUMNS = (
    ["run_id", "timestamp_utc", "name", "description", "model", "random_states", "n_features", "feature_hash"]
    + [f"auc_{s.name}" for s in FOLDS]
    + ["auc_mean", "auc_std", "auc_min", "ap_mean", "auc_horizon", "ap_horizon",
       "n_convergence_warnings", "n_runtime_warnings", "n_other_warnings",
       "splits", "runtime_s", "git_commit", "git_dirty", "code_hash", "public_lb", "notes"]
)


# ---------------------------------------------------------------------------
# Split checks
# ---------------------------------------------------------------------------
def _require_both_classes(y, where):
    if len(y) == 0:
        raise ValueError(f"{where}: no labelled rows")
    if y.nunique() < 2:
        raise ValueError(f"{where}: only one class present ({y.unique().tolist()})")


def check_splits(train, folds=FOLDS, horizon=HORIZON):
    """Validate split definitions and return labelled class counts per split."""
    max_step = int(train["time_step"].max())
    steps = train["time_step"]
    rows = []
    for s in list(folds) + [horizon]:
        if s.train_steps[0] < 1 or s.val_steps[1] > max_step:
            raise ValueError(f"{s.name}: steps out of range")
        if s.train_steps[1] >= s.val_steps[0]:
            raise ValueError(f"{s.name}: validation must come strictly after training")
        for part, mask in [("train", s.train_mask(steps)), ("val", s.val_mask(steps))]:
            sub = train[mask]
            lab = sub["label"].notna()
            _require_both_classes(sub.loc[lab, "label"], f"{s.name} {part}")
            rows.append({
                "split": s.name, "part": part,
                "steps": f"{s.train_steps if part == 'train' else s.val_steps}",
                "rows_total": len(sub), "unknown_excluded": int((~lab).sum()),
                "labelled": int(lab.sum()), "licit": int((sub["label"] == 0).sum()),
                "illicit": int((sub["label"] == 1).sum()),
                "illicit_rate": float(sub["label"].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_state():
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return "none", "unknown"
    # Only code counts as "dirty": regenerated outputs (logs, CSVs) are expected to change.
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--",
                 *[f":(glob){g}" for g in CODE_GLOBS], ":(glob)**/*.ipynb")
    return commit, ("unknown" if dirty is None else str(bool(dirty)))


def code_hash():
    """SHA-256 over the project's Python code, identifying it even when uncommitted."""
    h = hashlib.sha256()
    for pattern in CODE_GLOBS:
        for path in sorted(REPO_ROOT.glob(pattern)):
            h.update(str(path.relative_to(REPO_ROOT)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def frame_hash(frame, feature_cols):
    """Content hash of the identifier, label and feature values used for training.

    Changes if the input data, the feature-building logic or the column order
    changes, so a submission can prove it used exactly the validated inputs.
    Computed over all rows passed in (including unknown-label rows).
    """
    cols = ["txId", "time_step", "label"] + list(feature_cols)
    values = pd.util.hash_pandas_object(frame[cols], index=False).values
    return hashlib.sha256(values.tobytes() + json.dumps(cols).encode()).hexdigest()[:16]


def warning_summary(caught):
    """Counts per warning category plus up to 20 distinct messages."""
    counts = {}
    for w in caught:
        counts[w.category.__name__] = counts.get(w.category.__name__, 0) + 1
    messages = sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})[:20]
    return {"counts": counts, "messages": messages}


def _feature_hash(cols):
    return hashlib.md5(json.dumps(sorted(cols)).encode()).hexdigest()[:8]


def _params(model):
    return model.get_params(deep=True) if hasattr(model, "get_params") else {}


def _random_states(model):
    return {k: v for k, v in _params(model).items() if k == "random_state" or k.endswith("__random_state")}


def _jsonable(v):
    return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)


def _splits_text(splits):
    return ";".join(f"{s.name}:{s.train_steps[0]}-{s.train_steps[1]}>{s.val_steps[0]}-{s.val_steps[1]}" for s in splits)


def _check_log_header():
    if LOG_PATH.exists():
        with open(LOG_PATH, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != LOG_COLUMNS:
            raise ValueError(f"{LOG_PATH} has a different column layout (older version). "
                             "Archive or delete it before logging new runs.")


def _name_already_logged(name):
    if not LOG_PATH.exists():
        return False
    return name in set(pd.read_csv(LOG_PATH, usecols=["name"])["name"].astype(str))


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------
def _fit_predict(make_model, X_tr, y_tr, X_va, where, sample_weight=None):
    """Fit a fresh model and return class-1 probabilities plus captured warnings."""
    model = make_model()  # fresh, unfitted model: preprocessing inside it is fit on this fold only
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if sample_weight is None:
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        proba = model.predict_proba(X_va)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"{where}: model.classes_ {classes} does not contain class 1")
    p = np.asarray(proba)[:, classes.index(1)]
    if not np.isfinite(p).all():
        raise ValueError(f"{where}: non-finite predicted probabilities")
    if (p < 0).any() or (p > 1).any():
        raise ValueError(f"{where}: predicted probabilities outside [0, 1]")
    return p, list(caught)


def evaluate(name, make_model, frame, feature_cols, description="", notes="",
             folds=FOLDS, horizon=HORIZON, log=True, save_predictions=True, verbose=True,
             allow_duplicate_name=False, require_random_state=True,
             min_train_step=None, sample_weight_fn=None):
    """Train and score one configuration on every chronological split.

    Parameters
    ----------
    name : short, unique experiment name, e.g. "lgbm_raw_v1". A name already in
        the experiment log is rejected unless allow_duplicate_name=True.
    make_model : zero-argument callable returning a NEW unfitted estimator with
        fit(X, y), predict_proba(X) and classes_. Put scalers etc. inside a sklearn
        Pipeline so they are fit per split. Every random_state parameter
        (including nested ones) must be set explicitly unless
        require_random_state=False.
    frame : training DataFrame with txId, time_step, label and feature columns.
        Unknown-label rows are dropped here. Features must be built without
        using label values.
    feature_cols : list of predictor columns.

    Returns a dict with per-split metrics, summary metrics, per-step AUCs, the
    validation predictions (txId, time_step, label, split, pred) and warning
    counts. Horizon predictions overlap fold predictions for steps 22-35, so any
    later join must use (split, txId), not txId alone.

    Logging writes one row to experiments/experiment_log.csv and a full record
    (all estimator parameters, split definitions, feature list, environment,
    warnings, per-step AUCs) to experiments/runs/<run_id>.json, only after every
    split has finished successfully.
    """
    forbidden = {"txId", "time_step", "label", "index"} & set(feature_cols)
    if forbidden:
        raise ValueError(f"Non-feature columns in feature_cols: {forbidden}")
    probe = make_model()
    rs = _random_states(probe)
    if require_random_state and any(v is None for v in rs.values()):
        unset = [k for k, v in rs.items() if v is None]
        raise ValueError(f"random_state not set for: {unset}. Set it explicitly in make_model.")
    standard_folds = [s.name for s in folds] == [s.name for s in FOLDS]
    if log and standard_folds:
        _check_log_header()
        if standard_folds and not allow_duplicate_name and _name_already_logged(name):
            raise ValueError(f"Experiment name '{name}' is already in {LOG_PATH.name}; choose a new name "
                             "(scripts take --version, e.g. --version s6exp2)")

    start = time.time()
    data = frame[frame["label"].notna()]
    steps = data["time_step"]
    splits = list(folds) + [horizon]
    fold_rows, per_step, preds, all_warnings = [], [], [], []

    for s in splits:
        tr, va = data[s.train_mask(steps)], data[s.val_mask(steps)]
        if min_train_step is not None:
            tr = tr[tr["time_step"] >= min_train_step]
        _require_both_classes(tr["label"], f"{s.name} train")
        _require_both_classes(va["label"], f"{s.name} validation")
        # Structural guards: validation strictly later than training, no shared transactions.
        if tr["time_step"].max() >= va["time_step"].min():
            raise ValueError(f"{s.name}: training steps overlap or follow validation steps")
        if not set(tr["txId"]).isdisjoint(va["txId"]):
            raise ValueError(f"{s.name}: a transaction appears in both training and validation")

        w = None if sample_weight_fn is None else np.asarray(sample_weight_fn(tr), dtype=float)
        if w is not None and len(w) != len(tr):
            raise ValueError(f"{s.name}: sample_weight_fn returned {len(w)} weights for {len(tr)} rows")
        p, caught = _fit_predict(make_model, tr[feature_cols], tr["label"].astype(int),
                                 va[feature_cols], s.name, sample_weight=w)
        all_warnings += [(s.name, w) for w in caught]
        auc, ap = roc_auc_score(va["label"], p), average_precision_score(va["label"], p)
        fold_rows.append({"split": s.name, "auc": auc, "ap": ap, "n_val": len(va),
                          "n_illicit_val": int(va["label"].sum())})
        preds.append(pd.DataFrame({"txId": va["txId"].values, "time_step": va["time_step"].values,
                                   "label": va["label"].values, "split": s.name, "pred": p}))
        for t, g in pd.DataFrame({"t": va["time_step"].values, "y": va["label"].values, "p": p}).groupby("t"):
            per_step.append({"split": s.name, "time_step": int(t), "n_illicit": int(g["y"].sum()),
                             "auc": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan})

    res = pd.DataFrame(fold_rows).set_index("split")
    fold_names = [s.name for s in folds]
    f = res.loc[fold_names]
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for _, w in all_warnings)
    n_rt = sum(issubclass(w.category, RuntimeWarning) for _, w in all_warnings)
    warn_counts = {"n_convergence_warnings": n_conv, "n_runtime_warnings": n_rt,
                   "n_other_warnings": len(all_warnings) - n_conv - n_rt}
    summary = {
        "auc_mean": f["auc"].mean(), "auc_std": f["auc"].std(ddof=0), "auc_min": f["auc"].min(),
        "ap_mean": f["ap"].mean(), "auc_horizon": res.loc[horizon.name, "auc"],
        "ap_horizon": res.loc[horizon.name, "ap"], "runtime_s": round(time.time() - start, 1),
    }
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%d-%H%M%S}-{name}"
    predictions = pd.concat(preds, ignore_index=True)

    if verbose:
        print(f"\n{run_id}  ({len(feature_cols)} features)")
        print(res.round(4).to_string())
        print(f"folds: AUC mean {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f} "
              f"(min {summary['auc_min']:.4f}), AP mean {summary['ap_mean']:.4f} | "
              f"horizon AUC {summary['auc_horizon']:.4f}")
        if all_warnings:
            print(f"warnings captured: {warn_counts} (details in the run record when logged)")

    if log:
        commit, dirty = _git_state()
        chash = code_hash()
        for d in (EXPERIMENT_DIR, RUN_DIR, FEATURE_SET_DIR):
            d.mkdir(parents=True, exist_ok=True)
        fhash = _feature_hash(feature_cols)
        (FEATURE_SET_DIR / f"{fhash}.json").write_text(json.dumps(list(feature_cols)))
        record = {
            "run_id": run_id, "timestamp_utc": now.isoformat(timespec="seconds"), "name": name,
            "description": description, "notes": notes, "model": repr(probe),
            "params": {k: _jsonable(v) for k, v in _params(probe).items()},
            "random_states": {k: _jsonable(v) for k, v in rs.items()},
            "splits": [asdict(s) for s in splits], "feature_hash": fhash, "n_features": len(feature_cols),
            "min_train_step": min_train_step,
            "sample_weight_fn": None if sample_weight_fn is None else getattr(sample_weight_fn, "__name__", repr(sample_weight_fn)),
            "feature_columns": list(feature_cols), "data_hash": frame_hash(frame, feature_cols),
            "metrics": res.reset_index().to_dict(orient="records"), "summary": summary,
            "per_step": per_step, "warnings": warn_counts,
            "warning_messages": sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for _, w in all_warnings})[:20],
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "sklearn": sklearn.__version__,
                            "platform": platform.platform()},
            "git_commit": commit, "git_dirty": dirty, "code_hash": chash,
        }
        (RUN_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2, default=_jsonable))
        if not standard_folds:
            # Custom splits (e.g. the long-horizon set) get a full JSON record and saved
            # predictions, but no row in experiment_log.csv, whose columns are fixed to
            # the four standard folds.
            if verbose:
                print(f"custom splits: full record written to {RUN_DIR / (run_id + '.json')}, "
                      "no experiment_log.csv row")
            save_predictions = True
        row = {"run_id": run_id, "timestamp_utc": record["timestamp_utc"], "name": name,
               "description": description, "model": " ".join(repr(probe).split()),
               "random_states": json.dumps(record["random_states"]), "n_features": len(feature_cols),
               "feature_hash": fhash, "splits": _splits_text(splits), "git_commit": commit,
               "git_dirty": dirty, "code_hash": chash, "public_lb": "", "notes": notes,
               **{f"auc_{n}": round(res.loc[n, "auc"], 5) for n in fold_names},
               **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in summary.items()},
               **warn_counts}
        new = not LOG_PATH.exists()
        if standard_folds:
            _write_log_row(row, new)

    if save_predictions:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(PRED_DIR / f"{run_id}.csv.gz", index=False)

    return {"run_id": run_id, "folds": res, "summary": summary, "per_step": pd.DataFrame(per_step),
            "predictions": predictions, "warnings": warn_counts,
            "warning_messages": sorted({f"{w.category.__name__}: {str(w.message)[:200]}" for _, w in all_warnings})}


def _write_log_row(row, new):
    with open(LOG_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
