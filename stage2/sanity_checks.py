"""Stage 2: sanity checks for the evaluation pipeline (not model development).

Run from the repo root:  python3 stage2/sanity_checks.py [--data-dir data] [--n-perm 30]
The script exits with status 1 if any required check fails.

Required checks
1. Random scores     -> AUC close to 0.5 on every split: scoring works.
2. Positive control  -> a "canary" feature equal to the label gives AUC close to 1:
                        the pipeline can learn target information it is given.
3. Guards            -> evaluate() rejects identifier/label columns as predictors, a
                        split whose training steps overlap validation, a model with an
                        unset random_state, and a validation block with one class;
                        label parsing rejects unexpected text.
4. Reproducibility   -> the same configuration run twice gives identical AUCs.
5. Finite outputs and convergence -> logistic regression on real labels, on every
                        split, produces finite probabilities (enforced inside
                        evaluate()) and raises no ConvergenceWarning.

What these establish: rows are separated correctly and outputs are well formed.
They do not certify that any feature is free of leakage; that depends on how
features are built (never from label values) and must be reviewed per feature.

Informational (no pass/fail)
6. Matrix-multiplication warnings -> whether a plain NumPy matmul on small finite
   data emits RuntimeWarnings in this environment, plus the BLAS library in use.
   Warnings raised during model fits are counted by evaluate(), never silenced.
7. Label-permutation observation -> logistic regression on permuted labels,
   repeated n-perm times, with a description of how permuted-model AUC relates to
   row "extremeness" (sum of absolute standardised feature values, scaler fit on
   training rows only). This describes an association only; it is not evidence
   for or against leakage. Saved to stage2/permutation_check.csv.

Only the random-score run is written to the experiment log.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, load_data, parse_label, raw_feature_columns, set_seed  # noqa: E402
from src.validation import FOLDS, HORIZON, Split, evaluate  # noqa: E402

SEED = set_seed(42)
OUT_DIR = Path(__file__).resolve().parent
SPLITS = FOLDS + [HORIZON]


class RandomScores(BaseEstimator, ClassifierMixin):
    """Ignores the features and returns uniform random scores."""
    def __init__(self, random_state=SEED):
        self.random_state = random_state

    def fit(self, X, y):
        self.rng_ = np.random.default_rng(self.random_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.rng_.random(len(X))
        return np.column_stack([1 - p, p])


class ShuffledLabels(BaseEstimator, ClassifierMixin):
    """Fits the wrapped model on randomly permuted training labels."""
    def __init__(self, base=None, random_state=SEED):
        self.base = base
        self.random_state = random_state

    def fit(self, X, y):
        y_perm = np.random.default_rng(self.random_state).permutation(np.asarray(y))
        self.model_ = clone(self.base).fit(X, y_perm)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def logreg(random_state=SEED):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state))


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--n-perm", type=int, default=30, help="number of label permutations (informational)")
args = p.parse_args()

train, _, _ = load_data(args.data_dir)
cols = raw_feature_columns(train, exclude_time_candidates=True)
quiet = dict(log=False, save_predictions=False, verbose=False)
verdicts = []

# 1. Random scores (logged)
r1 = evaluate("sanity_random", lambda: RandomScores(), train, cols,
              description="Random scores; expected AUC about 0.5", allow_duplicate_name=True)
a = r1["folds"]["auc"]
verdicts.append(("random scores", bool(a.between(0.45, 0.55).all()), f"AUCs {a.round(3).tolist()} (expected 0.45-0.55)"))

# 2. Positive control
canary = train.assign(canary=train["label"])
r2 = evaluate("positive_control", lambda: LogisticRegression(max_iter=1000, random_state=SEED),
              canary, ["canary"], **quiet)
a = r2["folds"]["auc"]
verdicts.append(("positive control", bool((a > 0.99).all()), f"AUCs {a.round(3).tolist()} (expected > 0.99)"))

# 3. Guards
bad_split = Split("overlap", (1, 20), (16, 25))
one_class = train.copy()
one_class.loc[one_class["time_step"].between(31, 35) & one_class["label"].notna(), "label"] = 0.0
guards = {
    "forbidden column": raises(lambda: evaluate("g1", logreg, train, cols + ["time_step"], **quiet)),
    "overlapping split": raises(lambda: evaluate("g2", logreg, train, cols, folds=[bad_split], horizon=bad_split, **quiet)),
    "unset random_state": raises(lambda: evaluate("g3", lambda: logreg(random_state=None), train, cols, **quiet)),
    "one-class validation": raises(lambda: evaluate("g4", logreg, one_class, cols, **quiet)),
    "invalid label text": raises(lambda: parse_label("illicit?")),
}
verdicts.append(("guards", all(guards.values()), ", ".join(f"{k}: {'rejected' if v else 'NOT rejected'}" for k, v in guards.items())))

# 4. Reproducibility
ra = evaluate("repro_a", lambda: ShuffledLabels(logreg()), train, cols, **quiet)
rb = evaluate("repro_b", lambda: ShuffledLabels(logreg()), train, cols, **quiet)
same = bool(np.allclose(ra["folds"]["auc"].values, rb["folds"]["auc"].values))
verdicts.append(("reproducibility", same, "identical AUCs on repeated run" if same else "repeated run differs"))

# 5. Finite outputs and convergence on real labels, every split
r5 = evaluate("numeric", logreg, train, cols, **quiet)
w5 = r5["warnings"]
ok5 = w5["n_convergence_warnings"] == 0
verdicts.append(("finite + converged", ok5,
                 f"finite probabilities on all splits (enforced), convergence warnings: {w5['n_convergence_warnings']}, "
                 f"runtime warnings: {w5['n_runtime_warnings']}, other: {w5['n_other_warnings']}"))

# 6. Matmul warning diagnostic (informational)
rng = np.random.default_rng(0)
A_, b_ = rng.standard_normal((2000, 150)), rng.standard_normal(150)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = A_ @ b_
mm_warn = sum(issubclass(w.category, RuntimeWarning) for w in caught)
try:
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
except Exception:
    blas = "unknown"

# 7. Label-permutation observation (informational)
lab = train[train["label"].notna()]
extremeness, ext_auc = {}, {}
for s in SPLITS:
    t, v = lab[s.train_mask(lab["time_step"])], lab[s.val_mask(lab["time_step"])]
    z = np.abs(StandardScaler().fit(t[cols]).transform(v[cols])).sum(axis=1)
    extremeness[s.name] = pd.Series(z, index=v["txId"].values)
    ext_auc[s.name] = roc_auc_score(v["label"], z)
aucs, corrs = [], []
for k in range(args.n_perm):
    r = evaluate(f"perm_{k}", lambda k=k: ShuffledLabels(logreg(), random_state=SEED + k), train, cols, **quiet)
    aucs.append(r["folds"]["auc"])
    pr = r["predictions"]
    corrs.append(pd.Series({s: spearmanr(g["pred"], extremeness[s].loc[g["txId"]].values)[0]
                            for s, g in pr.groupby("split")}))
A = pd.DataFrame(aucs)
C = pd.DataFrame(corrs)[A.columns]
stats = pd.DataFrame({
    "perm_mean_auc": A.mean(), "perm_sd_auc": A.std(), "perm_min_auc": A.min(), "perm_max_auc": A.max(),
    "share_above_0.5": (A > 0.5).mean(),
    "extremeness_auc": pd.Series(ext_auc),
    "mean_corr_pred_extremeness": C.mean(),
    "corr_auc_vs_extremeness_corr": pd.Series({c: np.corrcoef(A[c], C[c])[0, 1] for c in A.columns}),
})
stats.round(4).to_csv(OUT_DIR / "permutation_check.csv", index_label="split")

print("\n=== Required checks ===")
for name, ok, detail in verdicts:
    print(f"{name:20s}: {'PASS' if ok else 'FAIL'}  {detail}")

print("\n=== Informational ===")
print(f"Matmul diagnostic: BLAS = {blas}; RuntimeWarnings from one plain matmul on finite data: {mm_warn}; "
      f"result finite: {bool(np.isfinite(out).all())}")
if w5["n_runtime_warnings"]:
    print("Runtime warnings during real-label fits (first few):")
    for m in [m for m in r5["warning_messages"] if m.startswith("RuntimeWarning")][:5]:
        print(f"  {m}")
print(f"\nLabel-permutation observation ({args.n_perm} permutations per split):")
print(stats.round(3).to_string())
print("\nThis table describes an association between permuted-model AUC and row extremeness.\n"
      "It is an observation about the data, not a test for leakage.")

failed = [n for n, ok, _ in verdicts if not ok]
if failed:
    print(f"\nFAILED required checks: {failed}")
    sys.exit(1)
print("\nAll required checks passed.")
