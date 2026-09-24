"""Stage 3: investigate the runtime warnings recorded by the logistic-regression runs.

Run from the repo root:  python3 stage3/investigate_lr_warnings.py

Until this passes, treat logistic-regression results as provisional. For every
split (four folds and horizon), with the time-encoding candidates excluded:

A. Arithmetic check. Fit the Stage 3 logreg pipeline (lbfgs). Recompute its
   decision values with np.einsum, which does not call the BLAS matrix routine
   that raises the warnings, and compare with the pipeline's own decision_function.
   Agreement to floating-point precision means the warned-about multiplications
   returned correct numbers for prediction.
B. Optimisation check. Refit on the same scaled data with two different solvers,
   both run to a tight tolerance: lbfgs (the code path that raises the warnings)
   and newton-cholesky. Both should reach the same optimum; close agreement in
   coefficients and validation AUC means the warnings did not steer lbfgs to a
   different solution. (The Stage 3 model uses the default tolerance, so it can
   differ slightly from this optimum for ordinary convergence reasons; that
   difference is reported for information, not judged.)
C. Warning sources. Counts warnings per category, separately for the scaler, the
   lbfgs fit and prediction, so their origin is recorded.

Writes stage3/lr_warning_investigation.csv and exits with status 1 if A or B fails.
Thresholds: A max relative difference < 1e-9; B |delta AUC| < 1e-4 and
max relative coefficient difference < 1e-3.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.models import SEED, logreg  # noqa: E402
from src.validation import FOLDS, HORIZON  # noqa: E402


def counted(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, len(caught), sorted({w.category.__name__ for w in caught})


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
args = p.parse_args()

train, _, _ = load_data(args.data_dir)
cols = get_feature_columns("raw_notime", train)
lab = train[train["label"].notna()]
rows = []
for s in FOLDS + [HORIZON]:
    tr, va = lab[s.train_mask(lab["time_step"])], lab[s.val_mask(lab["time_step"])]
    y_tr, y_va = tr["label"].astype(int).values, va["label"].astype(int).values

    # C. where do warnings come from?
    scaler, n_scale, _ = counted(lambda: StandardScaler().fit(tr[cols]))
    Xtr, n_tx1, _ = counted(lambda: scaler.transform(tr[cols]))
    Xva, n_tx2, _ = counted(lambda: scaler.transform(va[cols]))

    # A. arithmetic: BLAS decision values vs einsum
    pipe, n_fit, cats_fit = counted(lambda: logreg().fit(tr[cols], y_tr))
    dec_blas, n_pred, _ = counted(lambda: pipe.decision_function(va[cols]))
    lr = pipe[-1]
    Xva_pipe = pipe[0].transform(va[cols])
    dec_einsum = np.einsum("ij,j->i", Xva_pipe, lr.coef_[0]) + lr.intercept_[0]
    rel = np.max(np.abs(dec_blas - dec_einsum) / np.maximum(1.0, np.abs(dec_einsum)))
    auc_lbfgs = roc_auc_score(y_va, dec_blas)

    # B. optimisation: two solvers to a tight tolerance on the same scaled data
    tight = dict(tol=1e-10, random_state=SEED)
    lb, n_lbt, _ = counted(lambda: LogisticRegression(solver="lbfgs", max_iter=50000, **tight).fit(Xtr, y_tr))
    nc, n_alt, _ = counted(lambda: LogisticRegression(solver="newton-cholesky", max_iter=1000, **tight).fit(Xtr, y_tr))
    dec_lb, n_dec_lb, _ = counted(lambda: lb.decision_function(Xva))
    dec_nc, n_dec_nc, _ = counted(lambda: nc.decision_function(Xva))
    auc_lb = roc_auc_score(y_va, dec_lb)
    auc_nc = roc_auc_score(y_va, dec_nc)
    coef_rel = float(np.max(np.abs(lb.coef_[0] - nc.coef_[0]) / np.maximum(1.0, np.abs(nc.coef_[0]))))

    rows.append({"split": s.name, "lbfgs_iterations": int(lr.n_iter_[0]),
                 "A_max_rel_diff_blas_vs_einsum": float(rel),
                 "B_auc_lbfgs_tight": auc_lb, "B_auc_newton_tight": auc_nc,
                 "B_delta_auc": auc_nc - auc_lb, "B_max_rel_coef_diff": coef_rel,
                 "info_auc_stage3_model": auc_lbfgs, "info_stage3_minus_optimum": auc_lbfgs - auc_nc,
                 "C_warn_scaler": n_scale + n_tx1 + n_tx2, "C_warn_stage3_fit": n_fit,
                 "C_warn_predict": n_pred, "C_warn_lbfgs_tight_fit": n_lbt + n_dec_lb, "C_warn_newton_fit": n_alt + n_dec_nc,
                 "C_fit_categories": ",".join(cats_fit)})

res = pd.DataFrame(rows).set_index("split")
out = Path(__file__).resolve().parent / "lr_warning_investigation.csv"
res.to_csv(out)
pd.set_option("display.width", 200)
print(res.to_string(float_format=lambda x: f"{x:.3g}"))

ok_a = bool((res["A_max_rel_diff_blas_vs_einsum"] < 1e-9).all())
ok_b = bool((res["B_delta_auc"].abs() < 1e-4).all() and (res["B_max_rel_coef_diff"] < 1e-3).all())
print(f"\nA arithmetic (BLAS vs einsum decision values): {'PASS' if ok_a else 'FAIL'}")
print(f"B optimisation (tight lbfgs vs newton-cholesky): {'PASS' if ok_b else 'FAIL'}")
print(f"Saved {out}")
if not (ok_a and ok_b):
    print("Logistic-regression results stay provisional; see the table above.")
    sys.exit(1)
print("Both checks passed: decision values match an independent computation, and the lbfgs optimum matches a\n"
      "second solver. No evidence that the warnings affected logistic-regression results.")
