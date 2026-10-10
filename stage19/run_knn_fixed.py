"""Stage 19b: similarity-neighbour label features, with the time rule corrected.

Run from the repo root:  python3 stage19/run_knn_fixed.py [--version s19fix]

WHY THIS REPLACES stage19/run_knn.py
------------------------------------
The first version let every row use labels from STRICTLY EARLIER STEPS. For the test
rows that is exactly right: they sit in steps 36-49 and the only labels that exist are
steps 1-35. For VALIDATION rows it is too generous. A validation row in step 20 of fold1
(trained on steps 1-15) was allowed to use labels from steps 16-19 - rows inside the
block being scored. At submission time no such labels exist, so the +0.0366 multi-horizon
gain measured that way is inflated and cannot be trusted.

The fix: a row may only use labels from steps at or before the SPLIT'S TRAINING CUTOFF.
  fold1 (train 1-15)  -> labels from steps 1-15 only
  fold4 (train 1-30)  -> labels from steps 1-30 only
  test  (train 1-35)  -> labels from steps 1-35 only
Each split therefore sees its labels from the same distance as the test rows will, and
the features are rebuilt per split rather than once for everything.

This script builds the features per cutoff, runs the usual chronological splits, and
prints both the corrected numbers and the inflated ones for comparison.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, raw_feature_columns  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
SPLITS = [("fold1", 15, (16, 20)), ("fold2", 20, (21, 25)), ("fold3", 25, (26, 30)),
          ("fold4", 30, (31, 35)), ("h15", 15, (16, 29)), ("h18", 18, (19, 32)),
          ("h21", 21, (22, 35)), ("h24", 24, (25, 35))]
K, N_CANDIDATES, N_COMPONENTS, SEED = 20, 60, 32, 42


def embed(frame, cols, seed=SEED):
    Z = StandardScaler().fit_transform(frame[cols].to_numpy("float32"))
    return PCA(n_components=N_COMPONENTS, random_state=seed).fit_transform(Z).astype("float32")


def knn_features_for_cutoff(Z, steps, labels, cutoff, k=K, n_candidates=N_CANDIDATES):
    """Label features using only labelled rows at or before `cutoff`."""
    pool = np.where((~np.isnan(labels)) & (steps <= cutoff))[0]
    nn = NearestNeighbors(n_neighbors=min(n_candidates, len(pool))).fit(Z[pool])
    dist, ind = nn.kneighbors(Z)
    nb_y = labels[pool][ind]
    nb_step = steps[pool][ind]
    # a row never uses a label from its own step or later (relevant for training rows)
    valid = nb_step < steps[:, None]
    order = np.argsort(~valid, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(Z))[:, None]
    v_valid, v_y, v_d = valid[rows, order], nb_y[rows, order], dist[rows, order]
    count = v_valid.sum(1)
    safe = np.maximum(count, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(v_valid, 1.0 / (v_d + 1e-6), 0.0)
        out = pd.DataFrame({
            "knn_label_mean": np.where(count > 0, (v_y * v_valid).sum(1) / safe, np.nan),
            "knn_label_weighted": np.where(count > 0,
                                           (w * v_y * v_valid).sum(1) / np.maximum(w.sum(1), 1e-9),
                                           np.nan),
            "knn_n_valid": count.astype("float32"),
            "knn_dist_mean": np.where(count > 0, (v_d * v_valid).sum(1) / safe, np.nan),
            "knn_dist_min": np.where(count > 0, np.where(v_valid, v_d, np.inf).min(1), np.nan),
            "knn_dist_illicit": np.where(v_valid & (v_y == 1), v_d, np.inf).min(1),
            "knn_dist_licit": np.where(v_valid & (v_y == 0), v_d, np.inf).min(1),
        })
    out = out.replace([np.inf, -np.inf], np.nan)
    out["knn_dist_ratio"] = out["knn_dist_licit"] / (out["knn_dist_illicit"] + 1e-6)
    return out.astype("float32")


p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s19fix")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--base-set", default="raw_cmp", help="feature set the knn block is added to")
args = p.parse_args()

train, test = load_prepared(args.data_dir, feature_sets=[args.base_set])
base_cols = [c for c in get_feature_columns(args.base_set, train) if not c.startswith("knn_")]
raw = raw_feature_columns(train, exclude_time_candidates=True)

labelled = train[train["label"].notna()].copy()
Z_all = embed(train, raw)
steps = train["time_step"].to_numpy()
labels = train["label"].to_numpy()
print(f"{len(labelled):,} labelled rows; base feature set '{args.base_set}' has {len(base_cols)} columns")

rows = []
for name, cutoff, (va_lo, va_hi) in SPLITS:
    knn = knn_features_for_cutoff(Z_all, steps, labels, cutoff)
    frame = pd.concat([train[["time_step", "label"] + base_cols].reset_index(drop=True),
                       knn.reset_index(drop=True)], axis=1)
    cols = base_cols + list(knn.columns)
    lab = frame[frame["label"].notna()]
    tr = lab[lab["time_step"] <= cutoff]
    va = lab[lab["time_step"].between(va_lo, va_hi)]
    model = get_model_factory(args.model)().fit(tr[cols], tr["label"].astype(int))
    pred = model.predict_proba(va[cols])[:, 1]
    auc = roc_auc_score(va["label"], pred)
    rows.append({"split": name, "cutoff": cutoff, "auc": auc,
                 "ap": average_precision_score(va["label"], pred),
                 "knn_missing_val": float(va[list(knn.columns)].isna().mean().mean()),
                 "n_val": len(va)})
    print(f"{name:6s} train<= {cutoff:2d}, validate {va_lo}-{va_hi}: AUC {auc:.4f}")

res = pd.DataFrame(rows).set_index("split")
res.round(4).to_csv(OUT / f"knn_fixed_{args.version}.csv")
folds = ["fold1", "fold2", "fold3", "fold4"]
horizons = ["h15", "h18", "h21", "h24"]
print("\n" + res.round(4).to_string())
print(f"\ncorrected fold mean      {res.loc[folds, 'auc'].mean():.4f}   (ppr_cmp baseline 0.9865)")
print(f"corrected multi-horizon  {res.loc[horizons, 'auc'].mean():.4f}   (ppr_cmp baseline 0.9364)")
print("\nFor comparison, the version that let validation rows see labels from inside their own\n"
      "block reported 0.9887 and 0.9730. The difference is the size of that leak.")
print(f"\nSaved {OUT / f'knn_fixed_{args.version}.csv'}")
