"""Stage 19 patch: similarity-neighbour label features.

Run from the repo root:  python3 stage19/apply_stage19.py
Creates src/knn_features.py and registers it in src/features.py and src/frames.py.
Safe to run twice.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

KNN_MODULE = '''"""Label features from nearest neighbours in FEATURE space.

Why this is different from everything before it. All the graph work propagates feature
values, because the payment graph cannot carry labels: every edge lies inside one time
step, so a test transaction has no labelled neighbour, ever. Feature space has no such
restriction. A test transaction CAN be close to labelled training transactions, and
"which known transactions does this one resemble, and were they illicit" is information
no previous block could express.

Leakage control. A row's neighbours are drawn only from labelled rows in STRICTLY
EARLIER time steps. Training rows therefore never see their own label, nor labels from
their own step; and the rule matches test conditions exactly, since every labelled row
(steps 1-35) is earlier than every test row (steps 36-49). Rows with no earlier labelled
neighbour get NaN, which the model handles natively.

Features per row, computed over its K nearest earlier labelled rows:
    knn_label_mean        share of those neighbours that were illicit
    knn_label_weighted    the same, weighted by 1 / distance
    knn_n_valid           how many earlier labelled neighbours were found
    knn_dist_mean/min     how close they are at all
    knn_dist_illicit      distance to the nearest earlier illicit row
    knn_dist_licit        distance to the nearest earlier licit row
    knn_dist_ratio        licit distance / illicit distance
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data import REPO_ROOT, raw_feature_columns

CACHE_PATH = REPO_ROOT / "data" / "derived" / "knn_features.pkl"
BUILDER_VERSION = 1
PREFIX = "knn_"
N_COMPONENTS = 32      # PCA dimensions used for the distance metric
N_CANDIDATES = 60      # neighbours retrieved before the time filter
K = 20                 # neighbours kept after it


def _fingerprint(*frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(str(f.shape).encode())
        h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
    return h.hexdigest()[:16]


def build_knn_features(train, test, n_components=N_COMPONENTS, k=K, n_candidates=N_CANDIDATES,
                       seed=42):
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([train[["txId", "time_step", "label"] + cols],
                       test[["txId", "time_step"] + cols].assign(label=np.nan)],
                      ignore_index=True).copy()

    Z = StandardScaler().fit_transform(nodes[cols].to_numpy("float32"))
    Z = PCA(n_components=n_components, random_state=seed).fit_transform(Z).astype("float32")

    labelled = np.where(nodes["label"].notna().to_numpy())[0]
    nn = NearestNeighbors(n_neighbors=min(n_candidates, len(labelled))).fit(Z[labelled])
    dist, ind = nn.kneighbors(Z)

    steps = nodes["time_step"].to_numpy()
    y = nodes["label"].to_numpy()
    nb_step, nb_y = steps[labelled][ind], y[labelled][ind]
    valid = nb_step < steps[:, None]          # strictly earlier time steps only

    order = np.argsort(~valid, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(Z))[:, None]
    v_valid, v_y, v_d = valid[rows, order], nb_y[rows, order], dist[rows, order]
    count = v_valid.sum(1)
    safe = np.maximum(count, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        label_mean = np.where(count > 0, (v_y * v_valid).sum(1) / safe, np.nan)
        w = np.where(v_valid, 1.0 / (v_d + 1e-6), 0.0)
        label_weighted = np.where(count > 0, (w * v_y * v_valid).sum(1) / np.maximum(w.sum(1), 1e-9),
                                  np.nan)
        dist_mean = np.where(count > 0, (v_d * v_valid).sum(1) / safe, np.nan)
        dist_min = np.where(count > 0, np.where(v_valid, v_d, np.inf).min(1), np.nan)
        d_ill = np.where(v_valid & (v_y == 1), v_d, np.inf).min(1)
        d_lic = np.where(v_valid & (v_y == 0), v_d, np.inf).min(1)

    out = pd.DataFrame({
        f"{PREFIX}label_mean": label_mean,
        f"{PREFIX}label_weighted": label_weighted,
        f"{PREFIX}n_valid": count.astype("float32"),
        f"{PREFIX}dist_mean": dist_mean,
        f"{PREFIX}dist_min": np.where(np.isfinite(dist_min), dist_min, np.nan),
        f"{PREFIX}dist_illicit": np.where(np.isfinite(d_ill), d_ill, np.nan),
        f"{PREFIX}dist_licit": np.where(np.isfinite(d_lic), d_lic, np.nan),
    }, index=pd.Index(nodes["txId"].values, name="txId"))
    out[f"{PREFIX}dist_ratio"] = out[f"{PREFIX}dist_licit"] / (out[f"{PREFIX}dist_illicit"] + 1e-6)
    return out.astype("float32")


def build_or_load_knn(train, test, rebuild=False, verbose=True, **kwargs):
    cache_path = Path(CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test), "version": BUILDER_VERSION,
            "params": {"n_components": N_COMPONENTS, "k": K, "n_candidates": N_CANDIDATES}}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding similarity-neighbour features (different inputs or settings).")
    feats = build_knn_features(train, test, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def knn_columns(df):
    return [c for c in df.columns if c.startswith(PREFIX)]
'''

FEATURES_BLOCK = '''# Stage 19: similarity-neighbour label features (src/knn_features.py). These are the
# only features that carry LABEL information to the test rows: the payment graph cannot,
# because every edge lies inside one time step.
NEEDS_KNN = {"knn_cmp", "ppr_knn_cmp", "prop34_knn_cmp", "ppr_multi_knn_cmp"}
FEATURE_SETS["knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                           + knn_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + ppr_columns(df) + knn_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["prop34_knn_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                  + propagated_columns(df, hops=(2, 3, 4)) + knn_columns(df)
                                  + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_knn_cmp"}
NEEDS_PROPAGATED |= {"prop34_knn_cmp"}
'''

FRAMES_BLOCK = '''    if sets & NEEDS_KNN:
        from src import knn_features
        kf = knn_features.build_or_load_knn(train, test, rebuild=rebuild, verbose=verbose)
        train, test = graph_features.attach(train, kf), graph_features.attach(test, kf)
        if verbose:
            print(f"Attached {kf.shape[1]} similarity-neighbour columns")
'''


def patch(path, marker, fn, label):
    p = ROOT / path
    if not p.exists():
        print(f"SKIP {label}: {path} missing")
        return False
    s = p.read_text()
    if marker in s:
        print(f"ok   {label}: already present")
        return True
    new = fn(s)
    if new is None:
        print(f"FAIL {label}: anchor not found")
        return False
    p.write_text(new)
    print(f"done {label}")
    return True


module = ROOT / "src" / "knn_features.py"
if module.exists():
    print("ok   knn_features.py already present")
else:
    module.write_text(KNN_MODULE)
    print("done knn_features.py written")


def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "knn_columns" not in s:
        s = "from src.knn_features import knn_columns\n" + s
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    return s.replace("NEEDS_GRAPH |= {",
                     'NEEDS_GRAPH |= {"knn_cmp", "ppr_knn_cmp", "prop34_knn_cmp", ', 1)


ok = patch("src/features.py", "NEEDS_KNN", _features, "features: knn sets")


def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_KNN, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)


ok &= patch("src/frames.py", "NEEDS_KNN", _frames, "frames: knn loader")
print("\nAll patches applied." if ok else "\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
