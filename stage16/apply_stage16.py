"""Apply the Stage 16 changes to an existing repo, without overwriting your edits.

Run from the repo root:  python3 stage16/apply_stage16.py

It appends new code to src/propagation.py, inserts feature sets into src/features.py,
adds the loader hook to src/frames.py and a new smoothing mode to src/smoothing.py.
Every change is checked first, so running it twice is safe.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROP_CODE = '''

# ---------------------------------------------------------------------------
# Stage 16: contrast features - how a transaction DIFFERS from its neighbourhood.
#
# Every propagated block so far describes what the neighbourhood looks like; none says
# whether the transaction stands out from it. For fraud that difference is the natural
# signal: a laundering transaction often sits among ordinary ones. Stage 2 found that
# "how unusual a row looks" already separates the classes on its own.
#
#   contrast = x - (neighbourhood value of x)
#
# computed against the PageRank block (the best representation so far) and the 1-hop mean.
CONTRAST_CACHE_PATH = REPO_ROOT / "data" / "derived" / "contrast_features.pkl"
CONTRAST_VERSION = 1
CONTRAST_PPR_PREFIX = "cppr_"
CONTRAST_HOP1_PREFIX = "c1_"


def build_contrast_features(train, test, edges, alpha=0.15, iters=10):
    ids, cols, X, si, di, n = _usable_matrix(train, test, edges)
    P, deg = _row_normalised(np.r_[si, di], np.r_[di, si], n)
    H = X.copy()
    for _ in range(iters):
        H = (1 - alpha) * (P @ H) + alpha * X
    ppr_contrast = pd.DataFrame(X - H, index=pd.Index(ids, name="txId"),
                                columns=[f"{CONTRAST_PPR_PREFIX}{c}" for c in cols])
    hop1_contrast = pd.DataFrame(X - (P @ X), index=pd.Index(ids, name="txId"),
                                 columns=[f"{CONTRAST_HOP1_PREFIX}{c}" for c in cols])
    for frame in (ppr_contrast, hop1_contrast):
        frame[deg == 0] = np.nan
    return pd.concat([ppr_contrast, hop1_contrast], axis=1).astype("float32")


def build_or_load_contrast(train, test, edges, rebuild=False, alpha=0.15, verbose=True):
    cache_path = Path(CONTRAST_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "version": CONTRAST_VERSION, "alpha": alpha}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding contrast features (different inputs or settings).")
    feats = build_contrast_features(train, test, edges, alpha=alpha)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def contrast_columns(df, kinds=("ppr", "hop1")):
    prefixes = tuple({"ppr": CONTRAST_PPR_PREFIX, "hop1": CONTRAST_HOP1_PREFIX}[k] for k in kinds)
    return [c for c in df.columns if c.startswith(prefixes)]
'''

FEATURES_BLOCK = '''# Stage 16: contrast features (x minus its neighbourhood value).
NEEDS_CONTRAST = {"contrast_cmp", "ppr_contrast_cmp", "ppr_cppr_cmp"}
FEATURE_SETS["contrast_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + contrast_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_contrast_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                    + ppr_columns(df) + contrast_columns(df)
                                    + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_cppr_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + contrast_columns(df, kinds=("ppr",))
                                + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_contrast_cmp", "ppr_cppr_cmp"}   # these also need the PageRank block
'''

FRAMES_BLOCK = '''    if sets & NEEDS_CONTRAST:
        c = propagation.build_or_load_contrast(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, c), propagation.attach(test, c)
        if verbose:
            print(f"Attached {c.shape[1]} contrast columns")
'''

SMOOTH_BLOCK = '''    if mode == "ppr":
        # Iterated blend with the neighbourhood mean: the prediction-space version of the
        # PageRank propagation that produced the best feature set. alpha is the weight kept
        # on the node's own score at every iteration.
        p0 = s.copy()
        for _ in range(iters):
            joined = pairs.assign(score=pairs["other"].map(s))
            nbr = pd.Series(s.index.map(joined.groupby("node")["score"].mean()), index=s.index)
            s = alpha * p0 + (1 - alpha) * nbr.fillna(s)
        return s.values
'''


def patch(path, check, apply_fn, label):
    p = ROOT / path
    if not p.exists():
        print(f"SKIP {label}: {path} not found")
        return False
    s = p.read_text()
    if check in s:
        print(f"ok   {label}: already present")
        return True
    new = apply_fn(s)
    if new is None:
        print(f"FAIL {label}: anchor not found in {path}")
        return False
    p.write_text(new)
    print(f"done {label}")
    return True


ok = True
ok &= patch("src/propagation.py", "build_contrast_features", lambda s: s + PROP_CODE,
            "propagation: contrast builders")

def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "contrast_columns" not in s.split("\\n")[10]:
        s = s.replace("from src.propagation import (", "from src.propagation import (contrast_columns, ", 1) \
            if "from src.propagation import (" in s else \
            s.replace("from src.propagation import ", "from src.propagation import contrast_columns, ", 1)
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    s = s.replace("NEEDS_GRAPH |= {", 'NEEDS_GRAPH |= {"contrast_cmp", "ppr_contrast_cmp", "ppr_cppr_cmp", ', 1)
    return s

ok &= patch("src/features.py", "NEEDS_CONTRAST", _features, "features: contrast sets")

def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_CONTRAST, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)

ok &= patch("src/frames.py", "NEEDS_CONTRAST", _frames, "frames: contrast loader")

def _smooth(s):
    anchor = '    if mode == "comp_mean":'
    if anchor not in s:
        return None
    s = s.replace('MODES = ("nbr_mean", "nbr_max", "comp_mean")',
                  'MODES = ("nbr_mean", "nbr_max", "comp_mean", "ppr")')
    return s.replace(anchor, SMOOTH_BLOCK + anchor, 1)

ok &= patch("src/smoothing.py", 'if mode == "ppr"', _smooth, "smoothing: ppr mode")

print("\\nAll patches applied." if ok else "\\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
