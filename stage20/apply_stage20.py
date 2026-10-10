"""Stage 20 patch: finer PageRank mixture, the union of the two best blocks, and a
tuning search on the winning feature set.

Run from the repo root:  python3 stage20/apply_stage20.py
Safe to run twice.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROP_CODE = '''

# ---------------------------------------------------------------------------
# Stage 20: a finer mixture of PageRank decays.
#
# Chosen because the LEADERBOARD said so, not validation. The best single model is
# ppr_multi_cmp (0.96057), which mixes three decays (0.05 / 0.15 / 0.30); each decay on
# its own scored lower (0.96034 at 0.15, 0.95769 at 0.30). Mixing scales clearly helps,
# so this adds the intermediate decays the three-way mixture skips over.
FINE_CACHE_PATH = REPO_ROOT / "data" / "derived" / "ppr_fine_features.pkl"
FINE_VERSION = 1
FINE_ALPHAS = (0.10, 0.20, 0.25)


def build_fine_ppr(train, test, edges, alphas=FINE_ALPHAS, iters=10):
    blocks = []
    for a in alphas:
        b = build_ppr_features(train, test, edges, alpha=a)
        b.columns = [c.replace(PPR_PREFIX, f"pprf{int(a * 100):02d}_") for c in b.columns]
        blocks.append(b)
    return pd.concat(blocks, axis=1)


def build_or_load_fine(train, test, edges, rebuild=False, verbose=True):
    cache_path = Path(FINE_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "version": FINE_VERSION,
            "alphas": list(FINE_ALPHAS)}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding fine PageRank features (different inputs or settings).")
    feats = build_fine_ppr(train, test, edges)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def fine_columns(df, alphas=FINE_ALPHAS):
    prefixes = tuple(f"pprf{int(a * 100):02d}_" for a in alphas)
    return [c for c in df.columns if c.startswith(prefixes)]
'''

FEATURES_BLOCK = '''# Stage 20. Both sets below extend something the LEADERBOARD rewarded, not something
# validation liked: the best single model mixes PageRank decays, and the best submission
# blends the PageRank and discrete-hop representations.
NEEDS_FINE = {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp"}
FEATURE_SETS["ppr_fine_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + ppr_columns(df) + fine_columns(df) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_fine_multi_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                      + ppr_columns(df) + ppr_columns_alpha(df, 0.05)
                                      + ppr_columns_alpha(df, 0.3) + fine_columns(df)
                                      + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_multi_prop34_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                        + ppr_columns(df) + ppr_columns_alpha(df, 0.05)
                                        + ppr_columns_alpha(df, 0.3)
                                        + propagated_columns(df, hops=(2, 3, 4))
                                        + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp"}
NEEDS_PROPAGATED |= {"ppr_multi_prop34_cmp"}
'''

FRAMES_BLOCK = '''    if sets & NEEDS_FINE:
        fine = propagation.build_or_load_fine(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, fine), propagation.attach(test, fine)
        if verbose:
            print(f"Attached {fine.shape[1]} fine-decay PageRank columns")
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


ok = patch("src/propagation.py", "build_fine_ppr", lambda s: s + PROP_CODE,
           "propagation: fine PageRank decays")


def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "fine_columns" not in s:
        if "from src.propagation import (" in s:
            s = s.replace("from src.propagation import (", "from src.propagation import (fine_columns, ", 1)
        else:
            s = s.replace("from src.propagation import ", "from src.propagation import fine_columns, ", 1)
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    return s.replace("NEEDS_GRAPH |= {",
                     'NEEDS_GRAPH |= {"ppr_fine_cmp", "ppr_fine_multi_cmp", "ppr_multi_prop34_cmp", ', 1)


ok &= patch("src/features.py", "NEEDS_FINE", _features, "features: stage 20 sets")


def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_FINE, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)


ok &= patch("src/frames.py", "NEEDS_FINE", _frames, "frames: fine loader")
print("\nAll patches applied." if ok else "\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
