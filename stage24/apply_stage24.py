"""Stage 24 patch: make the PageRank iteration count a real, cached parameter.

Run from the repo root:  python3 stage24/apply_stage24.py

Fixes two things in src/propagation.py and adds the feature sets needed to test them:

1. build_fine_ppr(..., iters=10) accepted an iteration count and never passed it to
   build_ppr_features(), so varying it silently did nothing.
2. The cache metadata did not record the iteration count, so an experiment could
   reuse features built with a different one.

Then it registers blocks built with 30 and 100 iterations at alpha 0.15 and 0.05, so
the number of iterations can be compared while everything else is held fixed. With
alpha = 0.05 the ten-iteration result still carries 0.95^10 = 0.599 weight on the
ten-hop term, so ten iterations is nowhere near the converged representation.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROP_CODE = '''

# ---------------------------------------------------------------------------
# Stage 24: iteration count as a parameter in its own right.
#
# Every PageRank block so far used ten iterations. That is not the converged
# representation: the residual weight on the ten-hop term is (1 - alpha)^10, which is
# 0.197 at alpha = 0.15 but 0.599 at alpha = 0.05. Both the number of iterations and
# the decay shape the result, and only the decay was ever varied.
ITERS_CACHE_PATH = REPO_ROOT / "data" / "derived" / "iters_features.pkl"
ITERS_VERSION = 1
# (alpha, iterations) -> column prefix
ITER_SPECS = {(0.15, 30): "i15x30_", (0.15, 100): "i15x100_",
              (0.05, 30): "i05x30_", (0.05, 100): "i05x100_"}


def build_iteration_features(train, test, edges, specs=ITER_SPECS):
    blocks = []
    for (alpha, iters), prefix in specs.items():
        b = build_ppr_features(train, test, edges, alpha=alpha, iters=iters)
        b.columns = [c.replace(PPR_PREFIX, prefix) for c in b.columns]
        blocks.append(b)
    return pd.concat(blocks, axis=1)


def build_or_load_iters(train, test, edges, rebuild=False, verbose=True):
    cache_path = Path(ITERS_CACHE_PATH)
    meta = {"inputs": _fingerprint(train, test, edges), "version": ITERS_VERSION,
            "specs": sorted(f"{a}:{i}" for (a, i) in ITER_SPECS)}
    if cache_path.exists() and not rebuild:
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get("meta") == meta:
            return cached["features"]
        if verbose:
            print("Rebuilding iteration-count features (different inputs or settings).")
    feats = build_iteration_features(train, test, edges)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"features": feats, "meta": meta}, cache_path)
    return feats


def iteration_columns(df, alpha, iters):
    return [c for c in df.columns if c.startswith(ITER_SPECS[(alpha, iters)])]
'''

FEATURES_BLOCK = '''# Stage 24: the number of propagation iterations, held against the current blocks.
NEEDS_ITERS = {"ppr_i30_cmp", "ppr_i100_cmp", "ppr05_i100_cmp", "ppr_mixiters_cmp"}
FEATURE_SETS["ppr_i30_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                               + iteration_columns(df, 0.15, 30) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_i100_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                + iteration_columns(df, 0.15, 100) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr05_i100_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                  + iteration_columns(df, 0.05, 100) + group_columns(df, ["comp"]))
FEATURE_SETS["ppr_mixiters_cmp"] = (lambda df: raw_feature_columns(df, exclude_time_candidates=True)
                                    + ppr_columns(df) + iteration_columns(df, 0.15, 100)
                                    + group_columns(df, ["comp"]))
NEEDS_DIRECTED |= {"ppr_mixiters_cmp"}
'''

FRAMES_BLOCK = '''    if sets & NEEDS_ITERS:
        it = propagation.build_or_load_iters(train, test, edges, rebuild=rebuild, verbose=verbose)
        train, test = propagation.attach(train, it), propagation.attach(test, it)
        if verbose:
            print(f"Attached {it.shape[1]} iteration-count columns")
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


# ── bug fix 1: build_fine_ppr ignored its iters argument ──────────────────────
p = ROOT / "src" / "propagation.py"
s = p.read_text()
if "build_ppr_features(train, test, edges, alpha=a)" in s:
    s = s.replace("build_ppr_features(train, test, edges, alpha=a)",
                  "build_ppr_features(train, test, edges, alpha=a, iters=iters)")
    p.write_text(s)
    print("done propagation: build_fine_ppr now passes iters through")
else:
    print("ok   propagation: iters pass-through already correct or absent")

# ── bug fix 2: cache metadata must record the iteration count ─────────────────
s = p.read_text()
if '"alphas": list(FINE_ALPHAS)' in s and '"iters"' not in s.split('"alphas": list(FINE_ALPHAS)')[1][:120]:
    s = s.replace('"alphas": list(FINE_ALPHAS)}', '"alphas": list(FINE_ALPHAS), "iters": iters}')
    s = s.replace("def build_or_load_fine(train, test, edges, rebuild=False, verbose=True):",
                  "def build_or_load_fine(train, test, edges, rebuild=False, verbose=True, iters=10):")
    s = s.replace("feats = build_fine_ppr(train, test, edges)",
                  "feats = build_fine_ppr(train, test, edges, iters=iters)")
    p.write_text(s)
    print("done propagation: fine cache metadata now records iters")
else:
    print("ok   propagation: fine cache metadata already records iters (or block absent)")

ok = patch("src/propagation.py", "build_iteration_features", lambda s: s + PROP_CODE,
           "propagation: iteration-count builders")


def _features(s):
    anchor = 'FEATURE_SETS["ppr_cmp"] ='
    if anchor not in s:
        return None
    if "iteration_columns" not in s:
        if "from src.propagation import (" in s:
            s = s.replace("from src.propagation import (", "from src.propagation import (iteration_columns, ", 1)
        else:
            s = s.replace("from src.propagation import ", "from src.propagation import iteration_columns, ", 1)
    s = s.replace(anchor, FEATURES_BLOCK + anchor, 1)
    return s.replace("NEEDS_GRAPH |= {",
                     'NEEDS_GRAPH |= {"ppr_i30_cmp", "ppr_i100_cmp", "ppr05_i100_cmp", "ppr_mixiters_cmp", ', 1)


ok &= patch("src/features.py", "NEEDS_ITERS", _features, "features: iteration sets")


def _frames(s):
    anchor = "    if sets & NEEDS_MATCHED_RANKS:"
    if anchor not in s:
        return None
    s = s.replace("NEEDS_RANKS", "NEEDS_ITERS, NEEDS_RANKS", 1)
    return s.replace(anchor, FRAMES_BLOCK + anchor, 1)


ok &= patch("src/frames.py", "NEEDS_ITERS", _frames, "frames: iteration loader")
print("\nAll patches applied." if ok else "\nSome patches failed - see above.")
sys.exit(0 if ok else 1)
