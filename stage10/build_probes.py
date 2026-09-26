"""Stage 10b: build a fixed list of probe submissions.

Run from the repo root:  python3 stage10/build_probes.py [--version s10probe] [--only NAME ...]

Why probe at all. Local validation has agreed with the leaderboard once in four attempts,
and the differences now in play (0.002-0.005) are smaller than both the validation noise
and the leaderboard's own noise. With 3 submissions a day and weeks remaining, the
leaderboard is itself a measurement instrument, and a FIXED list of structurally
different candidates is the disciplined way to use it: the list is decided in advance,
so it cannot grow reactively around whichever result looks best, which is how public-
leaderboard overfitting happens.

Each candidate is validated first (a logged run with the usual guards), then built into
a submission through the normal make_submission checks. Nothing is uploaded for you.

Reading the results later: treat differences under ~0.008 as unresolved, prefer a
candidate that is good on both the leaderboard and the multi-horizon mean, and keep one
of your two final picks on the best-validated model rather than the best public score.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.ensembles import register_views, union_columns  # noqa: E402
from src.experiments import run_config  # noqa: E402
from src.features import FEATURE_SETS, get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import MODELS, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
VIEW_NAMES = ["raw_cmp", "raw_avl", "raw_all", "raw_notime", "raw_cmp_drop5"]

# (name, model, feature set, extra submission arguments, why it is worth a probe)
CANDIDATES = [
    ("avl", "hgb_balanced", "raw_avl", [], "best multi-horizon feature set in Stage 5 (0.9331), never submitted"),
    ("avl_cmp", "hgb_balanced", "raw_avl_cmp", [], "the two useful graph groups together"),
    ("all", "hgb_balanced", "raw_all", [], "best fold mean in Stage 3; keeps the time-encoding features"),
    ("per_step", "hgb_balanced", "raw_cmp", ["--sample-weight", "per_step_balanced"],
     "best fold mean (0.9860) and best worst-fold (0.9593) of any configuration"),
    ("lgbm", "lgbm_balanced", "raw_cmp", [], "a different library on the best feature set"),
    ("blend_lgbm", "blend_hgb75_lgbm25", "raw_cmp", [], "library blend, near-identical validation"),
    ("fsblend_cmp_avl", "fsblend_cmp_avl", "raw_cmp", [], "blend across feature sets"),
    ("fsblend_cmp_avl_all", "fsblend_cmp_avl_all", "raw_cmp", [], "three-way blend across feature sets"),
    ("drop5", "hgb_balanced", "raw_cmp_drop5", [], "drift-prone features removed (Stage 6 trade-off)"),
]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s10probe")
p.add_argument("--only", nargs="*", default=None, help="subset of candidate names")
p.add_argument("--validate-only", action="store_true", help="evaluate but do not build submissions")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=VIEW_NAMES + ["raw_avl_cmp"])
views = {}
for name in VIEW_NAMES:
    try:
        views[name] = get_feature_columns(name, train)
    except FileNotFoundError as e:
        print(f"view {name} unavailable: {e}")
register_views(views)

plan = []
for short, model, features, extra, why in CANDIDATES:
    if args.only and short not in args.only:
        continue
    if model not in MODELS or features not in FEATURE_SETS:
        print(f"skip {short}: {model}/{features} unavailable")
        continue
    try:
        cols = (union_columns(get_model_factory(model)().views) if model.startswith("fsblend")
                else get_feature_columns(features, train))
    except FileNotFoundError as e:
        print(f"skip {short}: {e}")
        continue
    label = f"probe_{short}"
    print(f"\n=== {label}: {model} on {features} {' '.join(extra)} ===\n{why}")
    kwargs = {}
    if extra[:1] == ["--sample-weight"] and extra[1] == "per_step_balanced":
        def per_step_balanced(tr):
            counts = tr["time_step"].map(tr["time_step"].value_counts())
            w = 1.0 / counts.values
            return w / w.mean()
        per_step_balanced.__name__ = "per_step_balanced"
        kwargs["sample_weight_fn"] = per_step_balanced
    r = run_config(label, get_model_factory(model), train, cols, args.version,
                   description=f"Stage 10 probe: {model} on {features} {' '.join(extra)}", **kwargs)
    row = {"candidate": short, "model": model, "features": features, "extra": " ".join(extra),
           "run_id": r["run_id"], "auc_fold_mean": r["auc_fold_mean"], "auc_mh_mean": r["auc_mh_mean"],
           "why": why}
    if not args.validate_only:
        cmd = [sys.executable, "stage3/make_submission.py", "--model", model, "--features", features,
               "--run-id", r["run_id"], "--data-dir", args.data_dir] + extra
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("Wrote ")]
        row["submission_file"] = line[0].replace("Wrote ", "") if line else f"FAILED: {out.stderr.strip()[-200:]}"
        print(row["submission_file"])
    plan.append(row)

df = pd.DataFrame(plan).sort_values("auc_mh_mean", ascending=False)
df.round(5).to_csv(OUT / f"probe_plan_{args.version}.csv", index=False)
print(f"\n=== Probe plan ({len(df)} candidates), best validated first ===")
print(df[["candidate", "model", "features", "auc_fold_mean", "auc_mh_mean"]].round(4).to_string(index=False))
print(f"\nSaved {OUT / f'probe_plan_{args.version}.csv'}")
print("Upload three per day, record each with stage3/record_lb.py, and do not add candidates\n"
      "to the list once probing has started.")
