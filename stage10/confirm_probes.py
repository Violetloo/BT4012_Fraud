"""Stage 10c: is 0.95060 a real effect or the luckiest of nine equivalent models?

Run from the repo root:  python3 stage10/confirm_probes.py [--version s10conf]

Across the nine probes, validation carried no information about test ranking (Spearman
0.00 for the multi-horizon mean, -0.18 for the fold mean), and their whole spread
(0.00772, sd 0.0024) fits inside the public leaderboard's own noise. So a single score
of 0.95060 cannot distinguish "raw_avl_cmp is a better feature set" from "that draw was
lucky".

These four probes separate the two. Three keep the winning FEATURE SET and change the
model; one keeps a related feature set and the winning model:

  lgbm_avl_cmp        lgbm_balanced on raw_avl_cmp
  blend_avl_cmp       blend_hgb75_lgbm25 on raw_avl_cmp
  tuned_avl_cmp       hgb_tuned on raw_avl_cmp
  g2                  hgb_balanced on raw_g2 (degree + availability, no component)

Reading them:
  - If two or three of the first three land above ~0.949, the feature set is doing the
    work and the result is worth building on.
  - If they cluster near 0.946-0.948, the earlier 0.95060 was a draw from the same
    distribution as everything else, and no further tuning will reliably beat it.
  - g2 separates availability from the component features: if it also scores high, the
    availability group is the active ingredient.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config  # noqa: E402
from src.features import FEATURE_SETS, get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import MODELS, get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
CONFIRM = [
    ("lgbm_avl_cmp", "lgbm_balanced", "raw_avl_cmp", "winning features, second library"),
    ("blend_avl_cmp", "blend_hgb75_lgbm25", "raw_avl_cmp", "winning features, best-scoring blend"),
    ("tuned_avl_cmp", "hgb_tuned", "raw_avl_cmp", "winning features, tuned settings"),
    ("g2", "hgb_balanced", "raw_g2", "availability without the component features"),
]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s10conf")
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--validate-only", action="store_true")
args = p.parse_args()

needed = sorted({fs for _, _, fs, _ in CONFIRM})
train, _ = load_prepared(args.data_dir, feature_sets=needed)

plan = []
for short, model, features, why in CONFIRM:
    if args.only and short not in args.only:
        continue
    if model not in MODELS or features not in FEATURE_SETS:
        print(f"skip {short}: {model}/{features} unavailable")
        continue
    cols = get_feature_columns(features, train)
    print(f"\n=== {short}: {model} on {features} ({len(cols)} features) ===\n{why}")
    r = run_config(f"confirm_{short}", get_model_factory(model), train, cols, args.version,
                   description=f"Stage 10c confirmation: {model} on {features}")
    row = {"candidate": short, "model": model, "features": features, "run_id": r["run_id"],
           "auc_fold_mean": r["auc_fold_mean"], "auc_mh_mean": r["auc_mh_mean"], "why": why}
    if not args.validate_only:
        cmd = [sys.executable, "stage3/make_submission.py", "--model", model, "--features", features,
               "--run-id", r["run_id"], "--data-dir", args.data_dir]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("Wrote ")]
        row["submission_file"] = line[0].replace("Wrote ", "") if line else f"FAILED: {out.stderr.strip()[-200:]}"
        print(row["submission_file"])
    plan.append(row)

df = pd.DataFrame(plan)
df.round(5).to_csv(OUT / f"confirm_plan_{args.version}.csv", index=False)
print(f"\n=== Confirmation probes ===")
print(df[["candidate", "model", "features", "auc_fold_mean", "auc_mh_mean"]].round(4).to_string(index=False))
print(f"\nSaved {OUT / f'confirm_plan_{args.version}.csv'}")
print("Upload the three raw_avl_cmp probes first; g2 can follow the next day.")
