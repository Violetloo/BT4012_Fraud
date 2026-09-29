"""Stage 11: blend submission files that already have leaderboard scores.

Run from the repo root, e.g.:
    python3 stage11/blend_submissions.py --files A.csv B.csv --weights 2 1 --label cmp_blend

Every earlier blend combined models chosen on validation, and validation carried no
information about test ranking here (Spearman 0.00 across nine probes). This blends
files whose test performance is already MEASURED, which is the one way leaderboard
feedback can be used constructively rather than as fishing.

Predictions are combined by average RANK, not average probability: AUC depends only on
ordering and the files are calibrated differently. Inputs are matched on `index`, and
the output passes the same format checks as make_submission.

Use it on files that scored well AND come from different feature sets or libraries;
blending two near-identical files changes nothing.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--files", nargs="+", required=True, help="submission CSVs (paths or names in submissions/)")
p.add_argument("--weights", nargs="*", type=float, default=None)
p.add_argument("--label", default="blend")
p.add_argument("--data-dir", default=str(DATA_DIR))
args = p.parse_args()

sub_dir = ROOT / "submissions"
paths = [Path(f) if Path(f).exists() else sub_dir / Path(f).name for f in args.files]
missing = [str(p_) for p_ in paths if not p_.exists()]
if missing:
    sys.exit(f"Not found: {missing}")
w = np.ones(len(paths)) if not args.weights else np.asarray(args.weights, dtype=float)
if len(w) != len(paths):
    sys.exit(f"{len(w)} weights for {len(paths)} files")
w = w / w.sum()

frames = []
for path in paths:
    df = pd.read_csv(path)
    if list(df.columns) != ["index", "target"]:
        sys.exit(f"{path.name}: columns {list(df.columns)} are not ['index', 'target']")
    frames.append(df.set_index("index")["target"])

base = frames[0].index
for f, path in zip(frames, paths):
    if not f.index.equals(base):
        sys.exit(f"{path.name}: index does not match {paths[0].name}")

ranks = np.zeros(len(base))
print("Blending (weights normalised):")
for weight, f, path in zip(w, frames, paths):
    ranks += weight * (rankdata(f.values) / (len(f) + 1))
    print(f"  {weight:5.3f}  {path.name}")

corr = pd.DataFrame({p_.name[:28]: rankdata(f.values) for p_, f in zip(paths, frames)}).corr(method="spearman")
print("\nRank correlation between the inputs (low = genuinely different models):")
print(corr.round(4).to_string())

sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
out = pd.DataFrame({"index": base.values, "target": ranks})
problems = []
if len(out) != len(sample):
    problems.append(f"row count {len(out)} != sample {len(sample)}")
if not out["index"].equals(sample["index"]):
    problems.append("index does not match sample_submission.csv in value and order")
if not np.isfinite(out["target"]).all() or out["target"].min() < 0 or out["target"].max() > 1:
    problems.append("targets outside [0, 1] or non-finite")
if problems:
    sys.exit("Blend failed validation: " + "; ".join(problems))

now = datetime.now(timezone.utc)
csv_path = sub_dir / f"{now:%Y%m%d-%H%M%S}_blend_{args.label}.csv"
out.to_csv(csv_path, index=False)
csv_path.with_suffix(".json").write_text(json.dumps({
    "file": csv_path.name, "created_utc": now.isoformat(timespec="seconds"), "label": args.label,
    "inputs": [p_.name for p_ in paths], "weights": [float(x) for x in w],
    "method": "weighted average of within-file ranks",
    "mean_input_rank_correlation": float(corr.values[np.triu_indices(len(paths), 1)].mean())
    if len(paths) > 1 else None}, indent=2))
print(f"\nAll checks passed.\nWrote {csv_path}")
print("Note: the inputs' public scores measure 30% of the test set, so a blend chosen on them\n"
      "carries some risk of fitting that sample; prefer blends of genuinely different models.")
