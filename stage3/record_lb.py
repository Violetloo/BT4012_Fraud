"""Record a Kaggle public leaderboard score against a logged run.

Run from the repo root, e.g.:
    python3 stage3/record_lb.py --submission submissions/20261001-101500_hgb_raw_notime.csv --score 0.91234

Reads the submission's JSON record to find the validated run, then writes the
score into that run's public_lb column and the file name into its notes.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.reporting import read_log, update_log_row  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--submission", required=True, help="path to the submitted CSV")
p.add_argument("--score", required=True, type=float, help="public leaderboard AUC shown by Kaggle")
args = p.parse_args()

if not 0 <= args.score <= 1:
    sys.exit("Score must be between 0 and 1")
rec_path = Path(args.submission).with_suffix(".json")
if not rec_path.exists():
    sys.exit(f"No record found at {rec_path}")
rec = json.loads(rec_path.read_text())
run_id = rec.get("validated_run_id")
if not run_id:
    sys.exit("This submission was made with --force and has no validated run to attach the score to.")

row = read_log().set_index("run_id").loc[run_id]
existing = "" if str(row["notes"]) == "nan" else str(row["notes"])
note = (existing + "; " if existing else "") + f"submitted {rec['file']} -> public {args.score}"
update_log_row(run_id, public_lb=f"{args.score}", notes=note)
rec["public_lb"] = args.score
rec_path.write_text(json.dumps(rec, indent=2))
print(f"Recorded public LB {args.score} for {run_id} (validation fold mean {row['auc_mean']:.4f}).")
