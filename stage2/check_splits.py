"""Stage 2: validate the chronological split definitions and save their class counts.

Run from the repo root:  python3 stage2/check_splits.py [--data-dir data]
Output: stage2/split_summary.csv (labelled class counts per split, unknowns excluded).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DATA_DIR, load_data           # noqa: E402
from src.validation import FOLDS, HORIZON, check_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
args = p.parse_args()

train, _, _ = load_data(args.data_dir)
summary = check_splits(train)
out = Path(__file__).resolve().parent / "split_summary.csv"
summary.to_csv(out, index=False)

print("Split definitions (inclusive time steps):")
for s in FOLDS + [HORIZON]:
    print(f"  {s.name:8s} train {s.train_steps[0]:>2}-{s.train_steps[1]:<2}  validate {s.val_steps[0]:>2}-{s.val_steps[1]}")
print("\nLabelled class counts (unknown rows are excluded from fitting and scoring):")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nAll checks passed (every train and validation part has both classes). Saved {out}")
