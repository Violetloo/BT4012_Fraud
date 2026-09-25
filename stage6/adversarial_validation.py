"""Stage 6a: find drift-prone features and test dropping them.

Run from the repo root:  python3 stage6/adversarial_validation.py [--version s6adv]

A classifier is trained to tell early rows from late rows using the features alone.
Its AUC measures how far the feature distributions move; permutation importance ranks
the features it leans on. The comparison uses steps 1-7 vs 8-15, inside the earliest
training window, so the ranking never sees a period that is in the future of any
validation block. Labels are never used and all rows take part, including unknown ones.

The ranking is written to stage6/drift_drop_lists.json, which defines the feature sets
raw_cmp_drop5 / drop10 / drop20 (the current best set minus the most drift-prone raw
features). Those are then evaluated against raw_cmp with the Stage 5 rule.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, raw_feature_columns  # noqa: E402
from src.drift import EARLY_STEPS, LATE_STEPS, adversarial_ranking  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE = "raw_cmp"
DROP_SIZES = (5, 10, 20)

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s6adv")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--skip-experiments", action="store_true", help="only write the ranking")
args = p.parse_args()

sets = [BASELINE] + [f"raw_cmp_drop{n}" for n in DROP_SIZES]
train, _ = load_prepared(args.data_dir, feature_sets=sets)
raw_cols = raw_feature_columns(train, exclude_time_candidates=True)

print(f"Adversarial validation: steps {EARLY_STEPS[0]}-{EARLY_STEPS[1]} vs "
      f"{LATE_STEPS[0]}-{LATE_STEPS[1]}, {len(raw_cols)} features")
auc, ranking = adversarial_ranking(train, raw_cols)
ranking.round(5).to_csv(OUT / f"drift_features_{args.version}.csv", index=False)
print(f"Early-vs-late AUC: {auc:.4f}  (0.5 = indistinguishable, 1.0 = fully separable)")
print("\nMost drift-prone features:")
print(ranking.head(20).round(4).to_string(index=False))

drop_lists = {str(n): ranking["feature"].head(n).tolist() for n in DROP_SIZES}
(OUT / "drift_drop_lists.json").write_text(json.dumps(drop_lists, indent=2))
print(f"\nWrote {OUT / 'drift_drop_lists.json'} (drop lists of {', '.join(map(str, DROP_SIZES))} features)")
if args.skip_experiments:
    sys.exit(0)

rows = []
for name in sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 6 drift drop: {name}"))
tab, dec, best = summarise(rows, f"{args.model}_{BASELINE}", OUT, args.version, title="feature set")
if best is not None:
    print(f"Submit with:\n  python3 stage3/make_submission.py --model {args.model} "
          f"--features {best['config'].replace(args.model + '_', '')} --run-id {best['run_id']}")
