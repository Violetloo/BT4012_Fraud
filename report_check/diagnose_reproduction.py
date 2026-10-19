"""Find where the notebook's predictions diverge from the uploaded submission.

Run from the repo root:  python3 diagnose_reproduction.py

The notebook's file and the scored file differ on 1,125 predictions (max 4.9e-4) and 1,324
ranks. This script localises the cause: it compares each of the five members against the
submission CSV that member was uploaded as, then each intermediate blend. If the members
already differ, the cause is feature construction or fitting; if only the blend differs,
the cause is the combination step.

It refits the five models the way the notebook does, so expect 10-15 minutes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(".").resolve()
SUBS = ROOT / "submissions"
sys.path.insert(0, str(ROOT))


def newest(pattern):
    c = sorted(SUBS.glob(pattern))
    return c[-1] if c else None


MEMBER_FILES = {
    "ppr_cmp": "*_hgb_balanced_ppr_cmp.csv",
    "prop34_cmp": "*_hgb_balanced_prop34_cmp.csv",
    "ppr_multi_cmp": "*_hgb_balanced_ppr_multi_cmp.csv",
    "ppr_fine_multi_cmp": "*_hgb_balanced_ppr_fine_multi_cmp.csv",
    "ppr_multi_cmp_bins127": "*_hgb_bins127_ppr_multi_cmp.csv",
}
FINAL_FILE = "*blend_best_plus_bins127.csv"


def compare(a, b, label):
    m = pd.DataFrame({"a": a, "b": b})
    d = (m.a - m.b).abs()
    ra, rb = rankdata(m.a), rankdata(m.b)
    print(f"{label:26s} max |diff| {d.max():.3e}   >1e-12: {int((d > 1e-12).sum()):>6,}   "
          f"rank changes: {int((ra != rb).sum()):>6,}   spearman {spearmanr(m.a, m.b)[0]:.8f}")
    return int((ra != rb).sum())


# ---- rebuild the notebook's features and models -----------------------------
from src.features import get_feature_columns          # noqa: E402
from src.frames import load_prepared                  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

SETS = {"ppr_cmp": "ppr_cmp", "prop34_cmp": "prop34_cmp", "ppr_multi_cmp": "ppr_multi_cmp",
        "ppr_fine_multi_cmp": "ppr_fine_multi_cmp", "ppr_multi_cmp_bins127": "ppr_multi_cmp"}
BINS = {"ppr_multi_cmp_bins127": 127}

train, test = load_prepared("data", feature_sets=sorted(set(SETS.values())))
lab = train[train["label"].notna()]
print(f"{len(lab):,} labelled training rows, {len(test):,} test rows\n")

preds = {}
for label, fs in SETS.items():
    cols = get_feature_columns(fs, train)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=0.0, max_bins=BINS.get(label, 255), early_stopping=False,
        class_weight="balanced", random_state=42).fit(lab[cols], lab["label"].astype(int))
    preds[label] = model.predict_proba(test[cols])[:, 1]
    print(f"fitted {label:24s} ({len(cols)} features)")

print("\n=== each member against the file it was uploaded as ===")
differing = []
for label, pattern in MEMBER_FILES.items():
    f = newest(pattern)
    if f is None:
        print(f"{label:26s} no submission file matching {pattern}")
        continue
    other = pd.read_csv(f).sort_values("index")["target"].to_numpy()
    if compare(preds[label], other, label):
        differing.append(label)

print("\n=== the blend, stage by stage ===")
r01 = lambda p: rankdata(p) / (len(p) + 1)
b1 = 0.5 * r01(preds["ppr_cmp"]) + 0.5 * r01(preds["prop34_cmp"])
b2 = 2/3 * r01(b1) + 1/3 * r01(preds["ppr_multi_cmp"])
b3 = 0.75 * r01(b2) + 0.25 * r01(preds["ppr_fine_multi_cmp"])
final = 0.75 * r01(b3) + 0.25 * r01(preds["ppr_multi_cmp_bins127"])

for stage, pattern in [("b1 (ppr+prop34)", "*blend_ppr_prop34.csv"),
                       ("b2 (+multi)", "*blend_best_plus_multi.csv"),
                       ("b3 (+fine)", "*blend_best_plus_fine.csv"),
                       ("final (+bins127)", FINAL_FILE)]:
    f = newest(pattern)
    if f is None:
        print(f"{stage:26s} no file matching {pattern}")
        continue
    other = pd.read_csv(f).sort_values("index")["target"].to_numpy()
    cur = {"b1 (ppr+prop34)": b1, "b2 (+multi)": b2, "b3 (+fine)": b3, "final (+bins127)": final}[stage]
    compare(cur, other, stage)

print("\nReading: if the members already differ, the cause is in feature construction or\n"
      "fitting, not in the blend. If the members match and only a blend differs, the cause\n"
      "is the combination step.")
if differing:
    print(f"\nMembers that differ: {', '.join(differing)}")
    print("Next check for those: compare the feature matrix column by column against the\n"
          "run record in experiments/runs/<validated_run_id>.json (ordered feature list).")
