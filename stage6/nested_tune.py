"""Stage 6c (corrected): properly nested hyperparameter selection.

Run from the repo root:  python3 stage6/nested_tune.py [--n-random 9] [--version s6nest]

Why this replaces tune_hgb.py: that script selected one parameter set on fixed inner
blocks (steps 11-15, 16-20, 21-25, 26-30), three of which are outer fold validation
blocks, then reported outer scores. The reported scores therefore reused labels that
had influenced the selection.

Here the search happens inside fit(): for every outer split, the model carves the last
two 5-step blocks out of THAT split's training rows, scores every candidate on them,
refits the winner on the whole training portion and predicts the outer block. The outer
score therefore estimates the tuning procedure, not one pre-chosen winner. The baseline
settings are always among the candidates, so the procedure can never do worse on the
inner blocks than leaving settings alone.

The script then runs one final search over all labelled history, whose settings are the
ones to submit; the honest estimate of their quality is the nested score above, not the
final search's own inner score.

Outputs: stage6/nested_selection_<version>.csv (what each split chose),
stage6/results_<version>.csv, stage6/decisions_<version>.csv,
stage6/final_params_<version>.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.tuning import (BASELINE_PARAMS, SELECTION_LOG, NestedTunedClassifier,  # noqa: E402
                        candidate_params, inner_scores)

OUT = Path(__file__).resolve().parent
FEATURES = "raw_cmp"
BASELINE_MODEL = "hgb_balanced"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s6nest")
p.add_argument("--n-random", type=int, default=9, help="random candidates besides the baseline")
p.add_argument("--inner-blocks", type=int, default=2)
p.add_argument("--inner-block-len", type=int, default=5)
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[FEATURES])
cols = get_feature_columns(FEATURES, train)
steps = train.set_index(train.index)["time_step"]
candidates = candidate_params(args.n_random)
print(f"{len(candidates)} candidates (baseline + {args.n_random} random), "
      f"{args.inner_blocks} inner blocks of {args.inner_block_len} steps inside each training window")


def nested():
    return NestedTunedClassifier(step_lookup=steps, candidates=candidates,
                                 n_blocks=args.inner_blocks, block_len=args.inner_block_len)


rows = []
print(f"\n--- {BASELINE_MODEL} (fixed baseline settings) ---")
rows.append(run_config(f"{BASELINE_MODEL}_{FEATURES}", get_model_factory(BASELINE_MODEL), train, cols,
                       args.version, description=f"Nested-tuning baseline: {BASELINE_MODEL} on {FEATURES}"))
print("\n--- nested tuning (search inside each split's training rows) ---")
rows.append(run_config(f"hgb_nested_{FEATURES}", nested, train, cols, args.version,
                       description=f"Nested random search ({len(candidates)} candidates) on {FEATURES}"))

sel = pd.DataFrame(SELECTION_LOG)
sel["params"] = sel["params"].apply(json.dumps)
sel.to_csv(OUT / f"nested_selection_{args.version}.csv", index=False)
print("\n=== What each split's inner search selected ===")
print(sel[["train_steps", "n_train", "mean_inner_auc", "baseline_inner_auc", "params"]]
      .to_string(index=False, max_colwidth=90))
chose_baseline = sel["params"].eq(json.dumps(BASELINE_PARAMS)).sum()
print(f"\nSplits that kept the baseline settings: {chose_baseline} of {len(sel)}")

tab, dec, best = summarise(rows, f"{BASELINE_MODEL}_{FEATURES}", OUT, args.version, title="procedure")

# Final search over all labelled history, for the submission model.
lab = train[train["label"].notna()]
final = []
for params in candidates:
    s = inner_scores(params, lab[cols], lab["label"].astype(int), lab["time_step"],
                     args.inner_blocks, args.inner_block_len)
    final.append({"params": params, "mean_inner_auc": float(pd.Series(s).mean()), "inner_scores": s})
final_tab = pd.DataFrame(final).sort_values("mean_inner_auc", ascending=False)
final_tab.assign(params=final_tab["params"].apply(json.dumps)).round(5).to_csv(
    OUT / f"final_search_{args.version}.csv", index=False)
best_params = final_tab.iloc[0]["params"]
(OUT / f"final_params_{args.version}.json").write_text(json.dumps(best_params, indent=2, default=float))
print("\n=== Final search over all labelled history (for the submission model) ===")
print(final_tab.assign(params=final_tab["params"].apply(json.dumps))
      .head(5).round(5).to_string(index=False, max_colwidth=90))
print(f"\nSelected: {json.dumps(best_params, default=float)}")
print(f"Saved to {OUT / f'final_params_{args.version}.json'}")
print("\nThe honest estimate of these settings' quality is the nested comparison above,\n"
      "not this search's own inner AUC. Submit them only if the nested procedure passed.")
