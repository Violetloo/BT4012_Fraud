"""Stage 16: contrast features, and PageRank-style smoothing of predictions.

Run from the repo root:  python3 stage16/run_contrast.py [--version s16v1]

Every representation so far answers "what does this transaction's neighbourhood look
like". None answers "how much does this transaction DIFFER from its neighbourhood",
which is the more natural question for fraud: laundering transactions sit among
ordinary ones. Stage 2 already found that how unusual a row looks separates the classes
on its own, and that signal has never been combined with the graph.

  contrast_cmp       raw + (x - neighbourhood value), both PageRank and 1-hop versions
  ppr_contrast_cmp   the best PageRank block AND both contrast blocks
  ppr_cppr_cmp       the PageRank block and only its own contrast (a smaller version)

Afterwards, run stage8/score_smoothing.py again: the patch adds a "ppr" smoothing mode
that applies the same iterated blend to PREDICTIONS rather than features. Stage 8a only
tested one hop; the feature-side result says more hops with a decay work better.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402

OUT = Path(__file__).resolve().parent
SETS = ["contrast_cmp", "ppr_contrast_cmp", "ppr_cppr_cmp"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s16v1")
p.add_argument("--model", default="hgb_balanced")
p.add_argument("--baseline", default="ppr_cmp")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline] + args.sets,
                         rebuild=args.rebuild_features)
rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- {args.model} on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"{args.model}_{name}", get_model_factory(args.model), train, cols,
                           args.version, description=f"Stage 16: {name}"))

tab, dec, best = summarise(rows, f"{args.model}_{args.baseline}", OUT, args.version,
                           title="feature set")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
