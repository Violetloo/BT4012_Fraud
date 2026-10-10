"""Stage 18: neighbourhood dispersion, and a two-stage propagated-score model.

Run from the repo root:  python3 stage18/run_stage18.py [--version s18v1]

Two mechanisms nothing in the project has used.

A. DISPERSION. Every propagated block so far is a weighted mean: it says where the
   neighbourhood sits, never how spread out it is, nor how far the transaction lies from
   it in units of that spread. A mixing service sits among wildly varied counterparties;
   a transaction several neighbourhood standard deviations from its neighbours is odd in
   a way no mean can express.
       nvar_ = P(x^2) - (P x)^2        nz_ = (x - P x) / sqrt(var)

B. TWO-STAGE PROPAGATED SCORES. Stage 8a smoothed final predictions with one hop and
   gained +0.0006 with a tiny weight; any larger weight hurt, because a fixed blend
   cannot vary by transaction. Here the first model's score, its propagated value and
   the difference between them become FEATURES of a second model, so the model learns
   when the neighbourhood is worth trusting.
   Leakage control: training rows get out-of-fold scores from models fitted on strictly
   earlier steps; the rows being predicted get scores from a model never fitted on them;
   and because every edge sits inside one time step, propagation cannot carry a score
   across the split.

Feature sets: disp_cmp, ppr_disp_cmp, ppr_z_cmp.  Models: two_stage_ppr (alpha 0.15),
two_stage_ppr_local (alpha 0.5, less weight on distant neighbours).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data  # noqa: E402
from src.experiments import run_config, summarise  # noqa: E402
from src.features import get_feature_columns  # noqa: E402
from src.frames import load_prepared  # noqa: E402
from src.models import get_model_factory  # noqa: E402
from src.two_stage import register_context  # noqa: E402

OUT = Path(__file__).resolve().parent
SETS = ["disp_cmp", "ppr_disp_cmp", "ppr_z_cmp"]
TWO_STAGE_MODELS = ["two_stage_ppr", "two_stage_ppr_local"]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=str(DATA_DIR))
p.add_argument("--version", default="s18v1")
p.add_argument("--baseline", default="ppr_cmp")
p.add_argument("--sets", nargs="*", default=SETS)
p.add_argument("--two-stage", nargs="*", default=TWO_STAGE_MODELS)
p.add_argument("--rebuild-features", action="store_true")
args = p.parse_args()

train, _ = load_prepared(args.data_dir, feature_sets=[args.baseline] + args.sets,
                         rebuild=args.rebuild_features)
_, _, edges = load_data(args.data_dir)
register_context(train, edges)

rows = []
for name in [args.baseline] + args.sets:
    cols = get_feature_columns(name, train)
    print(f"\n--- hgb_balanced on {name} ({len(cols)} features) ---")
    rows.append(run_config(f"hgb_balanced_{name}", get_model_factory("hgb_balanced"), train, cols,
                           args.version, description=f"Stage 18 features: {name}"))

base_cols = get_feature_columns(args.baseline, train)
for model in args.two_stage:
    print(f"\n--- {model} on {args.baseline} ---")
    rows.append(run_config(f"{model}_{args.baseline}", get_model_factory(model), train, base_cols,
                           args.version, description=f"Stage 18 two-stage: {model}"))

tab, dec, best = summarise(rows, f"hgb_balanced_{args.baseline}", OUT, args.version,
                           title="configuration")
print("\nRanked by fold mean:")
print(tab.sort_values("auc_fold_mean", ascending=False)[["n_features", "auc_fold_mean",
                                                         "auc_fold_min", "auc_mh_mean"]]
      .round(4).to_string())
