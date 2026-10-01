"""Stage 13b: a GraphSAGE-style graph neural network (PyTorch, no PyTorch Geometric).

Run from the repo root:
    pip install torch
    python3 stage13/gnn_graphsage.py                 # validate on the usual splits
    python3 stage13/gnn_graphsage.py --submit        # also write a submission file

Why a GNN at all. Stage 12's stacking showed the existing models are near-duplicates of
each other, so the 0.013 gap to the leader is unlikely to come from better combination.
A GNN LEARNS how to mix a transaction's own features with its neighbourhood, over two
hops, with the mixing weights trained against the labels. Stage 5's hand-built
aggregates and Stage 13a's fixed propagation both use mixing chosen in advance; this
does not.

Architecture: two SAGE layers (each node's own features and the mean of its usable
neighbours' features, through separate linear maps), ReLU, dropout, then a linear head.
Full-batch, since the graph fits in memory.

Honest caveats, stated up front:
  - PyTorch is not installed in the environment where this script was written, so it has
    been checked for correctness by reading, not by running. Expect to fix something.
  - The epoch count is chosen on an INNER chronological block (the last 5 steps of each
    split's training window), never on the outer validation block.
  - Coverage is matched as elsewhere: only labelled-train and test rows contribute
    features to a neighbourhood, so a training node's neighbourhood is populated like a
    test node's.
  - Scores are AUCs on the same splits as every other stage, so they are directly
    comparable with the hgb_balanced baseline (fold mean 0.9826, multi-horizon 0.9362).
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR, load_data, raw_feature_columns  # noqa: E402
from src.validation import FOLDS, MULTI_HORIZONS  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch is required: pip install torch")

OUT = Path(__file__).resolve().parent
SEED = 42


def build_graph(train, test, edges):
    """Node features, row-normalised adjacency, labels and time steps for every node."""
    cols = raw_feature_columns(train, exclude_time_candidates=True)
    nodes = pd.concat([
        train[["txId", "time_step", "label"] + cols],
        test[["txId", "time_step"] + cols].assign(label=np.nan)], ignore_index=True)
    nodes["usable"] = nodes["label"].notna().to_numpy() | np.r_[
        np.zeros(len(train), bool), np.ones(len(test), bool)]
    pos = pd.Series(np.arange(len(nodes)), index=nodes["txId"].values)

    e = edges.copy()
    e.columns = ["src", "dst"]
    e = e[e["src"].isin(pos.index) & e["dst"].isin(pos.index)]
    si, di = pos.loc[e["src"]].to_numpy(), pos.loc[e["dst"]].to_numpy()
    usable = nodes["usable"].to_numpy()
    keep = usable[si] & usable[di]
    si, di = si[keep], di[keep]
    r, c = np.r_[si, di], np.r_[di, si]
    A = sparse.coo_matrix((np.ones(len(r), "float32"), (r, c)), shape=(len(nodes), len(nodes))).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    P = (sparse.diags(inv.astype("float32")) @ A).tocoo()
    P_t = torch.sparse_coo_tensor(np.vstack([P.row, P.col]), P.data.astype("float32"),
                                  (len(nodes), len(nodes))).coalesce()
    X = nodes[cols].to_numpy("float32")
    return nodes, X, P_t


class SAGE(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.3):
        super().__init__()
        self.l1_self, self.l1_nbr = nn.Linear(n_in, hidden), nn.Linear(n_in, hidden)
        self.l2_self, self.l2_nbr = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act, self.drop = nn.ReLU(), nn.Dropout(dropout)

    def forward(self, x, P):
        h = self.act(self.l1_self(x) + self.l1_nbr(torch.sparse.mm(P, x)))
        h = self.drop(h)
        h = self.act(self.l2_self(h) + self.l2_nbr(torch.sparse.mm(P, h)))
        h = self.drop(h)
        return self.head(h).squeeze(-1)


def standardise(X, fit_rows):
    mu = X[fit_rows].mean(0)
    sd = X[fit_rows].std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def train_model(Xs, P, y, train_idx, watch_idx, max_epochs, lr, hidden, dropout, weight_decay,
                seed=SEED, verbose=False):
    """Train with the epoch count chosen on watch_idx (an inner block, never the outer one)."""
    torch.manual_seed(seed)
    model = SAGE(Xs.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)],
                              dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.tensor(Xs)
    yt = torch.tensor(y, dtype=torch.float32)
    tr = torch.tensor(train_idx)
    best = (-1.0, 0, None)
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, P)
        loss_fn(out[tr], yt[tr]).backward()
        opt.step()
        if watch_idx is not None and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x, P)).numpy()
            auc = roc_auc_score(y[watch_idx], p[watch_idx])
            if auc > best[0]:
                best = (auc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if verbose:
                print(f"    epoch {epoch:3d} inner AUC {auc:.4f}")
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, P)).numpy()
    return p, best[0], best[1]


p_ = argparse.ArgumentParser()
p_.add_argument("--data-dir", default=str(DATA_DIR))
p_.add_argument("--epochs", type=int, default=200)
p_.add_argument("--lr", type=float, default=0.005)
p_.add_argument("--hidden", type=int, default=128)
p_.add_argument("--dropout", type=float, default=0.3)
p_.add_argument("--weight-decay", type=float, default=1e-4)
p_.add_argument("--inner-block", type=int, default=5, help="steps held out inside the training window")
p_.add_argument("--submit", action="store_true", help="also train on all labelled rows and write a submission")
args = p_.parse_args()

train, test, edges = load_data(args.data_dir)
nodes, X, P = build_graph(train, test, edges)
steps = nodes["time_step"].to_numpy()
y = nodes["label"].to_numpy()
labelled = ~np.isnan(y)
y_filled = np.nan_to_num(y)

rows = []
for split in list(FOLDS) + list(MULTI_HORIZONS):
    tr_lo, tr_hi = split.train_steps
    va_lo, va_hi = split.val_steps
    inner_lo = tr_hi - args.inner_block + 1
    train_idx = np.where(labelled & (steps >= tr_lo) & (steps <= inner_lo - 1))[0]
    watch_idx = np.where(labelled & (steps >= inner_lo) & (steps <= tr_hi))[0]
    val_idx = np.where(labelled & (steps >= va_lo) & (steps <= va_hi))[0]
    Xs = standardise(X, np.where((steps >= tr_lo) & (steps <= tr_hi))[0])
    p, inner_auc, best_epoch = train_model(Xs, P, y_filled, train_idx, watch_idx, args.epochs,
                                           args.lr, args.hidden, args.dropout, args.weight_decay)
    auc = roc_auc_score(y_filled[val_idx], p[val_idx])
    rows.append({"split": split.name, "auc": auc, "inner_auc": inner_auc, "epochs": best_epoch,
                 "n_val": len(val_idx)})
    print(f"{split.name:8s} AUC {auc:.4f}  (inner {inner_auc:.4f} at epoch {best_epoch})")

res = pd.DataFrame(rows).set_index("split")
res.round(5).to_csv(OUT / "gnn_results.csv")
fold_names = [s.name for s in FOLDS]
mh_names = [s.name for s in MULTI_HORIZONS]
print("\n=== GraphSAGE ===")
print(res.round(4).to_string())
print(f"fold mean {res.loc[fold_names, 'auc'].mean():.4f} (baseline 0.9826) | "
      f"multi-horizon mean {res.loc[mh_names, 'auc'].mean():.4f} (baseline 0.9362)")

if args.submit:
    inner_lo = 35 - args.inner_block + 1
    train_idx = np.where(labelled & (steps <= inner_lo - 1))[0]
    watch_idx = np.where(labelled & (steps >= inner_lo) & (steps <= 35))[0]
    Xs = standardise(X, np.where(steps <= 35)[0])
    p, inner_auc, best_epoch = train_model(Xs, P, y_filled, train_idx, watch_idx, args.epochs,
                                           args.lr, args.hidden, args.dropout, args.weight_decay)
    test_pos = np.where(steps >= 36)[0]
    pred = p[test_pos]
    sample = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")
    sub = pd.DataFrame({"index": test["index"].to_numpy(), "target": pred})
    problems = []
    if len(sub) != len(sample) or not sub["index"].equals(sample["index"]):
        problems.append("index does not match sample_submission.csv")
    if not np.isfinite(sub["target"]).all() or sub["target"].min() < 0 or sub["target"].max() > 1:
        problems.append("targets outside [0, 1] or non-finite")
    if problems:
        sys.exit("Submission failed validation: " + "; ".join(problems))
    path = OUT.parent / "submissions" / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_gnn_sage.csv"
    path.parent.mkdir(exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"\nTrained on steps 1-{inner_lo - 1}, epochs chosen on {inner_lo}-35 "
          f"(inner AUC {inner_auc:.4f} at epoch {best_epoch})")
    print(f"Wrote {path}")
