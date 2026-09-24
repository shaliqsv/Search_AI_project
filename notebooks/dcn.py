"""DCN-V2 with a multi-gate mixture of experts (MMoE), in PyTorch (issue #27, reused by #28).

Import from a notebook in this folder with `from dcn import DCNMTL, train_model, ...`.
Stays here, not in `src/ranking/`, until issue #52.

Architecture
  input x0     = standardised numeric features + a category embedding
  cross layers : x_{l+1} = x0 * (W_l x_l + b_l) + x_l          (DCN-V2, full-rank weights)
  mode "mmoe"  : several expert MLPs on the cross output; one softmax gate per task (input x0)
                 mixes the experts; one tower per task gives a logit
  mode "shared": one expert shared by all tasks (the shared-bottom baseline for #28)
The number of tasks is the number of label columns (3 = click, cart, order; 1 = single task).
"""

import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

OTTO_WEIGHTS = (0.10, 0.30, 0.60)   # click, cart, order


class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, x0, x):
        return x0 * self.w(x) + x


def mlp(d_in, d_hidden, d_out=None, last_relu=True):
    layers = [nn.Linear(d_in, d_hidden), nn.ReLU()]
    if d_out is None:
        layers += [nn.Linear(d_hidden, d_hidden)]
        if last_relu:
            layers += [nn.ReLU()]
    else:
        layers += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*layers)


class DCNMTL(nn.Module):
    def __init__(self, n_num, n_cats, n_tasks=3, mode="mmoe", cat_dim=8, n_cross=3, n_experts=4, expert_dim=64, tower_dim=32):
        super().__init__()
        assert mode in ("mmoe", "shared")
        self.mode, self.n_tasks = mode, n_tasks
        self.emb = nn.Embedding(n_cats, cat_dim)
        d = n_num + cat_dim
        self.cross = nn.ModuleList([CrossLayer(d) for _ in range(n_cross)])
        k = n_experts if mode == "mmoe" else 1
        self.experts = nn.ModuleList([mlp(d, expert_dim) for _ in range(k)])
        self.gates = nn.ModuleList([nn.Linear(d, k) for _ in range(n_tasks)]) if mode == "mmoe" else None
        self.towers = nn.ModuleList([mlp(expert_dim, tower_dim, 1) for _ in range(n_tasks)])

    def forward(self, x_num, cat):
        x0 = torch.cat([x_num, self.emb(cat)], dim=-1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        experts = torch.stack([e(x) for e in self.experts], dim=1)       # B, K, H
        outs = []
        for t in range(self.n_tasks):
            if self.mode == "mmoe":
                gate = torch.softmax(self.gates[t](x0), dim=-1).unsqueeze(-1)
                z = (gate * experts).sum(dim=1)
            else:
                z = experts[:, 0]
            outs.append(self.towers[t](z).squeeze(-1))
        return torch.stack(outs, dim=1)                                    # B, T logits


def blend(probs, weights=OTTO_WEIGHTS):
    """Final score: weighted sum of task probabilities (columns click, cart, order)."""
    return probs @ np.asarray(weights, dtype=probs.dtype)


def predict(model, x_num, cat, batch=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x_num), batch):
            out.append(torch.sigmoid(model(x_num[s : s + batch], cat[s : s + batch])).numpy())
    return np.concatenate(out)


def ndcg10_pool(scores, labels, pool=200):
    """Mean NDCG@10 inside the pool (ideal ordering of the pool's own labels). For tuning only."""
    s = scores.reshape(-1, pool)
    y = labels.reshape(-1, pool).astype(np.float64)
    top = np.argsort(-s, axis=1, kind="stable")[:, :10]
    disc = 1.0 / np.log2(np.arange(2, 12))
    dcg = (np.take_along_axis(y, top, axis=1) * disc).sum(axis=1)
    ideal = (-np.sort(-y, axis=1)[:, :10] * disc).sum(axis=1)
    keep = ideal > 0
    return float((dcg[keep] / ideal[keep]).mean())


def bce_loss(logits, y, task_weights, sample_weight=None):
    """Weighted sum over tasks of the mean per-task BCE.

    sample_weight: optional (B,) weight per example (issue #32/#33: inverse propensity weights).
    None means every example has weight 1, and gives exactly the same loss as before.
    """
    per = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if sample_weight is not None:
        per = per * sample_weight.unsqueeze(1)
    per_task = per.mean(dim=0)
    return (per_task * torch.as_tensor(task_weights, dtype=torch.float32)).sum(), per_task


def train_model(model, x_tr, c_tr, y_tr, x_va, c_va, y_va, task_weights, epochs=3, batch=4096, lr=2e-3, seed=0, label="", w_tr=None):
    """Train with a weighted sum of per-task BCE. Keeps the epoch with the lowest validation loss.

    w_tr: optional per-example training weights (N,). Default None = all ones. Validation is never weighted.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], (float("inf"), None, 0)
    start = time.time()

    def loss_fn(logits, y, sw=None):
        return bce_loss(logits, y, task_weights, sw)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        tot, per = [], []
        for s in range(0, len(order) - batch + 1, batch):
            ids = torch.from_numpy(order[s : s + batch])
            loss, pt = loss_fn(model(x_tr[ids], c_tr[ids]), y_tr[ids], None if w_tr is None else w_tr[ids])
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append(loss.item()); per.append(pt.detach().numpy())
        model.eval()
        with torch.no_grad():
            v_loss, v_pt = [], []
            for s in range(0, len(x_va), 65536):
                l, pt = loss_fn(model(x_va[s : s + 65536], c_va[s : s + 65536]), y_va[s : s + 65536])
                v_loss.append(l.item()); v_pt.append(pt.numpy())
        row = {"epoch": epoch + 1, "train": float(np.mean(tot)), "val": float(np.mean(v_loss)),
               "train_tasks": np.mean(per, axis=0).tolist(), "val_tasks": np.mean(v_pt, axis=0).tolist()}
        history.append(row)
        if row["val"] < best[0]:
            best = (row["val"], {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1)
        print(f"{label} epoch {epoch + 1}: train {row['train']:.4f} val {row['val']:.4f}  per-task val " + ", ".join(f"{x:.4f}" for x in row["val_tasks"]))
    model.load_state_dict(best[1])
    print(f"{label} best epoch {best[2]}, trained in {time.time() - start:.0f} s")
    return history, best[2], time.time() - start


def task_auc(probs, y, max_rows=2_000_000, seed=0):
    idx = np.random.default_rng(seed).choice(len(y), size=min(max_rows, len(y)), replace=False)
    return [float(roc_auc_score(y[idx, t], probs[idx, t])) if 0 < y[idx, t].sum() < len(idx) else float("nan") for t in range(y.shape[1])]
