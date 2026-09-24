"""Two-Tower retrieval model in PyTorch (issue #24).

Import from a notebook in this folder with `from twotower import TwoTower, ...`.
Stays here, not in `src/ranking/`, until issue #52.

Query tower  : category distribution -> weighted mix of category embeddings (32-d)
               + session vector: mean of the prefix item embeddings, weighted by event type and recency
               -> MLP -> 64-d, unit length.
Item tower   : item ID embedding (64-d) + category embedding (32-d) + popularity bucket embedding (16-d)
               -> MLP -> 64-d, unit length.
Score        : dot product of the two unit vectors.
Loss         : sampled softmax with in-batch negatives. With `logq=True` the logit of each item is
               reduced by log(p_item), where p_item is how often the item is a positive in the data,
               so popular items are not punished more just because they appear in more batches.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MAX_HIST = 30
TYPE_WEIGHTS = np.array([1.0, 3.0, 6.0], dtype=np.float32)
DECAY = 0.8


class TwoTower(nn.Module):
    def __init__(self, n_items, n_cats, n_pop, dim=64, cat_dim=32, pop_dim=16, hidden=128):
        super().__init__()
        self.n_items = n_items
        self.hist = nn.Embedding(n_items + 1, dim, padding_idx=n_items)   # extra row = padding / unknown item
        self.q_cat = nn.Embedding(n_cats, cat_dim)
        self.q_mlp = nn.Sequential(nn.Linear(dim + cat_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.i_id = nn.Embedding(n_items, dim)
        self.i_cat = nn.Embedding(n_cats, cat_dim)
        self.i_pop = nn.Embedding(n_pop, pop_dim)
        self.i_mlp = nn.Sequential(nn.Linear(dim + cat_dim + pop_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        for emb in (self.hist, self.i_id):
            nn.init.normal_(emb.weight, std=0.1)
        with torch.no_grad():
            self.hist.weight[n_items].zero_()

    def query(self, hist_idx, hist_w, cat_dist):
        """hist_idx (B, L) catalog indices (n_items = padding), hist_w (B, L) weights, cat_dist (B, n_cats)."""
        e = self.hist(hist_idx)
        pooled = (e * hist_w.unsqueeze(-1)).sum(1) / (hist_w.sum(1, keepdim=True) + 1e-6)
        c = cat_dist @ self.q_cat.weight
        return F.normalize(self.q_mlp(torch.cat([pooled, c], dim=-1)), dim=-1)

    def item(self, item_idx, item_cat, item_pop):
        x = torch.cat([self.i_id(item_idx), self.i_cat(item_cat), self.i_pop(item_pop)], dim=-1)
        return F.normalize(self.i_mlp(x), dim=-1)


def sampled_softmax_loss(q, i, item_idx, log_q=None, temperature=0.07):
    """In-batch sampled softmax. q, i: (B, d) unit vectors, row k of i is the positive of row k of q."""
    logits = q @ i.T / temperature
    if log_q is not None:
        logits = logits - log_q[item_idx].unsqueeze(0)          # logQ correction
    same = item_idx.unsqueeze(0) == item_idx.unsqueeze(1)        # the same item twice in a batch is no negative
    eye = torch.eye(len(item_idx), dtype=torch.bool)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(item_idx)))


def encode_history(prefix_list, prefix_types, index, n_items):
    """Catalog indices and weights of the last MAX_HIST prefix events (padding = n_items, weight 0)."""
    idx = np.full(MAX_HIST, n_items, dtype=np.int64)
    w = np.zeros(MAX_HIST, dtype=np.float32)
    n = len(prefix_list)
    start = max(0, n - MAX_HIST)
    for slot, pos in enumerate(range(start, n)):
        j = index.get(int(prefix_list[pos]))
        if j is None:
            continue
        idx[slot] = j
        w[slot] = TYPE_WEIGHTS[prefix_types[pos]] * DECAY ** (n - 1 - pos)
    return idx, w


def encode_queries(prepared, index, n_items, n_cats):
    """Arrays for a list of prepared queries (see evalset.py)."""
    hist = np.stack([encode_history(q["prefix_list"], q["prefix_types"], index, n_items)[0] for q in prepared])
    weights = np.stack([encode_history(q["prefix_list"], q["prefix_types"], index, n_items)[1] for q in prepared])
    cats = np.array([q["category"] for q in prepared], dtype=np.int64)
    onehot = np.zeros((len(prepared), n_cats), dtype=np.float32)
    onehot[np.arange(len(prepared)), cats] = 1.0
    return torch.from_numpy(hist), torch.from_numpy(weights), torch.from_numpy(onehot), cats
