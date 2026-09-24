"""Rank items by co-visitation with the session prefix (issue #18).

Import from a notebook in this folder with `from covis_rank import CovisRanker`.
Stays here, not in `src/ranking/`, until issue #52.

Score of a candidate = sum over the last `last_n` prefix events of
    type_weight[event type] * decay ** (position from the end) * covis[event item, candidate]
Recent events and carts/orders count more. Candidates must be in the query category and not in
the prefix. If that gives fewer than `top` items, `fill` (a popularity list) tops it up.
"""

import numpy as np

TYPE_WEIGHTS = np.array([1.0, 3.0, 6.0])


class CovisRanker:
    def __init__(self, matrix, item_ids, item_cluster, fill_by_cat=None, last_n=30, decay=0.8, top=100, weights=(1.0, 3.0, 6.0)):
        self.tw = np.asarray(weights, dtype=float)
        self.m = matrix.tocsr()
        self.item_ids = item_ids
        self.index = {int(a): i for i, a in enumerate(item_ids)}
        self.cat = np.array([item_cluster.get(int(a), -1) for a in item_ids], dtype=np.int32)
        self.fill_by_cat = fill_by_cat
        self.last_n, self.decay, self.top = last_n, decay, top
        self.counts = {"no_neighbours": 0, "filled": 0, "queries": 0}

    def scores(self, prefix_aids, prefix_types):
        """Dict candidate-index -> score from the last `last_n` prefix events."""
        cols, vals = [], []
        n = len(prefix_aids)
        for pos in range(max(0, n - self.last_n), n):
            i = self.index.get(int(prefix_aids[pos]))
            if i is None:
                continue
            a, b = self.m.indptr[i], self.m.indptr[i + 1]
            if a == b:
                continue
            weight = self.tw[prefix_types[pos]] * self.decay ** (n - 1 - pos)
            cols.append(self.m.indices[a:b])
            vals.append(self.m.data[a:b] * weight)
        if not cols:
            return None
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        uniq, inv = np.unique(cols, return_inverse=True)
        return uniq, np.bincount(inv, weights=vals)

    def rank(self, q, pure=False):
        self.counts["queries"] += 1
        found = self.scores(q["prefix_list"], q["prefix_types"])
        out = []
        if found is None:
            self.counts["no_neighbours"] += 1
        else:
            cand, score = found
            keep = self.cat[cand] == q["category"]
            cand, score = cand[keep], score[keep]
            aids = self.item_ids[cand]
            order = np.lexsort((aids, -score))          # best score first, ties by item id
            for a in aids[order]:
                if int(a) not in q["prefix"]:
                    out.append(int(a))
                    if len(out) == self.top:
                        break
        if not pure and len(out) < self.top and self.fill_by_cat is not None:
            self.counts["filled"] += 1
            seen = set(out)
            for a in self.fill_by_cat[q["category"]]:
                a = int(a)
                if a not in q["prefix"] and a not in seen:
                    out.append(a)
                    if len(out) == self.top:
                        break
        return out
