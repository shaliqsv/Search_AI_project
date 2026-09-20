"""Item-to-item co-visitation matrix (issue #7).

Import from a notebook in this folder with `from covis import build_covisitation`.
Stays here, not in `src/ranking/`, until issue #52.

Definition
- Two events of one session are a pair if the second is at most `window` events after the first
  (in the same session, after removing rare items), at most `max_gap_ms` later, and they are
  different items.
- weight = (type_weight[first] + type_weight[second]) / 2 * 0.5 ** (gap / halflife)
- The matrix is symmetric before the neighbour cap: pair (a, b) counts for a->b and b->a.
- Only items with at least `min_freq` events (in the data passed in) get a row and column.
- `top_k` keeps only each item's strongest neighbours, so the result is no longer symmetric.
"""

import numpy as np
from scipy import sparse

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
TYPE_WEIGHTS = (1.0, 3.0, 6.0)  # click, cart, order


def _keep_top_k(matrix, top_k):
    """Keep the top_k largest entries of every row."""
    matrix = matrix.tocsr()
    counts = np.diff(matrix.indptr)
    keep = np.ones(matrix.nnz, dtype=bool)
    for row in np.flatnonzero(counts > top_k):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        order = np.argpartition(matrix.data[start:stop], -top_k)[:-top_k]
        keep[start + order] = False
    coo = matrix.tocoo()
    return sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape, dtype=matrix.dtype
    )


def build_covisitation(
    session,
    aid,
    ts,
    etype,
    min_freq=5,
    window=10,
    max_gap_ms=DAY_MS,
    halflife_ms=6 * HOUR_MS,
    top_k=100,
    type_weights=TYPE_WEIGHTS,
):
    """Return (matrix, item_ids). Row/column i of the matrix is the item item_ids[i].

    Inputs are equal-length numpy arrays sorted by session, then time.
    """
    ids, inverse, counts = np.unique(aid, return_inverse=True, return_counts=True)
    keep_item = counts >= min_freq
    keep_event = keep_item[inverse]
    new_index = np.cumsum(keep_item) - 1
    item = new_index[inverse[keep_event]]
    session, ts, etype = session[keep_event], ts[keep_event], etype[keep_event]
    item_ids = ids[keep_item]
    n = len(item_ids)
    w_type = np.asarray(type_weights, dtype=np.float32)

    matrix = sparse.csr_matrix((n, n), dtype=np.float32)
    for k in range(1, window + 1):
        gap = ts[k:] - ts[:-k]
        ok = (session[:-k] == session[k:]) & (gap <= max_gap_ms) & (item[:-k] != item[k:])
        first, second = item[:-k][ok], item[k:][ok]
        weight = (
            0.5 * (w_type[etype[:-k][ok]] + w_type[etype[k:][ok]]) * 0.5 ** (gap[ok] / halflife_ms)
        ).astype(np.float32)
        matrix = matrix + sparse.coo_matrix((weight, (first, second)), shape=(n, n)).tocsr()
    matrix = matrix + matrix.T
    if top_k:
        matrix = _keep_top_k(matrix, top_k)
    return matrix.tocsr(), item_ids
