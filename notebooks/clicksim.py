"""Position-based click simulator with known propensities (issue #31, reused by #32).

Import from a notebook in this folder with `from clicksim import simulate_clicks, propensity`.
Stays here, not in `src/ranking/`, until issue #52.

Model: an item shown at position k is clicked with probability  theta_k * r  where
  theta_k = (1 / k) ** eta   the chance the user looks at position k (the propensity), known exactly
  r                          the item's true relevance, taken from the held-out behaviour
                             (label 0 not found later, 1 click, 2 cart, 3 order)
Logging policy: score = popularity + Gaussian noise; the top K by score are shown, best first.
"""

import numpy as np

K = 20
ETA = 1.0
SIGMA = 1.0
R_BY_LABEL = np.array([0.02, 0.50, 0.70, 0.90])   # true relevance for label 0, 1, 2, 3


def propensity(position, eta=ETA):
    """Examination probability of a 1-based position."""
    return (1.0 / np.asarray(position, dtype=float)) ** eta


def simulate_clicks(popularity, labels, seed, k=K, eta=ETA, sigma=SIGMA, r_by_label=R_BY_LABEL):
    """Simulate one impression list and its clicks for every query.

    popularity, labels: arrays (n_queries, n_candidates). Returns a dict of flat arrays over the
    shown rows: query, candidate (column index), position (1..k), click, propensity, relevance.
    """
    rng = np.random.default_rng(seed)
    n, m = popularity.shape
    score = popularity + rng.normal(0.0, sigma, size=(n, m)) if sigma > 0 else popularity.astype(float)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]          # shown candidates, best first
    q = np.repeat(np.arange(n), k)
    cand = order.reshape(-1)
    pos = np.tile(np.arange(1, k + 1), n)
    rel = r_by_label[labels[q, cand]]
    theta = propensity(pos, eta)
    click = (rng.random(len(q)) < theta * rel).astype(np.int8)
    return {"query": q, "candidate": cand, "position": pos.astype(np.int16), "click": click,
            "propensity": theta.astype(np.float32), "relevance": rel.astype(np.float32)}
