"""Synthetic item categories from co-visitation structure (D4-equivalent).

OTTO ships item IDs with no text or category labels, so categories are invented by
clustering items on how they co-occur in real sessions. This is a documented proxy,
not a real taxonomy - callers must label anything downstream as synthetic.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def cluster_categories(
    covisitation: pl.DataFrame,
    all_aids: np.ndarray,
    n_categories: int = 64,
    n_components: int = 32,
    seed: int = 0,
) -> pl.DataFrame:
    """Cluster items into `n_categories` synthetic categories from co-visitation weights.

    Builds a sparse item x item weight matrix restricted to `all_aids` (a dense
    matrix would not fit in memory once the catalog reaches tens of thousands of
    items), reduces with truncated SVD, then k-means on the reduced embedding.

    Both the input matrix (row-wise) and the SVD output are L2-normalized before
    k-means. Raw co-visitation weight scales vary hugely by item popularity; without
    normalizing, Euclidean k-means groups by *magnitude* (how connected an item is
    overall) rather than *direction* (which items it actually co-occurs with), which
    collapses almost every item into one giant "not very connected" cluster (observed:
    99.2% of clustered items in a single cluster before this fix - see Phase 3 issue).

    Returns columns: aid, category (int in [0, n_categories)).
    """
    aid_index = {int(a): i for i, a in enumerate(all_aids)}
    n = len(all_aids)

    rows, cols, weights = [], [], []
    for aid_a, aid_b, weight in covisitation.select(["aid_a", "aid_b", "weight"]).iter_rows():
        if aid_a in aid_index and aid_b in aid_index:
            i, j = aid_index[aid_a], aid_index[aid_b]
            rows += [i, j]
            cols += [j, i]
            weights += [weight, weight]

    matrix = coo_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    matrix = normalize(matrix, norm="l2", axis=1)

    n_components = min(n_components, n - 1) if n > 1 else 1
    embedding = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(matrix)
    embedding = normalize(embedding, norm="l2", axis=1)

    n_categories = min(n_categories, n)
    labels = KMeans(n_clusters=n_categories, random_state=seed, n_init=10).fit_predict(embedding)

    return pl.DataFrame({"aid": all_aids, "category": labels})
