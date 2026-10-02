from pathlib import Path

import polars as pl
from sklearn.metrics import adjusted_rand_score

from ranking.data.categories import cluster_categories
from ranking.data.covisitation import covisitation_pairs

MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"


def test_covisitation_pairs_are_symmetric_and_self_free():
    events = pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "aid": [10, 20, 30, 10, 20],
            "ts": [0, 1, 2, 0, 1],
            "type": [0, 0, 2, 0, 1],
        }
    )
    pairs = covisitation_pairs(events, max_gap_events=2)
    assert (pairs["aid_a"] < pairs["aid_b"]).all()
    assert pairs.filter(pl.col("aid_a") == pl.col("aid_b")).is_empty()


def test_clustering_recovers_mock_hidden_categories():
    events = pl.read_parquet(MOCK_DIR / "mock_events.parquet")
    true_categories = pl.read_parquet(MOCK_DIR / "mock_true_categories.parquet")

    covis = covisitation_pairs(events, max_gap_events=5)
    all_aids = true_categories["aid"].to_numpy()
    result = cluster_categories(covis, all_aids, n_categories=20, n_components=16, seed=0)

    joined = result.join(true_categories, on="aid")
    ari = adjusted_rand_score(
        joined["true_hidden_category"].to_list(), joined["category"].to_list()
    )
    # L2-normalizing before k-means (see categories.py) recovers mock structure almost
    # exactly; a regression back to magnitude-based clustering would collapse this badly
    assert ari > 0.9, f"clustering found little real structure in mock data (ARI={ari:.3f})"

    sizes = result["category"].value_counts()["count"]
    assert sizes.max() / sizes.sum() < 0.5, "one cluster dominates - normalization likely broke"
