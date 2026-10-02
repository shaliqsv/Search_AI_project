import polars as pl

from ranking.data.candidates import build_candidate_pool
from ranking.data.covisitation import top_k_neighbors


def _neighbors(pairs_dict, k=50):
    return top_k_neighbors(pl.DataFrame(pairs_dict), k=k)


def test_candidates_ranked_and_capped():
    neighbors = _neighbors({"aid_a": [1, 1, 1], "aid_b": [2, 3, 4], "weight": [5.0, 3.0, 1.0]})
    prefix = pl.DataFrame({"session": [10, 10], "aid": [1, 1]})
    result = build_candidate_pool(prefix, neighbors, top_k=2)
    assert result["aid"].to_list() == [2, 3]
    assert result["rank"].to_list() == [1, 2]


def test_candidates_exclude_prefix_items():
    neighbors = _neighbors({"aid_a": [1, 2], "aid_b": [2, 3], "weight": [5.0, 1.0]})
    prefix = pl.DataFrame({"session": [10, 10], "aid": [1, 2]})
    result = build_candidate_pool(prefix, neighbors, top_k=10)
    # 2 co-occurs with 1, but 2 is itself a prefix item, so only 3 should surface
    assert result["aid"].to_list() == [3]


def test_candidates_aggregate_score_across_prefix_items():
    neighbors = _neighbors({"aid_a": [1, 2], "aid_b": [9, 9], "weight": [3.0, 4.0]})
    prefix = pl.DataFrame({"session": [10, 10], "aid": [1, 2]})
    result = build_candidate_pool(prefix, neighbors, top_k=10)
    assert result["aid"].to_list() == [9]
    assert result["score"].to_list() == [7.0]


def test_candidates_are_per_session():
    neighbors = _neighbors({"aid_a": [1], "aid_b": [2], "weight": [1.0]})
    prefix = pl.DataFrame({"session": [10, 20], "aid": [1, 5]})
    result = build_candidate_pool(prefix, neighbors, top_k=10)
    assert set(result["session"].to_list()) == {10}


def test_top_k_neighbors_is_directed_and_bounded():
    pairs = pl.DataFrame({"aid_a": [1, 1, 1], "aid_b": [2, 3, 4], "weight": [5.0, 3.0, 1.0]})
    neighbors = top_k_neighbors(pairs, k=2)
    from_1 = neighbors.filter(pl.col("aid") == 1).sort("weight", descending=True)
    assert from_1["neighbor"].to_list() == [2, 3]
    # the reverse direction exists too (undirected input -> directed output)
    assert neighbors.filter(pl.col("aid") == 2)["neighbor"].to_list() == [1]


def test_top_k_neighbors_bounds_a_high_degree_item():
    # item 1 co-occurs with 100 other items; only the top 5 by weight should survive
    n = 100
    pairs = pl.DataFrame(
        {"aid_a": [1] * n, "aid_b": list(range(2, n + 2)), "weight": [float(w) for w in range(n)]}
    )
    neighbors = top_k_neighbors(pairs, k=5)
    assert neighbors.filter(pl.col("aid") == 1).height == 5
