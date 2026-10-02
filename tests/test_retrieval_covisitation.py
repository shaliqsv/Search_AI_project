import polars as pl

from ranking.data.retrieval_covisitation import (
    DAY_MS,
    build_matrix_batched,
    buy2buy_matrix,
    buys_matrix,
    clicks_matrix,
    windowed_covisitation_pairs,
)


def test_uniform_weight_counts_occurrences_per_direction():
    events = pl.DataFrame(
        {
            "session": [1, 1, 2, 2],
            "aid": [10, 20, 10, 20],
            "ts": [0, 1000, 0, 1000],
            "type": [0, 0, 0, 0],
        }
    )
    pairs = windowed_covisitation_pairs(events, within_ms=DAY_MS, weight_mode="uniform")
    fwd = pairs.filter((pl.col("aid") == 10) & (pl.col("neighbor") == 20))
    rev = pairs.filter((pl.col("aid") == 20) & (pl.col("neighbor") == 10))
    assert fwd["weight"][0] == 2.0  # co-occurs in both sessions
    assert rev["weight"][0] == 2.0  # the reverse direction is tracked independently


def test_pairs_outside_time_window_are_excluded():
    events = pl.DataFrame(
        {
            "session": [1, 1],
            "aid": [10, 20],
            "ts": [0, DAY_MS * 2],  # 2 days apart
            "type": [0, 0],
        }
    )
    pairs = windowed_covisitation_pairs(events, within_ms=DAY_MS, weight_mode="uniform")
    assert pairs.height == 0


def test_type_weight_favours_cart_over_order_over_click():
    base = {"session": [1, 1], "aid": [10, 20], "ts": [0, 1]}
    click_target = windowed_covisitation_pairs(
        pl.DataFrame({**base, "type": [1, 0]}), within_ms=DAY_MS, weight_mode="type"
    )
    cart_target = windowed_covisitation_pairs(
        pl.DataFrame({**base, "type": [1, 1]}), within_ms=DAY_MS, weight_mode="type"
    )
    order_target = windowed_covisitation_pairs(
        pl.DataFrame({**base, "type": [1, 2]}), within_ms=DAY_MS, weight_mode="type"
    )
    # weight on the (aid=10 -> neighbor=20) direction, where neighbor's type varies
    w = lambda df: df.filter((pl.col("aid") == 10) & (pl.col("neighbor") == 20))["weight"][0]
    assert w(cart_target) > w(order_target) > w(click_target)


def test_recency_weight_favours_later_source_events():
    events = pl.DataFrame(
        {"session": [1, 1, 2, 2], "aid": [10, 20, 30, 20], "ts": [0, 1, 1000, 1001], "type": [0, 0, 0, 0]}
    )
    pairs = windowed_covisitation_pairs(events, within_ms=DAY_MS, weight_mode="recency")
    w_early = pairs.filter((pl.col("aid") == 10) & (pl.col("neighbor") == 20))["weight"][0]
    w_late = pairs.filter((pl.col("aid") == 30) & (pl.col("neighbor") == 20))["weight"][0]
    assert w_late > w_early


def test_buy2buy_restricts_both_sides_to_cart_and_order():
    # click(10) then cart(20) then order(30), all within window
    events = pl.DataFrame(
        {"session": [1, 1, 1], "aid": [10, 20, 30], "ts": [0, 1, 2], "type": [0, 1, 2]}
    )
    pairs = buy2buy_matrix(events)
    got = set(zip(pairs["aid"].to_list(), pairs["neighbor"].to_list()))
    assert got == {(20, 30), (30, 20)}


def test_buys_matrix_targets_cart_and_order_not_click():
    events = pl.DataFrame(
        {"session": [1, 1, 1], "aid": [10, 20, 30], "ts": [0, 1, 2], "type": [0, 1, 0]}
    )
    pairs = buys_matrix(events)
    # neighbor=30 is a click -> must never appear as a target in the buys matrix
    assert 30 not in pairs["neighbor"].to_list()


def test_clicks_matrix_targets_clicks_only():
    events = pl.DataFrame(
        {"session": [1, 1, 1], "aid": [10, 20, 30], "ts": [0, 1, 2], "type": [1, 0, 1]}
    )
    pairs = clicks_matrix(events)
    # only aid=20 (the click) may appear as a neighbor/target
    assert set(pairs["neighbor"].to_list()) == {20}


def test_long_session_capped_before_cross_join():
    # 50 events in one session - without capping this is a 50x50 cross join;
    # with max_events_per_session=10 it must only consider the most recent 10
    n = 50
    events = pl.DataFrame(
        {
            "session": [1] * n,
            "aid": list(range(n)),
            "ts": list(range(n)),
            "type": [0] * n,
        }
    )
    pairs = windowed_covisitation_pairs(
        events, within_ms=DAY_MS, weight_mode="uniform", max_events_per_session=10
    )
    involved = set(pairs["aid"].to_list()) | set(pairs["neighbor"].to_list())
    assert involved.issubset(set(range(n - 10, n)))


def test_top_k_truncates_per_aid():
    events = pl.DataFrame(
        {
            "session": [1] * 5,
            "aid": [0, 1, 2, 3, 4],
            "ts": [0, 10, 20, 30, 40],
            "type": [0] * 5,
        }
    )
    pairs = windowed_covisitation_pairs(
        events, within_ms=DAY_MS, weight_mode="uniform", top_k=2
    )
    counts = pairs.group_by("aid").agg(pl.len().alias("n"))
    assert counts["n"].max() <= 2


def test_batched_matches_unbatched_exactly():
    # enough sessions to force multiple small batches, with items repeating
    # across sessions so a true global top-k requires merging across batches
    events = pl.DataFrame(
        {
            "session": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "aid": [10, 20, 10, 20, 10, 30, 10, 40, 10, 20],
            "ts": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "type": [0] * 10,
        }
    )
    unbatched = clicks_matrix(events, top_k=2).sort(["aid", "neighbor"])
    batched = build_matrix_batched(
        clicks_matrix, events, top_k=2, batch_size=2
    ).sort(["aid", "neighbor"])
    assert unbatched.equals(batched)
