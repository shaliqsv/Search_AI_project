from pathlib import Path

import polars as pl

from ranking.data.multi_labels import build_multi_label_examples

MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"


def test_single_event_sessions_are_dropped():
    events = pl.DataFrame(
        {"session": [1, 2], "aid": [10, 20], "ts": [0, 0], "type": [0, 0]}
    )
    out = build_multi_label_examples(events, seed=0)
    assert out.height == 0


def test_click_label_is_the_single_next_click_not_a_set():
    # future = cart(11), click(12), click(13) - cut forces prefix={click(10)}
    events = pl.DataFrame(
        {
            "session": [1, 1, 1, 1],
            "aid": [10, 11, 12, 13],
            "ts": [0, 1, 2, 3],
            "type": [0, 1, 0, 0],  # click, cart, click, click
        }
    )
    # with only one possible cut (session len 4, cut in [0, 2]), force cut=0 via seed search
    out = None
    for seed in range(20):
        candidate = build_multi_label_examples(events, seed=seed)
        if candidate.height and candidate["prefix_len"][0] == 1:
            out = candidate
            break
    assert out is not None
    assert out["click_label"][0] == 12  # first future click, not 13
    assert set(out["cart_labels"][0]) == {11}
    assert out["order_labels"][0].to_list() == []


def test_cart_and_order_labels_are_deduplicated_sets():
    events = pl.DataFrame(
        {
            "session": [1] * 6,
            "aid": [10, 20, 20, 30, 30, 30],
            "ts": [0, 1, 2, 3, 4, 5],
            "type": [0, 1, 1, 2, 2, 2],  # click, cart, cart, order, order, order
        }
    )
    out = None
    for seed in range(20):
        candidate = build_multi_label_examples(events, seed=seed)
        if candidate.height and candidate["prefix_len"][0] == 1:
            out = candidate
            break
    assert out is not None
    assert set(out["cart_labels"][0]) == {20}
    assert set(out["order_labels"][0]) == {30}


def test_prefix_and_future_never_overlap_in_time():
    events = pl.DataFrame(
        {
            "session": [1] * 5,
            "aid": [10, 11, 12, 13, 14],
            "ts": [0, 10, 20, 30, 40],
            "type": [0, 0, 0, 0, 0],
        }
    )
    out = build_multi_label_examples(events, seed=0)
    assert out.height == 1
    assert out["prefix_last_ts"][0] < 40  # the last event's ts is never the prefix boundary alone
    assert out["prefix_len"][0] >= 1


def test_no_future_click_gives_null_label():
    events = pl.DataFrame(
        {
            "session": [1, 1],
            "aid": [10, 20],
            "ts": [0, 1],
            "type": [1, 1],  # cart, cart - no clicks at all
        }
    )
    out = build_multi_label_examples(events, seed=0)
    assert out.height == 1
    assert out["click_label"][0] is None
    assert set(out["cart_labels"][0]) == {20}


def test_on_mock_data_runs_without_error():
    events = pl.read_parquet(MOCK_DIR / "mock_events.parquet")
    out = build_multi_label_examples(events, seed=0)
    assert out.height > 0
    assert out["prefix_len"].min() >= 1
