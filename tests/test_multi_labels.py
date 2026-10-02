from pathlib import Path

import polars as pl

from ranking.data.multi_labels import build_multi_label_examples, build_weekly_multi_label_examples

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


def test_weekly_cut_point_falls_within_the_window():
    # session has events in "week A" (ts 0-9) and "week B" (ts 10-19)
    events = pl.DataFrame(
        {
            "session": [1] * 6,
            "aid": [10, 11, 12, 13, 14, 15],
            "ts": [0, 5, 10, 12, 15, 18],
            "type": [0, 0, 0, 0, 0, 0],
        }
    )
    out = build_weekly_multi_label_examples(events, week_start_ms=10, week_end_ms=20, seed=0)
    assert out.height == 1
    # prefix_last_ts must be one of the in-window candidate events (10, 12, 15 - not 18, the last)
    assert out["prefix_last_ts"][0] in (10, 12, 15)
    # prefix can include earlier, out-of-window history
    assert out["prefix_len"][0] >= 3  # at least the two week-A events plus one in-window event


def test_weekly_excludes_sessions_with_no_in_window_events():
    events = pl.DataFrame(
        {"session": [1, 1], "aid": [10, 11], "ts": [0, 5], "type": [0, 0]}
    )
    out = build_weekly_multi_label_examples(events, week_start_ms=100, week_end_ms=200, seed=0)
    assert out.height == 0


def test_weekly_excludes_sessions_whose_only_in_window_event_is_the_last():
    # the only in-window event (ts=15) is also the session's absolute last event -
    # nothing left to form a label from, so this session must be excluded
    events = pl.DataFrame(
        {"session": [1, 1], "aid": [10, 11], "ts": [0, 15], "type": [0, 0]}
    )
    out = build_weekly_multi_label_examples(events, week_start_ms=10, week_end_ms=20, seed=0)
    assert out.height == 0


def test_weekly_two_disjoint_windows_can_both_use_the_same_session():
    # a long session with candidate cuts available in both week A and week B
    events = pl.DataFrame(
        {
            "session": [1] * 4,
            "aid": [10, 11, 12, 13],
            "ts": [0, 5, 10, 15],
            "type": [0, 0, 0, 0],
        }
    )
    week_a = build_weekly_multi_label_examples(events, week_start_ms=0, week_end_ms=10, seed=0)
    week_b = build_weekly_multi_label_examples(events, week_start_ms=10, week_end_ms=20, seed=0)
    assert week_a.height == 1
    assert week_b.height == 1
    assert week_a["prefix_last_ts"][0] in (0, 5)  # two valid in-window candidates (ranks 0,1)
    assert week_b["prefix_last_ts"][0] == 10  # only one valid candidate in week B (rank 2; rank 3 is last)


def test_weekly_is_deterministic_with_same_seed():
    events = pl.read_parquet(MOCK_DIR / "mock_events.parquet")
    ts_min, ts_max = events["ts"].min(), events["ts"].max()
    mid = (ts_min + ts_max) // 2
    a = build_weekly_multi_label_examples(events, week_start_ms=mid, week_end_ms=ts_max + 1, seed=0)
    b = build_weekly_multi_label_examples(events, week_start_ms=mid, week_end_ms=ts_max + 1, seed=0)
    assert a.sort("session").equals(b.sort("session"))
