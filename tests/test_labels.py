from pathlib import Path

import polars as pl

from ranking.data.labels import build_ranking_examples

MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"


def test_single_event_sessions_are_dropped():
    events = pl.DataFrame(
        {
            "session": [1, 2, 2],
            "aid": [10, 20, 21],
            "ts": [0, 0, 1],
            "type": [0, 0, 2],
        }
    )
    categories = pl.DataFrame({"aid": [10, 20, 21], "category": [0, 1, 1]})
    examples = build_ranking_examples(events, categories)
    assert examples["session"].to_list() == [2]
    assert examples["label_aid"].to_list() == [21]
    assert examples["label_grade"].to_list() == [3]
    assert examples["query_category"].to_list() == [1]
    assert examples["prefix_len"].to_list() == [1]
    assert examples["prefix_last_ts"].to_list() == [0]
    assert examples["label_ts"].to_list() == [1]


def test_label_grade_matches_event_type():
    events = pl.DataFrame(
        {
            "session": [1, 1, 2, 2, 3, 3],
            "aid": [10, 11, 10, 11, 10, 11],
            "ts": [0, 1, 0, 1, 0, 1],
            "type": [0, 0, 0, 1, 0, 2],
        }
    )
    categories = pl.DataFrame({"aid": [10, 11], "category": [0, 0]})
    examples = build_ranking_examples(events, categories).sort("session")
    assert examples["label_grade"].to_list() == [1, 2, 3]


def test_prefix_never_includes_the_held_out_event():
    events = pl.DataFrame(
        {
            "session": [1, 1, 1],
            "aid": [10, 11, 12],
            "ts": [0, 5, 9],
            "type": [0, 0, 2],
        }
    )
    categories = pl.DataFrame({"aid": [10, 11, 12], "category": [0, 0, 1]})
    examples = build_ranking_examples(events, categories)
    assert examples["prefix_last_ts"].to_list() == [5]
    assert examples["label_aid"].to_list() == [12]


def test_on_mock_data_every_multi_event_session_gets_one_example():
    events = pl.read_parquet(MOCK_DIR / "mock_events.parquet")
    true_categories = pl.read_parquet(MOCK_DIR / "mock_true_categories.parquet").rename(
        {"true_hidden_category": "category"}
    )
    examples = build_ranking_examples(events, true_categories)

    session_lengths = events.group_by("session").agg(pl.len().alias("n"))
    multi_event_sessions = session_lengths.filter(pl.col("n") >= 2)["session"]

    assert set(examples["session"].to_list()) == set(multi_event_sessions.to_list())
    assert examples["label_grade"].is_between(0, 3).all()
    assert examples["query_category"].null_count() == 0
    assert examples["label_ts"].null_count() == 0
