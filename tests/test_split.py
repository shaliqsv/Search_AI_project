from pathlib import Path

import polars as pl
import pytest

from ranking.data.split import MS_PER_WEEK, assign_week, split_by_week

MOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"


def test_assign_week_buckets_by_first_event():
    events = pl.DataFrame(
        {
            "session": [1, 1, 2, 2, 3],
            "ts": [0, 1000, MS_PER_WEEK, MS_PER_WEEK + 500, 2 * MS_PER_WEEK],
        }
    )
    weeks = assign_week(events).sort("session")
    assert weeks["week"].to_list() == [0, 1, 2]


def test_split_by_week_partitions_sessions_exactly():
    session_weeks = pl.DataFrame({"session": [1, 2, 3, 4], "week": [0, 0, 1, 2]})
    splits = split_by_week(session_weeks, train_weeks={0}, val_weeks={1}, test_weeks={2})
    assert set(splits["train"].to_list()) == {1, 2}
    assert set(splits["val"].to_list()) == {3}
    assert set(splits["test"].to_list()) == {4}


def test_split_by_week_rejects_overlapping_assignment():
    session_weeks = pl.DataFrame({"session": [1], "week": [0]})
    with pytest.raises(ValueError, match="more than one split"):
        split_by_week(session_weeks, train_weeks={0}, val_weeks={0}, test_weeks=set())


def test_split_by_week_rejects_unassigned_week():
    session_weeks = pl.DataFrame({"session": [1, 2], "week": [0, 5]})
    with pytest.raises(ValueError, match="not assigned to any split"):
        split_by_week(session_weeks, train_weeks={0}, val_weeks=set(), test_weeks=set())


def test_split_on_mock_data_covers_all_sessions():
    events = pl.read_parquet(MOCK_DIR / "mock_events.parquet")
    session_weeks = assign_week(events)
    weeks_present = set(session_weeks["week"].unique().to_list())
    splits = split_by_week(
        session_weeks, train_weeks=weeks_present, val_weeks=set(), test_weeks=set()
    )
    assert set(splits["train"].to_list()) == set(session_weeks["session"].to_list())
