"""Step 1 of the modelling guide: the Level 0 split rule (decision D3)."""

import polars as pl
import pytest

from ranking.data.split import DAY_MS, LEVEL0, START_MS, arrival_slice, level0_slice


def _events():
    rows = []
    for s, day in enumerate([0, 5, 9, 9, 10, 11, 11, 12, 13, 13], start=1):     # one session per start day, two events each
        t0 = START_MS + int(day * DAY_MS + 3_600_000)
        rows += [{"session": s, "aid": 1, "ts": t0, "type": 0}, {"session": s, "aid": 2, "ts": t0 + 2 * DAY_MS if s in (3, 6, 9) else t0 + 1000, "type": 0}]
    return pl.DataFrame(rows)


def test_slices_are_session_disjoint_and_inside_weeks_1_and_2():
    ev = _events()
    val = set(level0_slice(ev, "validation").collect()["session"].to_list())
    test = set(level0_slice(ev, "test").collect()["session"].to_list())
    assert val == {5, 6, 7} and test == {8, 9, 10} and not (val & test)
    assert level0_slice(ev, "train").collect()["ts"].max() < START_MS + 10 * DAY_MS
    assert LEVEL0["test"]["end_day"] <= 14


def test_a_session_crossing_a_boundary_is_cut_there():
    ev = _events()
    train = level0_slice(ev, "train").collect()
    assert train.filter(pl.col("session") == 3).height == 1          # its second event is 2 days later, after the training end
    val = level0_slice(ev, "validation").collect()
    assert val.filter(pl.col("session") == 6).height == 1            # cut at the validation end (day 12)


def test_the_same_rule_applies_to_a_level_1_arrival():
    _, evaluation = arrival_slice(_events(), train_end_day=12)
    assert set(evaluation.collect()["session"].to_list()) == {8, 9, 10}


def test_week_3_is_never_read():
    with pytest.raises(ValueError):
        LEVEL0["oops"] = {"start_day": 14, "end_day": 16}
        level0_slice(_events(), "oops")
    LEVEL0.pop("oops", None)
