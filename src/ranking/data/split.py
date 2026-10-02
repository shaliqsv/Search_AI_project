"""Time-based session split (Phase 2: no random or entity-based split - see AGENTS.md).

Sessions are assigned to a week bucket from their first event's timestamp, then
whole weeks are assigned to train/validation/test. Splitting by week (not by a
single global cutoff) keeps every session's events together in one bucket, since
OTTO sessions do not span multiple weeks in practice, but the assignment is
computed from the actual timestamps rather than assumed.
"""

from __future__ import annotations

import polars as pl

MS_PER_WEEK = 7 * 24 * 60 * 60 * 1000


def assign_week(events: pl.DataFrame) -> pl.DataFrame:
    """Add a `week` column (0-indexed) per session, from that session's first ts.

    `events` needs columns: session, ts. Returns one row per session: session, week.
    """
    ts_min = events["ts"].min()
    return (
        events.group_by("session")
        .agg(pl.col("ts").min().alias("session_start_ts"))
        .with_columns(((pl.col("session_start_ts") - ts_min) // MS_PER_WEEK).alias("week"))
        .select("session", "week")
    )


def split_by_week(
    session_weeks: pl.DataFrame, train_weeks: set[int], val_weeks: set[int], test_weeks: set[int]
) -> dict[str, pl.Series]:
    """Partition session ids into train/val/test by their week bucket.

    Raises if any week is assigned to more than one split, or if a session's week
    falls outside all three sets (caller must account for every week explicitly).
    """
    overlap = (train_weeks & val_weeks) | (train_weeks & test_weeks) | (val_weeks & test_weeks)
    if overlap:
        raise ValueError(f"weeks assigned to more than one split: {overlap}")

    known_weeks = train_weeks | val_weeks | test_weeks
    unassigned = session_weeks.filter(~pl.col("week").is_in(known_weeks))
    if not unassigned.is_empty():
        seen = sorted(unassigned["week"].unique().to_list())
        raise ValueError(f"weeks present in data but not assigned to any split: {seen}")

    return {
        "train": session_weeks.filter(pl.col("week").is_in(train_weeks))["session"],
        "val": session_weeks.filter(pl.col("week").is_in(val_weeks))["session"],
        "test": session_weeks.filter(pl.col("week").is_in(test_weeks))["session"],
    }
