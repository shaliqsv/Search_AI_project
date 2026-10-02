"""Ranking-eval unit of prediction: (session, synthetic query) -> graded candidate labels.

D5-equivalent, leave-last-out: each session's last event is held out. The category
of that held-out item is the synthetic query (a documented leak used as an intent
proxy - see Phase 0 issue #1 and _doc/outdate/plan.md D5). The session prefix (every
earlier event) is the context a model would see; the held-out event's type gives the
graded label for its item, and every other candidate item gets label 0 in this scheme
(one relevant item per example, since only one event is held out).
"""

from __future__ import annotations

import polars as pl

from ranking.data.event_types import GRADE


def build_ranking_examples(events: pl.DataFrame, item_categories: pl.DataFrame) -> pl.DataFrame:
    """One row per session with >= 2 events: the held-out label and its query category.

    `events` needs columns: session, aid, ts, type. `item_categories` needs: aid, category.
    Returns: session, query_category, label_aid, label_type, label_grade, label_ts,
    prefix_len, prefix_last_ts (the last prefix event's ts - a model must not use
    anything after it).

    `label_ts` (the held-out event's own timestamp) is what a train/val/test split
    must key off, not the session id or its first event's time - OTTO sessions can
    span the entire observation window (see Phase 1 issue #2's correction), so
    grouping by session start silently mixes in much later events.
    """
    ordered = events.sort(["session", "ts"]).with_columns(
        pl.int_range(pl.len()).over("session").alias("_rank"),
        pl.len().over("session").alias("_session_len"),
    )

    held_out = ordered.filter(pl.col("_rank") == pl.col("_session_len") - 1).filter(
        pl.col("_session_len") >= 2
    )

    prefix_last = ordered.filter(pl.col("_rank") == pl.col("_session_len") - 2).select(
        "session", pl.col("ts").alias("prefix_last_ts")
    )

    return (
        held_out.join(item_categories, on="aid", how="left")
        .join(prefix_last, on="session", how="left")
        .select(
            "session",
            "prefix_last_ts",
            query_category=pl.col("category"),
            label_aid=pl.col("aid"),
            label_type=pl.col("type"),
            label_grade=pl.col("type").replace_strict(GRADE, default=0),
            label_ts=pl.col("ts"),
            prefix_len=pl.col("_session_len") - 1,
        )
    )
