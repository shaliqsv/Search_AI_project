"""Item-to-item co-visitation counts from session event logs.

Must be built from the training window only - see D4 (synthetic categories):
computing this on future weeks would leak information into the category
labels every downstream phase relies on.
"""

from __future__ import annotations

import polars as pl

from ranking.data.event_types import GRADE

TYPE_WEIGHT = {code: float(grade) for code, grade in GRADE.items()}


def covisitation_pairs(events: pl.DataFrame, max_gap_events: int = 5) -> pl.DataFrame:
    """Weighted, undirected item-item co-visitation counts within sessions.

    `events` needs columns: session, aid, ts, type. Events are ordered by ts within
    each session, and every item is paired with up to `max_gap_events` neighbours
    that follow it in the same session (a sliding window, not the full cross-product,
    so long sessions don't produce O(n^2) pairs).

    Returns columns: aid_a, aid_b, weight (aid_a < aid_b, weights summed both directions).
    """
    ordered = events.sort(["session", "ts"]).with_columns(
        pl.col("type").replace_strict(TYPE_WEIGHT, default=1.0).alias("weight")
    )
    pairs = []
    for gap in range(1, max_gap_events + 1):
        shifted = ordered.with_columns(
            pl.col("aid").shift(-gap).over("session").alias("aid_next"),
            pl.col("session").shift(-gap).over("session").alias("session_next"),
            pl.col("weight").shift(-gap).over("session").alias("weight_next"),
        )
        pair = shifted.filter(
            pl.col("session_next").is_not_null() & (pl.col("aid") != pl.col("aid_next"))
        ).select(
            aid_a=pl.min_horizontal("aid", "aid_next"),
            aid_b=pl.max_horizontal("aid", "aid_next"),
            weight=(pl.col("weight") + pl.col("weight_next")) / 2,
        )
        pairs.append(pair)

    return (
        pl.concat(pairs)
        .group_by(["aid_a", "aid_b"])
        .agg(pl.col("weight").sum())
        .sort("weight", descending=True)
    )


def top_k_neighbors(pairs: pl.DataFrame, k: int = 50) -> pl.DataFrame:
    """Directed top-K neighbours per item, from undirected `covisitation_pairs` output.

    Must run before joining a co-visitation table against sessions/prefixes. A
    popular item can have thousands of co-visitation partners; joining every prefix
    item's *full* neighbour list against millions of sessions before aggregating is a
    join fan-out that exhausts memory (observed: OOM-killed on the real sample,
    Phase 4 issue). Pruning to each item's top-K first bounds that fan-out to K.

    Returns columns: aid, neighbor, weight (directed - both (a,b) and (b,a) appear,
    each independently ranked and truncated to the requesting item's own top-K).
    """
    directed = pl.concat(
        [
            pairs.select(aid=pl.col("aid_a"), neighbor=pl.col("aid_b"), weight=pl.col("weight")),
            pairs.select(aid=pl.col("aid_b"), neighbor=pl.col("aid_a"), weight=pl.col("weight")),
        ]
    )
    return (
        # sort by neighbor first so ties in weight break on neighbor id, not on
        # whatever row order the join/batch happened to produce - otherwise which
        # items land in the top-K (and thus which end up in a candidate pool) is
        # nondeterministic across runs, violating the project's reproducibility rule
        directed.sort(["aid", "neighbor"])
        .with_columns(
            pl.col("weight").rank(method="ordinal", descending=True).over("aid").alias("rank")
        )
        .filter(pl.col("rank") <= k)
        .drop("rank")
    )
