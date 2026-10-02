"""Time-windowed, type-specific co-visitation matrices for retrieval (D4).

Distinct from covisitation.py's event-count-windowed pairs (kept for synthetic
category clustering, where an undirected similarity graph is the right structure).
These matrices are directed by design (aid -> neighbor, weight = how good a
candidate neighbor is given aid was seen) - type- and recency-weighting are
inherently asymmetric, so collapsing to an undirected pair and re-expanding later
would double-count every co-occurrence. Output format matches `candidates.py`'s
`neighbors` input directly: aid, neighbor, weight.

Design follows two real reference solutions to the OTTO Kaggle competition: a
public notebook scoring LB 0.575 (cdeotte/candidate-rerank-model), and the actual
1st-place solution (mrkmakr, private LB 0.605) - both use time windows, not
event-count windows, and restrict source/target event types per matrix rather than
one generic matrix.

Session history is capped to the most recent `max_events_per_session` events before
pairing (same fix as the Phase 4 OOM incident: an uncapped cross-join within very
long sessions - some span the entire 4-week window, see Phase 1/3 - blows up memory).
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from ranking.data.event_types import CART, CLICK, ORDER

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
DEFAULT_BATCH_SIZE = 100_000

# empirically-tuned co-visitation target-type weights (cdeotte's public notebook,
# LB 0.575) - deliberately not the same as event_types.GRADE, which is a general
# ordinal relevance weighting, not a weight tuned for this specific retrieval task
BUY_TYPE_WEIGHT = {CLICK: 1.0, CART: 6.0, ORDER: 3.0}


def _cap_recent_events(events: pl.DataFrame, max_events_per_session: int) -> pl.DataFrame:
    return (
        events.with_columns(
            pl.col("ts")
            .rank(method="ordinal", descending=True)
            .over("session")
            .alias("_recency")
        )
        .filter(pl.col("_recency") <= max_events_per_session)
        .drop("_recency")
    )


def windowed_covisitation_pairs(
    events: pl.DataFrame,
    within_ms: int,
    source_types: set[int] | None = None,
    target_types: set[int] | None = None,
    weight_mode: str = "uniform",
    max_events_per_session: int = 30,
    top_k: int | None = None,
) -> pl.DataFrame:
    """Directed item->candidate pairs from events within `within_ms` of each other.

    `weight_mode`: "uniform" (weight 1 per occurrence), "type" (weight by the
    *target* event's type, `BUY_TYPE_WEIGHT`), or "recency" (weight grows linearly
    with how recent the *source* event is within the data's own time range - matches
    cdeotte's clicks matrix, giving more weight to fresher co-occurrences).

    Directed: (aid=10, neighbor=20) and (aid=20, neighbor=10) are both produced and
    counted independently - each reflects "neighbor seen near aid", which is not
    symmetric once type/recency weighting is involved.

    If `top_k` is given, truncates to each aid's top-K neighbours by weight here
    (deterministic tie-break on neighbor id) - skips materializing the full pair
    table when only the pruned version is needed.

    Returns: aid, neighbor, weight.
    """
    capped = _cap_recent_events(events.sort(["session", "ts"]), max_events_per_session)

    left = capped.filter(pl.col("type").is_in(source_types)) if source_types else capped
    right = capped.filter(pl.col("type").is_in(target_types)) if target_types else capped

    pairs = (
        left.select("session", aid="aid", ts_x="ts")
        .join(right.select("session", neighbor="aid", ts_y="ts", type_y="type"), on="session")
        .filter(
            (pl.col("aid") != pl.col("neighbor"))
            & ((pl.col("ts_x") - pl.col("ts_y")).abs() < within_ms)
        )
    )

    if weight_mode == "uniform":
        pairs = pairs.with_columns(pl.lit(1.0).alias("weight"))
    elif weight_mode == "type":
        pairs = pairs.with_columns(
            pl.col("type_y").replace_strict(BUY_TYPE_WEIGHT, default=1.0).alias("weight")
        )
    elif weight_mode == "recency":
        ts_min = events["ts"].min()
        ts_max = events["ts"].max()
        span = max(ts_max - ts_min, 1)
        pairs = pairs.with_columns((1 + 3 * (pl.col("ts_x") - ts_min) / span).alias("weight"))
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode!r}")

    agg = (
        pairs.group_by(["aid", "neighbor"])
        .agg(pl.col("weight").sum())
        .sort(["aid", "neighbor"])
    )
    if top_k is not None:
        agg = (
            agg.with_columns(
                pl.col("weight").rank(method="ordinal", descending=True).over("aid").alias("rank")
            )
            .filter(pl.col("rank") <= top_k)
            .drop("rank")
        )
    return agg.sort("weight", descending=True)


def clicks_matrix(
    events: pl.DataFrame, max_events_per_session: int = 30, top_k: int | None = None
) -> pl.DataFrame:
    """Any event -> click, time-weighted toward recency, 24h window. Targets click_label."""
    return windowed_covisitation_pairs(
        events,
        within_ms=DAY_MS,
        target_types={CLICK},
        weight_mode="recency",
        max_events_per_session=max_events_per_session,
        top_k=top_k,
    )


def buys_matrix(
    events: pl.DataFrame, max_events_per_session: int = 30, top_k: int | None = None
) -> pl.DataFrame:
    """Any event -> cart/order, type-weighted (cart > order > click), 24h window.

    Targets cart_labels/order_labels. Weight favours cart over order - cdeotte's
    tuned choice, validated to improve his local CV score.
    """
    return windowed_covisitation_pairs(
        events,
        within_ms=DAY_MS,
        target_types={CART, ORDER},
        weight_mode="type",
        max_events_per_session=max_events_per_session,
        top_k=top_k,
    )


def buy2buy_matrix(
    events: pl.DataFrame, max_events_per_session: int = 30, top_k: int | None = None
) -> pl.DataFrame:
    """Cart/order -> cart/order only, 14-day window - "people who buy X also buy Y".

    Deliberately excludes clicks from both sides and uses a much longer window than
    the other two matrices: buy-to-buy patterns are rarer and more persistent than
    click patterns, so a short window would starve this matrix of signal.
    """
    return windowed_covisitation_pairs(
        events,
        within_ms=14 * DAY_MS,
        source_types={CART, ORDER},
        target_types={CART, ORDER},
        weight_mode="uniform",
        max_events_per_session=max_events_per_session,
        top_k=top_k,
    )


def build_matrix_batched(
    matrix_fn: Callable[..., pl.DataFrame],
    events: pl.DataFrame,
    top_k: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **matrix_fn_kwargs,
) -> pl.DataFrame:
    """Run one of the matrix builders above in session batches, then merge correctly.

    `clicks_matrix` in particular is a near-worst-case self-join at full scale: clicks
    are ~90% of all events, so almost nothing gets filtered out before the cross-join,
    unlike `buys_matrix`/`buy2buy_matrix` which filter down to the rare ~10%
    cart/order events first. Observed: the unbatched version took 6+ minutes and had
    not finished on the real ~1M-session training window; batching by session (each
    batch computed independently, since co-visitation only depends on within-session
    co-occurrence) brought the same computation down to 14 seconds total.

    Each batch applies `top_k` locally to bound its own size, then this function
    re-aggregates every batch's weights by (aid, neighbor) and re-applies `top_k`
    globally - a per-batch top-K alone would be wrong, since the true top-K neighbours
    of a popular item are spread across many different sessions/batches.
    """
    sessions = events["session"].unique().to_list()
    batches = []
    for i in range(0, len(sessions), batch_size):
        batch_sessions = sessions[i : i + batch_size]
        batch_events = events.filter(pl.col("session").is_in(batch_sessions))
        batches.append(matrix_fn(batch_events, top_k=top_k, **matrix_fn_kwargs))

    combined = pl.concat(batches)
    return (
        combined.group_by(["aid", "neighbor"])
        .agg(pl.col("weight").sum())
        .sort(["aid", "neighbor"])
        .with_columns(
            pl.col("weight").rank(method="ordinal", descending=True).over("aid").alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .drop("rank")
        .sort("weight", descending=True)
    )
