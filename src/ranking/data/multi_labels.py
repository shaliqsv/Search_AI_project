"""Multi-event-type labels matching the real OTTO Kaggle competition format (D3).

Supersedes labels.py's leave-last-out, search-styled construction. Per the official
spec (otto-de/recsys-dataset/KAGGLE.md): for a session truncated at a random point,
the ground truth is the single *next click* (not a set) plus the *sets* of all future
cart and order aids. Sessions are randomly truncated the same way the real Kaggle test
file was, so local validation matches the real task.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ranking.data.event_types import CART, CLICK, ORDER


def _summarize_prefix_future(ordered: pl.DataFrame, cut_per_session: pl.DataFrame) -> pl.DataFrame:
    """Shared aggregation: given each session's chosen cut rank, build the label row.

    `cut_per_session` needs columns: session, _cut (the rank of the last prefix event).
    """
    tagged = ordered.join(cut_per_session, on="session", how="inner").with_columns(
        (pl.col("_rank") <= pl.col("_cut")).alias("_is_prefix")
    )
    prefix = tagged.filter(pl.col("_is_prefix"))
    future = tagged.filter(~pl.col("_is_prefix"))

    prefix_summary = prefix.group_by("session").agg(
        pl.len().alias("prefix_len"), pl.col("ts").max().alias("prefix_last_ts")
    )
    click_label = (
        future.filter(pl.col("type") == CLICK)
        .sort(["session", "ts"])
        .group_by("session", maintain_order=True)
        .agg(pl.col("aid").first().alias("click_label"))
    )
    cart_labels = (
        future.filter(pl.col("type") == CART)
        .group_by("session")
        .agg(pl.col("aid").unique().alias("cart_labels"))
    )
    order_labels = (
        future.filter(pl.col("type") == ORDER)
        .group_by("session")
        .agg(pl.col("aid").unique().alias("order_labels"))
    )
    return (
        prefix_summary.join(click_label, on="session", how="left")
        .join(cart_labels, on="session", how="left")
        .join(order_labels, on="session", how="left")
        .with_columns(
            pl.col("cart_labels").fill_null([]),
            pl.col("order_labels").fill_null([]),
        )
    )


def build_multi_label_examples(events: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """One row per session with >= 2 events: a random prefix/future split and labels.

    The cut point is sampled uniformly across the *whole* session, with no regard for
    which week it lands in. Fine for a single global dataset; use
    `build_weekly_multi_label_examples` instead when you need to control which week
    each example's cut point falls in (training-stats weeks vs. a specific
    train/validation week - see Phase 1 issue #2's split design).

    `events` needs columns: session, aid, ts, type. Returns: session, prefix_len,
    prefix_last_ts, click_label (Int32, null if no future click exists), cart_labels
    (List[Int32], possibly empty), order_labels (List[Int32], possibly empty).
    """
    ordered = events.sort(["session", "ts"]).with_columns(
        pl.int_range(pl.len()).over("session").alias("_rank"),
        pl.len().over("session").alias("_session_len"),
    )
    eligible = ordered.filter(pl.col("_session_len") >= 2)

    sessions = eligible.select("session", "_session_len").unique(subset="session")
    rng = np.random.default_rng(seed)
    # cut in [0, _session_len - 2]: prefix = ranks <= cut, future = ranks > cut,
    # so both the prefix and the future always have at least one event
    cuts = rng.integers(0, (sessions["_session_len"] - 1).to_numpy())
    cut_per_session = sessions.select("session").with_columns(pl.Series("_cut", cuts))

    return _summarize_prefix_future(eligible, cut_per_session)


def build_weekly_multi_label_examples(
    events: pl.DataFrame, week_start_ms: int, week_end_ms: int, seed: int = 0
) -> pl.DataFrame:
    """Like `build_multi_label_examples`, but the cut point is constrained to fall
    within [week_start_ms, week_end_ms).

    Needed because OTTO sessions can span the entire observation window (Phase 1 issue
    #2's correction) - sampling a cut point uniformly across a whole session gives no
    control over which week it lands in. This instead only considers a session's
    *in-window* events as candidate cut points, so the caller can build one week's
    examples (e.g. "week 3 = reranker training examples") independently of another
    (e.g. "week 4 = validation examples") with no overlap. A session qualifies only if
    it has an in-window event that is *not* the session's own last event overall (there
    must be something after the cut to form labels from). The prefix can still include
    out-of-window history from earlier in the same session - that's the user's own
    legitimate past, not a leak.

    Returns the same schema as `build_multi_label_examples`.
    """
    ordered = events.sort(["session", "ts"]).with_columns(
        pl.int_range(pl.len()).over("session").alias("_rank"),
        pl.len().over("session").alias("_session_len"),
    )
    candidates = ordered.filter(
        (pl.col("ts") >= week_start_ms)
        & (pl.col("ts") < week_end_ms)
        & (pl.col("_rank") < pl.col("_session_len") - 1)
    )

    rng = np.random.default_rng(seed)
    candidates = candidates.with_columns(pl.Series("_rand", rng.random(candidates.height)))
    cut_per_session = (
        candidates.sort(["session", "_rand"])
        .group_by("session", maintain_order=True)
        .agg(pl.col("_rank").first().alias("_cut"))
    )

    return _summarize_prefix_future(ordered, cut_per_session)
