"""Multi-event-type labels matching the real OTTO Kaggle competition format (D3).

Supersedes labels.py's leave-last-out, search-styled construction. Per the official
spec (otto-de/recsys-dataset/KAGGLE.md): for a session truncated at a random point,
the ground truth is the single *next click* (not a set) plus the *sets* of all future
cart and order aids. Training sessions are randomly truncated the same way the real
Kaggle test file was, so local validation matches the real task.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ranking.data.event_types import CART, CLICK, ORDER


def build_multi_label_examples(events: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """One row per session with >= 2 events: a random prefix/future split and labels.

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

    tagged = eligible.join(cut_per_session, on="session", how="inner").with_columns(
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
