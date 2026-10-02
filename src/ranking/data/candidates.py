"""Candidate pool generation: co-visitation neighbours of a session's prefix items.

Phase 4 (feature engineering) / Phase 5-6 (baseline, reranker) input. Candidates are
generated purely from co-visitation - the label is never consulted, so recall against
the true held-out item is a real, measurable property of the pool, not guaranteed.
"""

from __future__ import annotations

import polars as pl


def build_candidate_pool(
    prefix_events: pl.DataFrame, neighbors: pl.DataFrame, top_k: int = 50
) -> pl.DataFrame:
    """Top-K co-visitation candidates per session, aggregated over its prefix items.

    `prefix_events` needs columns: session, aid (one row per prefix event; a session
    may repeat an aid, which correctly gives it more weight below). `neighbors` must
    already be pruned to each item's own top-K (`covisitation.top_k_neighbors`) -
    joining against a raw, unpruned co-visitation table here is a join fan-out that
    can exhaust memory on real data (a popular item can have thousands of neighbours
    across millions of sessions).

    Returns: session, aid (candidate), score (summed weight against every prefix
    item), rank (1 = highest score), for the top `top_k` per session.
    """
    prefix_aids = prefix_events.select("session", "aid").unique()

    scored = (
        prefix_aids.join(neighbors, on="aid", how="inner")
        .join(
            prefix_aids,
            left_on=["session", "neighbor"],
            right_on=["session", "aid"],
            how="anti",
        )
        .group_by(["session", "neighbor"])
        .agg(pl.col("weight").sum().alias("score"))
        # sort by neighbor before ranking so ties break deterministically, not on
        # whatever row order the groupby/batch happened to produce (reproducibility)
        .sort(["session", "neighbor"])
        .with_columns(
            pl.col("score").rank(method="ordinal", descending=True).over("session").alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .rename({"neighbor": "aid"})
        .sort(["session", "rank"])
    )
    return scored
