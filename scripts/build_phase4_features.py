"""Build the Phase 4 candidate + feature table for one split.

For every example in the split: generate a co-visitation candidate pool from its
prefix (label never consulted - see candidates.py), then attach point-in-time features
computed from the training window only (item popularity, category), plus the graded
label (label_grade if the candidate is the true held-out item, else 0).

Processed in batches of sessions (BATCH_SIZE) rather than one giant join: the
candidate-generation join fans out to (prefix items x up to NEIGHBOR_K neighbours)
before it can aggregate back down, and on the test split that blew up to 23GB and
thrashed swap (Phase 4 issue). Batching bounds the fan-out regardless of how many
long/high-degree-item sessions land in any one split.

Usage: uv run python scripts/build_phase4_features.py <split>
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import polars as pl

from ranking.data.candidates import build_candidate_pool
from ranking.data.covisitation import top_k_neighbors

TOP_K = 50
NEIGHBOR_K = 50
MAX_PREFIX_ITEMS = 20
BATCH_SIZE = 50_000
FROZEN_DIR = Path(__file__).resolve().parent.parent / "data" / "frozen"
OUT_DIR = FROZEN_DIR / "features"


def log(t0: float, msg: str) -> None:
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)


def main(split: str) -> None:
    t0 = time.time()
    events = pl.read_parquet(FROZEN_DIR / "events.parquet")
    examples = pl.read_parquet(FROZEN_DIR / "ranking_examples.parquet").filter(
        pl.col("split") == split
    )
    covis = pl.read_parquet(FROZEN_DIR / "train_covisitation.parquet")
    item_categories = pl.read_parquet(FROZEN_DIR / "item_categories.parquet")
    log(t0, f"loaded: {examples.height:,} examples in split={split}")

    neighbors = top_k_neighbors(covis, k=NEIGHBOR_K)
    log(t0, f"pruned neighbors: {neighbors.height:,} rows (<= {NEIGHBOR_K} per item)")

    manifest = json.loads((FROZEN_DIR / "manifest.json").read_text())
    train_end_ts = manifest["split_cutoffs_ms"]["train_end"]
    train_events = events.filter(pl.col("ts") < train_end_ts)
    item_popularity = train_events.group_by("aid").agg(pl.len().alias("item_popularity"))
    log(t0, f"item_popularity: {item_popularity.height:,} items")

    all_sessions = examples["session"].to_list()
    n_batches = (len(all_sessions) + BATCH_SIZE - 1) // BATCH_SIZE
    log(t0, f"processing {len(all_sessions):,} sessions in {n_batches} batches of {BATCH_SIZE:,}")

    tmp_dir = OUT_DIR / f"_tmp_{split}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    total_candidates = 0
    total_hits = 0
    for i in range(n_batches):
        batch_sessions = all_sessions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_examples = examples.filter(pl.col("session").is_in(batch_sessions))

        prefix_events = (
            events.join(
                batch_examples.select("session", "prefix_last_ts"), on="session", how="inner"
            )
            .filter(pl.col("ts") <= pl.col("prefix_last_ts"))
            .with_columns(
                pl.col("ts")
                .rank(method="ordinal", descending=True)
                .over("session")
                .alias("_recency")
            )
            .filter(pl.col("_recency") <= MAX_PREFIX_ITEMS)
            .drop("_recency")
        )

        candidates = build_candidate_pool(prefix_events, neighbors, top_k=TOP_K)

        features = (
            candidates.join(item_popularity, on="aid", how="left")
            .with_columns(pl.col("item_popularity").fill_null(0))
            .join(item_categories, on="aid", how="left")
            .with_columns(pl.col("category").fill_null(-1))
            .join(
                batch_examples.select("session", "query_category", "label_aid", "label_grade"),
                on="session",
                how="left",
            )
            .with_columns(
                (pl.col("category") == pl.col("query_category")).alias("category_match"),
                pl.when(pl.col("aid") == pl.col("label_aid"))
                .then(pl.col("label_grade"))
                .otherwise(0)
                .alias("grade"),
            )
            .drop("label_aid", "label_grade")
        )
        features.write_parquet(tmp_dir / f"batch_{i:04d}.parquet")

        hit = (
            batch_examples.join(
                candidates.select("session", "aid").rename({"aid": "cand"}),
                on="session",
                how="left",
            )
            .filter(pl.col("cand") == pl.col("label_aid"))["session"]
            .n_unique()
        )
        total_candidates += candidates.height
        total_hits += hit
        log(
            t0,
            f"batch {i + 1}/{n_batches}: {candidates.height:,} candidates, "
            f"{hit}/{len(batch_sessions)} recall hits",
        )

    log(t0, f"total candidates: {total_candidates:,}")
    log(t0, f"recall@{TOP_K} of true label in pool: {total_hits / len(all_sessions):.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.parquet"
    pl.scan_parquet(tmp_dir / "*.parquet").sink_parquet(out_path)
    shutil.rmtree(tmp_dir)
    log(t0, f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
