"""Build and freeze the Phase 1 dataset snapshot from the real OTTO sample.

Split boundary (formally locked in the Phase 2 issue): the first 2 weeks by ts = train,
week 3 = validation, week 4 = test. Correction (see Phase 1 issue #2): OTTO "sessions"
are not short browsing sessions - many span the entire 4-week window (median span
~7.4 days, 51% > 7 days). Bucketing by a session's first event (as an earlier version
of this script did) silently pulled later weeks' events into "training" data. The
split must key off each EVENT's own timestamp, not session identity - so train/val/test
membership for both the co-visitation/category fit and each ranking example is decided
by comparing timestamps directly against the cutoffs below.

Items with fewer than MIN_ITEM_FREQ occurrences in the training window get
category = -1 (too sparse to place reliably; a disclosed, real data-quality limit).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ranking.data.categories import cluster_categories
from ranking.data.covisitation import covisitation_pairs
from ranking.data.labels import build_ranking_examples

MIN_ITEM_FREQ = 5
N_CATEGORIES = 80
SEED = 0
MS_PER_WEEK = 7 * 24 * 60 * 60 * 1000

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "frozen"


def main() -> None:
    events = pl.read_parquet(SAMPLE_DIR / "events.parquet")
    ts_min = events["ts"].min()
    train_end = ts_min + 2 * MS_PER_WEEK
    val_end = ts_min + 3 * MS_PER_WEEK
    print(f"train_end={train_end}  val_end={val_end}")

    train_events = events.filter(pl.col("ts") < train_end)
    print(f"train events (ts-based, any session): {train_events.height:,}")

    item_freq = train_events.group_by("aid").agg(pl.len().alias("n"))
    frequent_aids = item_freq.filter(pl.col("n") >= MIN_ITEM_FREQ)["aid"].to_numpy()
    sparse_aid_count = item_freq.height - len(frequent_aids)
    print(
        f"frequent items (>= {MIN_ITEM_FREQ} in train): {len(frequent_aids):,} / "
        f"{item_freq.height:,} ({sparse_aid_count:,} too sparse -> category -1)"
    )

    print("computing co-visitation on training window...")
    covis = covisitation_pairs(
        train_events.filter(pl.col("aid").is_in(frequent_aids.tolist())), max_gap_events=5
    )
    print(f"co-visitation pairs: {covis.height:,}")

    print("clustering categories...")
    clustered = cluster_categories(
        covis, frequent_aids, n_categories=N_CATEGORIES, n_components=32, seed=SEED
    )

    all_aids_in_sample = events["aid"].unique().to_frame()
    item_categories = all_aids_in_sample.join(clustered, on="aid", how="left").with_columns(
        pl.col("category").fill_null(-1)
    )
    unknown_share = (item_categories["category"] == -1).mean()
    print(f"items with category -1 (unknown/sparse): {unknown_share:.1%}")

    print("building ranking examples (all sessions)...")
    examples = build_ranking_examples(events, item_categories).with_columns(
        pl.when(pl.col("label_ts") < train_end)
        .then(pl.lit("train"))
        .when(pl.col("label_ts") < val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )

    split_counts = examples["split"].value_counts().sort("split")
    print(split_counts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.write_parquet(OUT_DIR / "events.parquet")
    item_categories.write_parquet(OUT_DIR / "item_categories.parquet")
    examples.write_parquet(OUT_DIR / "ranking_examples.parquet")
    covis.write_parquet(OUT_DIR / "train_covisitation.parquet")

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    manifest = {
        "version": datetime.now(UTC).strftime("%Y-%m-%d"),
        "source_sample_sha256": (SAMPLE_DIR / "sample_manifest.json").read_text(),
        "split_rule": "by label_ts (held-out event's own ts), not session identity",
        "split_cutoffs_ms": {"ts_min": int(ts_min), "train_end": train_end, "val_end": val_end},
        "example_counts": {
            row["split"]: row["count"] for row in split_counts.to_dicts()
        },
        "min_item_freq_for_category": MIN_ITEM_FREQ,
        "n_categories": N_CATEGORIES,
        "items_total": item_categories.height,
        "items_unknown_category": int((item_categories["category"] == -1).sum()),
        "ranking_examples": examples.height,
        "seed": SEED,
        "files": {
            f.name: sha256(f)
            for f in [
                OUT_DIR / "events.parquet",
                OUT_DIR / "item_categories.parquet",
                OUT_DIR / "ranking_examples.parquet",
                OUT_DIR / "train_covisitation.parquet",
            ]
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
