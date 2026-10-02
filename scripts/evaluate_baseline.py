"""Phase 5: evaluate the co-visitation-score baseline with NDCG@10 + bootstrap CI.

The baseline ranks each session's candidate pool by its co-visitation `score`
(already computed in Phase 4) - no learning involved. Every val example counts,
including sessions absent from the features table entirely (their candidate pool
was empty - a real, honest 0, not something to silently exclude).

Usage: uv run python scripts/evaluate_baseline.py <split>   (val for Phase 5; test only in Phase 7)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from ranking.eval.bootstrap import bootstrap_ci
from ranking.eval.metrics import ndcg_at_k

FROZEN_DIR = Path(__file__).resolve().parent.parent / "data" / "frozen"


def main(split: str) -> None:
    examples = pl.read_parquet(FROZEN_DIR / "ranking_examples.parquet").filter(
        pl.col("split") == split
    )
    features = pl.read_parquet(FROZEN_DIR / "features" / f"{split}.parquet")

    per_session_gains = (
        features.sort(["session", "rank"])
        .group_by("session", maintain_order=True)
        .agg(pl.col("grade").alias("gains"))
    )

    joined = examples.select("session").join(per_session_gains, on="session", how="left")
    n_missing = joined["gains"].null_count()
    print(f"{split}: {examples.height:,} examples, {n_missing:,} with an empty candidate pool")

    ndcg_scores = [
        ndcg_at_k(gains, k=10) if gains is not None else 0.0
        for gains in joined["gains"].to_list()
    ]

    mean, lower, upper = bootstrap_ci(ndcg_scores, n_resamples=10_000, seed=0)
    print(f"NDCG@10: {mean:.4f}  95% CI [{lower:.4f}, {upper:.4f}]  (n={len(ndcg_scores):,})")

    out = {
        "split": split,
        "method": "covisitation_score_baseline",
        "n_examples": examples.height,
        "n_empty_pool": n_missing,
        "ndcg_at_10": {"mean": mean, "ci_lower": lower, "ci_upper": upper, "n_resamples": 10_000},
    }
    out_path = FROZEN_DIR / f"baseline_{split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
