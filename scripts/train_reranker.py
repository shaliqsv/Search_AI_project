"""Phase 6: train a LightGBM lambdarank reranker on the Phase 4 candidate pool.

Scope trimmed per D2 (_doc/decisions/D2-phase6-scope.md): one learned model
(LightGBM) compared against the Phase 5 co-visitation baseline via paired
bootstrap on val. Trained/tuned on train/val only - test stays locked until Phase 7.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from ranking.eval.bootstrap import bootstrap_ci, paired_bootstrap_pvalue
from ranking.eval.metrics import ndcg_at_k

FROZEN_DIR = Path(__file__).resolve().parent.parent / "data" / "frozen"
FEATURE_COLS = ["score", "item_popularity", "prefix_len", "category_match", "category", "query_category"]
CATEGORICAL_COLS = ["category", "query_category"]
SEED = 42


def load(split: str) -> pl.DataFrame:
    features = pl.read_parquet(FROZEN_DIR / "features" / f"{split}.parquet")
    prefix_len = pl.read_parquet(FROZEN_DIR / "ranking_examples.parquet").filter(
        pl.col("split") == split
    ).select("session", "prefix_len")
    return (
        features.join(prefix_len, on="session", how="left")
        .with_columns(pl.col("category_match").cast(pl.Int8))
        .sort(["session", "rank"])
    )


def group_sizes(df: pl.DataFrame) -> list[int]:
    return df.group_by("session", maintain_order=True).agg(pl.len()).select("len").to_series().to_list()


def per_session_ndcg(df: pl.DataFrame, examples: pl.DataFrame, score_col: str) -> list[float]:
    """NDCG@10 per session, ranking by `score_col` (descending), 0 for empty-pool sessions."""
    ranked = df.sort(["session", score_col], descending=[False, True])
    per_session = ranked.group_by("session", maintain_order=True).agg(pl.col("grade").alias("gains"))
    joined = examples.select("session").join(per_session, on="session", how="left")
    return [ndcg_at_k(g, k=10) if g is not None else 0.0 for g in joined["gains"].to_list()]


def main() -> None:
    train = load("train")
    val = load("val")
    train_groups = group_sizes(train)
    val_groups = group_sizes(val)
    print(f"train: {train.height:,} rows, {len(train_groups):,} groups")
    print(f"val: {val.height:,} rows, {len(val_groups):,} groups")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=SEED,
        verbosity=-1,
    )
    ranker.fit(
        train.select(FEATURE_COLS).to_pandas(),
        train["grade"].to_pandas(),
        group=train_groups,
        eval_set=[(val.select(FEATURE_COLS).to_pandas(), val["grade"].to_pandas())],
        eval_group=[val_groups],
        eval_at=[10],
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"best iteration: {ranker.best_iteration_}")

    val = val.with_columns(pl.Series("pred", ranker.predict(val.select(FEATURE_COLS).to_pandas())))

    val_examples = pl.read_parquet(FROZEN_DIR / "ranking_examples.parquet").filter(pl.col("split") == "val")
    baseline_ndcg = per_session_ndcg(val, val_examples, "score")
    reranker_ndcg = per_session_ndcg(val, val_examples, "pred")

    b_mean, b_lo, b_hi = bootstrap_ci(baseline_ndcg, seed=0)
    r_mean, r_lo, r_hi = bootstrap_ci(reranker_ndcg, seed=0)
    diff, p_value = paired_bootstrap_pvalue(reranker_ndcg, baseline_ndcg, seed=0)

    print(f"baseline (score-ranked) NDCG@10: {b_mean:.4f} [{b_lo:.4f}, {b_hi:.4f}]")
    print(f"reranker (LightGBM)     NDCG@10: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"paired diff (reranker - baseline): {diff:.4f}, p={p_value:.4f}")

    importance = dict(zip(FEATURE_COLS, [int(x) for x in ranker.feature_importances_]))
    print("feature importance (gain-split counts):", importance)

    model_path = FROZEN_DIR / "reranker_lgbm.txt"
    ranker.booster_.save_model(str(model_path))

    out = {
        "seed": SEED,
        "feature_cols": FEATURE_COLS,
        "best_iteration": ranker.best_iteration_,
        "baseline_ndcg_at_10": {"mean": b_mean, "ci_lower": b_lo, "ci_upper": b_hi},
        "reranker_ndcg_at_10": {"mean": r_mean, "ci_lower": r_lo, "ci_upper": r_hi},
        "paired_diff": diff,
        "paired_p_value": p_value,
        "feature_importance": importance,
        "model_path": str(model_path),
    }
    (FROZEN_DIR / "reranker_val_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {FROZEN_DIR / 'reranker_val_results.json'}")


if __name__ == "__main__":
    main()
