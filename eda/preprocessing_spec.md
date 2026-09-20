# Preprocessing specification

Ordered list of every transformation chosen in the EDA (`notebooks/EDA.ipynb`, decisions in `eda/eda_log.md`). Applies to the candidate table `data/features/train_asof11.parquet` (8,000,000 rows = 40,000 sessions x 200 candidates, development window only). Target: `y = label > 0` (assumed, see the review list in `eda_summary.md`).

Anything that learns from data (median, IQR, class weights) is fitted on the training part only, never on validation or the holdout.

## Common to both model families

| # | Step | Detail | Log entry |
|---|---|---|---|
| 1 | Target | `y = (label > 0)`; `label`, `y_click`, `y_cart`, `y_order` are outcome columns and are never features | Step 0 |
| 2 | Keep all rows | No row is removed: no missing-value drop, no outlier removal, no capping. The flagged extreme rows carry the signal | Steps 5, 7 |
| 3 | Drop columns | `item_seen` and `category_conf` (constant), `rrf` (derived from `rank_covis` and `rank_pop`), `hours_since_last_seen` (drifts by construction, tied when dropped), `category` and `aid` (no encoding beats dropping them), `session` (identifier) | Steps 6, 10, 13, 15, 17 |
| 4 | Feature set (15) | `rank_covis`, `rank_pop`, `log_clicks`, `log_clicks_3d`, `cart_rate`, `order_rate`, `pop_cat_pct`, `item_cat_size`, `covis_max`, `covis_mean`, `covis_wsum`, `prefix_len`, `prefix_n_cart`, `prefix_n_order`, `prefix_share_same_cat` | Steps 6-15 |
| 5 | Duplicates | When the events table is rebuilt: drop exact duplicate click and cart events (keep the first); keep order duplicates (likely quantities). Not applied to the current candidate table (effect measured in issue #109) | Step 4 |
| 6 | Split | Train on sessions starting on days 11-12, validate on day 13; option comparisons used session-grouped 3-fold CV; the holdout (sessions starting on days 14-15) is not read until the final modelling evaluation | Steps 0, 15 |

## Linear model (logistic regression)

| # | Step | Detail | Log entry |
|---|---|---|---|
| L1 | Missing values | `rank_covis` and `rank_pop` code "not in this list" as 201: keep 201 as the value and add two indicator columns `rank_covis == 201`, `rank_pop == 201` (17 columns in total) | Step 5 |
| L2 | Transform | `log1p(max(x, 0))` on the 10 skewed features: `rank_covis`, `cart_rate`, `order_rate`, `pop_cat_pct`, `covis_max`, `covis_mean`, `covis_wsum`, `prefix_len`, `prefix_n_cart`, `prefix_n_order`. Indicators and the other features are not transformed | Steps 7, 8 |
| L3 | Scale | Robust scaling (median and IQR) fitted on the training rows, applied to all 17 columns | Step 8 |
| L4 | Categorical | None: `category` and `aid` are not used | Step 13 |
| L5 | Imbalance | None: no class weights, no resampling | Step 14 |
| L6 | Model | `LogisticRegression(C=1.0, max_iter=300)` | Step 16 |

## Gradient boosting (LightGBM)

| # | Step | Detail | Log entry |
|---|---|---|---|
| G1 | Missing values | Replace the code 201 by NaN in `rank_covis` and `rank_pop` and let LightGBM handle missing values natively; no indicator | Step 5 |
| G2 | Transform and scale | None (trees are insensitive to monotone transforms: all variants tied) | Steps 7, 8 |
| G3 | Categorical | None: `category` and `aid` are not used | Step 13 |
| G4 | Imbalance | `class_weight="balanced"` (default LightGBM without weights scores 0.037 against 0.060 with) | Step 14 |
| G5 | Model | `LGBMClassifier(n_estimators=100, random_state=42)` at otherwise default settings | Step 16 |

## Reproducibility

Seed 42 everywhere; three fold seeds (42, 43, 44) for repeated comparisons and three 6,000-query samples (seeds 42-44) for the time-based estimate. Events sample checked against `data/sample/sample_manifest.json` (sha256). Library versions are pinned in `uv.lock`.

## Confidence

Low: G1 (gradient boosting missing-value handling, all options tied) and the choice to keep the whole co-visitation group for gradient boosting (Step 10). Medium: L1, L3, the class-weight choice G4 and the drop of `hours_since_last_seen` (all rest on ties broken by the simplicity rule). High: L2 (log1p beats no transform by more than the noise), keeping all rows.
