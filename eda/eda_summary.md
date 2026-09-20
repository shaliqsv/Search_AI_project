# EDA summary: search and ranking on OTTO sessions

Written from `eda/eda_log.md` (one entry per step, with the options rejected), the tables in `eda/tables/`, and `notebooks/EDA.ipynb`. GitHub issues #91 and #102-#108 carry the same trail; #5 is the umbrella.

## 1. Dataset and task

OTTO e-commerce sessions (clicks, carts, orders) sampled to 1,037,192 development sessions and 13,064,647 events (4 weeks; development = events before day 14). The item catalogue has no text, queries or positions, so query categories are synthetic (80 clusters) and labelled so throughout.

The EDA works on the candidate table built in issue #22: 40,000 sessions that start on days 11-13, 200 candidate items each (8,000,000 rows). Each row is one (session prefix, candidate item) pair with 18 numeric features and a graded outcome `label` (0-3). **Assumed target: `y = label > 0`** (the candidate was found in the rest of the session), 35,055 positives = 0.438% (1 in 227). The metric for the EDA is PR-AUC (ROC-AUC hides the difference, see Step 14); the modelling phase reports NDCG@10 with bootstrap intervals.

## 2. Decision table

Winners for the linear model / gradient boosting (GBM). "Tie" means the options were within the fold noise and the simplest won by the guide's rule.

| Step | What was compared | Decision | Key numbers | Confidence |
|---|---|---|---|---|
| 0 Framing | random, grouped, time-ordered holdout | time-ordered holdout: sessions starting on days 14-15 (100,437 sessions), never read | dev sessions 1,037,192 | High (target is an assumption) |
| 1 First look | manual, per-column table, automated profile | ids and `ts` flagged; profile skipped (needs numpy < 2.2) | 13,064,647 x 4, no nulls | High |
| 2 Validity | dtypes, sentinels, ranges, cross-field | nothing to fix; sentinel-like ids are ordinary ids | 0 range violations; 2.1% carts without earlier click (normal) | High |
| 3 Target | distribution, minority count, stability | severe imbalance, stable over days and categories | 0.438% positive; clicks 34,759, carts 2,512, orders 589 | High |
| 4 Duplicates | exact, key, near, conflicting labels; treatments | drop click/cart copies, keep order copies | 19,990 exact copies (0.153%); 8.7% of order events involved | Medium |
| 5 Missing | imputation and drop options (mean, median, sentinel, indicator, KNN, iterative, drop columns/rows) | linear: sentinel 201 + indicator; GBM: leave NaN | `rank_covis` missing (code 201) in 77.0%, `rank_pop` 12.3%; target 0.13% vs 1.48% | Linear Medium, GBM **Low** (tie) |
| 6 Univariate | summaries, plots, tests | mark 12 skewed features; drop constant `item_seen` | 13 of 18 features skewed; 5 with more than half zeros | High |
| 7 Outliers | IQR, z, percentile, Isolation Forest, LOF; keep/cap/remove | keep every row, no cap | flagged rows carry the signal: 1.99% vs 0.27% target | High |
| 8 Transforms | 6 transforms x 4 scalers | linear: log1p + robust; GBM: none | log1p + robust 0.0594 vs none + standard 0.0485 (PR-AUC) | High (log1p), Medium (scaler) |
| 9 Features vs target | Pearson, Spearman, MI, AUC/KS, decile rates | keep all, no leakage suspect | `rank_covis` AUC 0.18 (0.82 reversed), covis features about 0.80; `category` Cramer's V 0.015 | High |
| 10 Features vs features | correlation, VIF, clustering, drop tests | drop `rrf`; keep the four-feature covis group | Spearman 0.98-0.998 inside the group; dropping the group costs 0.012 (linear) | Linear Medium, GBM **Low** |
| 11 Multivariate | PCA, t-SNE, KMeans, GMM | no segment model; cluster id tested later | 5 of 16 components = 90% variance; k = 2 silhouette 0.68 (5.6% of rows, 0.98% target) | Medium |
| 12 Time, drift, leakage | trends, PSI, adversarial AUC, leakage screen | no leakage; `hours_since_last_seen` flagged | PSI 11.3 for that feature, others below 0.01; adversarial AUC 0.983; medians 19, 43, 66 h on days 11-13 | High (cause), Medium (effect) |
| 13 Categorical | drop, ordinal, frequency, one-hot, target enc. | drop `category` and `aid` (tie) | all within 0.002-0.004 noise; target encoding of `aid` hurts GBM (0.031 vs 0.037) | Medium |
| 14 Imbalance | none, weights, under/over-sampling, SMOTE, threshold | linear: none; GBM: class weights | GBM 0.037 unweighted vs 0.059-0.061 weighted; linear 0.0598 best untouched | Medium |
| 15 Split | random, stratified, grouped, time-based; re-verify 5, 7, 8, 10, 13, 14 | time-based split; earlier winners confirmed; drop `hours_since_last_seen` | random 0.0598 / 0.0585 vs grouped 0.0598 / 0.0590; time-based 0.0581 / 0.0541, inside the spread | Medium |
| 16 Baselines | dummy, single features, linear, GBM, permutation importance, slices | no change to earlier decisions; covis dominance checked against Step 12 | dummy 0.0044, covis rank 0.0500, linear 0.0581, GBM 0.0597 | Medium |

## 3. Top data-quality issues

| Issue | Evidence | Suggested owner and fix |
|---|---|---|
| `hours_since_last_seen` is measured from a fixed cutoff (day 11), so it grows by about a day per day | PSI 11.3, adversarial AUC 0.983 | Feature builder (issue #22): measure from the session start, then re-test; needs a rebuilt candidate table |
| Missing co-visitation and popularity ranks are stored as the code 201 | 77.0% and 12.3% of rows | Feature builder: store as null plus explicit flags; the preprocessing spec handles it meanwhile |
| Duplicate click/cart events double-count in co-visitation and popularity | 19,990 exact copies (0.15% of events), mostly orders (kept) | Data loading: drop click/cart copies; measure the effect on closed results in issue #109 |
| Constant placeholder columns | `item_seen` = 1 everywhere; `category_conf` = 1.0 everywhere (placeholder for the classifier of #34) | Feature builder: remove or fill with real values when the classifier exists |
| Only three days of candidate sessions (40,000 sessions) | Steps 12 and 15 | Data owner: more sessions would allow rolling validation and seasonality checks |

## 4. Features excluded

| Column | Reason | Step |
|---|---|---|
| `item_seen`, `category_conf` | constant | 6, 17 |
| `rrf` | exact function of `rank_covis` and `rank_pop` (VIF 18.6); dropping is tied or better | 10 |
| `hours_since_last_seen` | drifts by construction; no signal alone (AUC 0.51); dropping is tied | 12, 15 |
| `category`, `aid` | no encoding beats dropping them beyond the noise | 13 |
| `session` | identifier, used only for grouping folds | 6 |
| `label`, `y_click`, `y_cart`, `y_order` | outcome columns | 0 |

Kept although weak: `prefix_n_cart`, `prefix_len`, `item_cat_size` (shuffling them does not lower the GBM score in Step 16; dropping on importance alone was not tested). Rows dropped: none (`eda/tables/step17_row_ledger.csv`). Column ledger: `eda/tables/step17_excluded_columns.csv`.

## 5. Preprocessing specification

See `eda/preprocessing_spec.md` (ordered, with the variant for each model family). In short: linear = sentinel 201 + two indicators, log1p on 10 skewed features, robust scaling, no class weights; GBM = 201 to NaN, no transform, class weights balanced.

## 6. Split scheme and holdout

Time-based: train on sessions starting on days 11-12, validate on day 13 (three independent 6,000-query samples give the spread). Model comparisons within steps used 3-fold CV grouped by session on a 6,000-query sample (noise = the larger fold standard deviation; close calls repeated over 9 folds). The holdout is the sessions starting on days 14-15 (100,437 sessions), later than every development session and disjoint by session; it is only counted in Step 0 and checked for overlap in Steps 0 and 4. The row-level schemes (random, stratified) scored the same as the grouped one here, so they did not inflate the estimate; they are rejected on the guide's rule.

## 7. Baseline results and headroom

PR-AUC on day 13 (base rate 0.44%), mean over three samples: dummy 0.0044; popularity rank 0.0062 (no signal); co-visitation rank alone 0.0500; logistic regression 0.0581; LightGBM 0.0597 (spread 0.0016-0.0024, so the two models are tied). Shuffling the four co-visitation features costs about 90% of the score: the signal is one family of features. The models add only +0.008 to +0.010 over the co-visitation rank. 76% of candidate rows have no co-visitation list and hold 20% of the positives; there the models barely beat the base rate (0.0034 GBM, 0.0015 linear against 0.0011). Longer prefixes are harder (0.048 for 4+ items against 0.071 for one item). The k = 2 cluster id of Step 11 adds nothing. ROC-AUC is 0.84 for both models and hides these differences.

## 8. Assumptions needing confirmation

1. **Target**: `y = label > 0` was assumed from issues #10 and #22 (issue #91 stays open until confirmed). The graded labels (click 1, cart 2, order 3) are kept for the ranking metric.
2. **`hours_since_last_seen`**: dropped by the tie rule. Whether a version measured from the session start would carry signal is untested (issue #103 stays open).
3. Order duplicates are real quantities (an inference; Medium).
4. The co-visitation matrix and item statistics are point in time as of day 11 (documented in `notebooks/features.py`, audited in closed issue #12; not re-audited in the EDA).

## 9. Needs human review (everything logged with Low confidence, plus ties that changed a column set)

- **Step 5, GBM missing values (Low)**: all options tied; left as NaN by the simplicity rule.
- **Step 10, GBM redundancy (Low)**: dropping the covis group is also within the noise for the GBM; the whole group is kept.
- **Step 0 target (assumed)** and **Step 12 hours feature** as above.
- **Step 7 (informational)**: outliers are real and informative; no action, but do not clip in later phases.
- **Step 16 dominance**: one feature family carries about 90% of the score; the models' extra value is small. Worth a deliberate decision on whether the modelling phase should spend effort on the 76% of rows without a co-visitation list.
- **Not done**: a full recompute with the caches cleared (`EDA_FORCE=1`); the effect of de-duplication on the closed results (issue #109).

## 10. Recommended next steps for the modelling phase

1. Start from the 15-feature set, the two preprocessing variants and the time-based split; compare to the Step 16 baselines with NDCG@10 and paired bootstrap (the guide for that phase is `_doc/modeling_guide.md`).
2. Report every result twice: on all rows and on the no-co-visitation slice, where the current models are close to the base rate.
3. Decide whether to rebuild `hours_since_last_seen` from the session start and re-test it.
4. Tune the GBM only after the baseline table exists; recalibrate probabilities if a probability is ever shown (class weights inflate them to about 0.26-0.31 against a true 0.0044).
5. Repeat the split check when more days of sessions are available.
