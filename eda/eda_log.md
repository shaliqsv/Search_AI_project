# EDA log

One entry per step of `_doc/eda_guide.md`, in the guide's template.

### Step 0. Setup and framing
Kind: Diagnostic
Options run: Framing table above (each answer marked given or assumed). Holdout scheme options: time-ordered, grouped by entity, random. Layout eda/{tables,figures}, seed 42, versions printed.
Observed: Development events: ts < day 14 (1,037,192 sessions). Holdout: 100,437 sessions that start on days 14-15. No session is on both sides. The candidates table has 40,000 sessions, all in development (they start on days 11-13).
Decision: Time-ordered holdout = the evaluation window of issue #6 (days 14-15). Everything target-aware reads only events before day 14.
Why: The model scores future sessions, so the latest rows are held out (guide rule). Grouping by session is implied: holdout sessions start after the development window.
Rejected: Random split: rejected, it would mix time periods. Group-only split: rejected, it does not test future data.
Confidence: High
Affects later steps: Steps 12 and 15 re-check the scheme; the holdout is never read before the modelling phase.
Needs human input: yes: the target and task type are assumed from issues #10/#22. Please confirm `label > 0` is the target to use for the guide.

### Step 1. Load and first look
Kind: Diagnostic
Options run: A manual inspection (shape, dtypes, first/last/random rows, describe); B per-column overview table; C automated profile SKIPPED: ydata-profiling requires numpy<2.2 and matplotlib<=3.10, the project uses numpy 2.4.6 and matplotlib 3.11.2, so installing it would downgrade core libraries for every earlier result; with 4 ID-like columns A and B already show its alerts.
Observed: 13,064,647 rows x 4 columns (session int64, aid int32, ts int64, type int8), 274 MB; no missing values; no constant column; session and aid are ID-like; ts is epoch milliseconds; type has 3 codes.
Decision: Prioritised issue list saved to eda/tables/step01_issue_list.csv: convert ts, treat type as categorical, exclude ids from features.
Why: The three options were meant to be combined into one list; two were run and their findings agree.
Rejected: C profile: skipped for the reason above; installing at the cost of downgrading numpy was rejected.
Confidence: High
Affects later steps: Step 2 converts types; Step 13 handles aid as high-cardinality.
Needs human input: no

### Step 2. Types, validity and consistency
Kind: Diagnostic then fixes
Options run: A dtype vs meaning; B sentinel-like values (-1, 0, 999, 9999, 99999); C range and domain rules; D cross-field consistency (ordering inside sessions, cart/order after click/cart). All four run on the development events.
Observed: A: 4 columns, all correctly typed except that ts is epoch ms. B: no placeholder found. The scan flags aid values 0, 999, 9999 and 99999 (2, 4, 44 and 1 events), but item ids are dense integers from 0 to 1,855,602, so these are ordinary ids; session 0 (86 events) is the first session; type 0 is the code for a click (89.8% of events). Details in eda/tables/step02_B.csv. C: 0 range violations (ts inside the 4 weeks, type in {0,1,2}, ids non-negative). D: 0 backwards timestamps inside sessions; 2.1% of carted items had no earlier click in the same session and 0.3% of ordered items had no earlier cart.
Decision: No fixes needed for A-C. D is a property of the data (users can add to the cart from a page without a logged click), not an error. The sentinel-like values are not placeholders (evidence above), so nothing goes on the human question list.
Why: Rule: apply unambiguous fixes, convert impossible values to missing with a flag, never delete rows. There was nothing impossible.
Rejected: Nothing to reject: all impossible-value options found no violations. Treating cart-without-click as an error was rejected because it is common and plausible.
Confidence: High
Affects later steps: Step 4 uses the key (session, aid, ts, type); Step 12 must not assume a click precedes a cart.
Needs human input: no

### Step 3. Target analysis
Kind: Diagnostic
Options run: A class counts and shares; B absolute minority counts; C target rate by day and by the 5 largest categories; D missing target.
Observed: 8,000,000 rows, 35,055 positives (0.438%); ratio 1:227. Positives by type: click 34,759, cart 2,512, order 589. Target rate by start day and by category in eda/tables/step03_C_*.csv. No missing target.
Decision: Class imbalance is SEVERE (minority under 1 percent): Step 14 handles it. The cart and order classes are small but above 100 rows; orders are the rarest.
Why: Guide rules: flag imbalance under 10 percent, severe under 1 percent; remove rows with no target (none).
Rejected: No option was rejected; all four were run.
Confidence: High
Affects later steps: Step 14 must use PR-AUC; Step 15 uses grouped folds because a session has 200 rows; the order task has few positives, so per-task results are noisy.
Needs human input: no

### Step 4. Duplicates
Kind: Selection (how to treat them)
Options run: A exact; B key-based ((session, aid, type, ts) and (session, aid, ts)); C near-duplicates (same session, item, type in the same second); D conflicting labels on the candidates table (features rounded to 3 decimals). Treatments considered: keep everything, drop exact duplicates keeping the first, drop only the click and cart duplicates, aggregate per key.
Observed: A: 19,990 rows (0.153%) are exact copies of an earlier row, in 19,413 groups (largest group 6). B is identical to A (no key repeats with a different value or type). C: 35,752 rows (0.274%). By event type, the copies are mostly ORDERS: 27,031 order events (8.7% of all order events) are involved, against 11,454 clicks and 918 carts. D: 7 feature combinations have conflicting targets, covering 14 rows (0.00017% of 8,000,000); each holds one positive and one negative, i.e. 7 positives (0.02% of all positives). Duplicates across the development/holdout boundary: none (no session is on both sides).
Decision: Drop exact duplicate CLICK and CART events, keeping the first (double logging in the same millisecond has no business reason). KEEP order duplicates: an order of the same item logged twice at the same millisecond is most likely a quantity above 1, which is real. Conflicting labels: none to treat. Not applied to the data used by the earlier issues (#7, #17-#29): the events involved are 0.10% of clicks and 0.09% of carts (copies including the first occurrence), and the effect on those results is measured in follow-up issue #109.
Why: Guide rules: exact duplicates with no business reason are dropped keeping the first; legitimate repeated observations are kept; conflicting labels are quantified and never deleted without approval. Orders have a plausible business reason (quantity), clicks and carts do not.
Rejected: Keep everything: rejected for clicks and carts (they would double count in co-visitation and popularity). Drop all exact duplicates: rejected because it would remove real order quantities (8.7% of order events are involved). Aggregate per key: rejected, it changes the event grain that Steps 12 and 16 rely on. Near-duplicates (C): NOT dropped: a second click on the same item within one second cannot be told from a real fast repeat without inspecting a sample, and the guide says drop near duplicates only if a sample shows they are the same record.
Confidence: Medium (the order-quantity reading is an inference, not something the data proves)
Affects later steps: Affects the event counts by 0.15%; co-visitation, popularity and features from earlier issues used the un-deduplicated events. Follow-up issue #109 quantifies the effect on the closed results. Step 15 uses grouped folds by session regardless.
Needs human input: no

### Step 5. Missing values
Kind: Diagnostic (5a) then Selection (5b)
Options run: 5a: A counts, B co-missingness, C relation to target/prefix length. 5b (10 options, both families, 3-fold grouped CV, 6,000 queries): leave missing (native GBM), sentinel 201 with and without indicator, mean, median, median + indicator, segment median, drop columns above 30% / 50% missing, keep up to 80% (median fill); drop rows (reported, not ranked); KNN (k=5) and iterative imputation (on a 500-query sample, next to two cheap options on the same rows). Close calls repeated with 2 extra fold seeds. Categorical fills: not applicable (category has no missing values).
Observed: No explicit nulls in either table. The candidates table CODES missing values as 201: rank_covis in 77.0% of rows and rank_pop in 12.3%. Both are informative: target rate 0.13% when rank_covis is missing vs 1.48% present; 1.20% vs 0.33% for rank_pop. The two are almost mutually exclusive (phi -0.685): a candidate comes from the co-visitation list, the popularity list, or both, so it is missing from at most one. Classification: MNAR-like (informative, structural). Linear model: median + indicator, sentinel + indicator and sentinel alone are tied at PR-AUC about 0.048; mean, median and dropping the column are lower (0.042-0.044). Gradient boosting: all options within noise (PR-AUC 0.034-0.039). KNN and iterative imputation do not beat median + indicator on the small sample (linear 0.048 and 0.049 vs 0.051). Dropping rows keeps only 11.1% of rows and 44.4% of positives (PR-AUC not comparable). No imputed distribution matches the observed one (KS 0.40-1.0), which is expected when the value means 'not in the list'.
Decision: Linear model: sentinel 201 + indicator. Gradient boosting: leave missing (native, gbm only). One variant per model family in the preprocessing spec.
Why: Guide rules: options within the larger fold standard deviation of the best are tied (linear: 3 options, gbm: 8 options after repeating the CV with 3 fold seeds); among tied options take the simplest (leave as is < fixed rule < learned statistic < model). Missingness is informative, so an option that keeps the information (an indicator, or a sentinel the trees can split on) is preferred.
Rejected: Drop columns above 30% or 50%: rejected, it removes the strongest missingness signal (rank_covis is missing for 77%), PR-AUC 0.042 (linear) and 0.034 (gbm). Keeping columns up to 80%: identical to the median fill (no column is above 80%). Mean/median/segment median: lower for the linear model because they erase the fact that the candidate was not returned. KNN and iterative: no better than the simple options and far more expensive (KNN took 86 s on 100,000 rows). Drop rows: rejected, it discards 89% of rows and 56% of positives and would bias the sample.
Confidence: Linear: Medium (three tied options, chosen by the simplicity rule). Gradient boosting: Low (all options tied: the choice does not matter much, the tie-break picked the simplest). The gbm choice goes on the needs-human-review list.
Affects later steps: Step 6-7 use the coded values as they are; Step 14 keeps in mind that most negatives have rank_covis = 201, so one code carries much of the signal. The two coded columns must be treated the same in training and serving (issue #58).
Needs human input: no

### Step 6. Univariate analysis
Kind: Diagnostic
Options run: Numeric: A summary statistics (mean, sd, min, percentiles 1-99, max, skew, kurtosis, share of zeros), B histogram + box plot + ECDF for all 18 features (eda/figures/step06_*.png), C Q-Q plots of the 4 most skewed features and a Shapiro test on 5,000 rows. Categorical: A level counts, B cardinality and rare levels, C consistency. Also the shape of the events table (events per session, per item, per day).
Observed: 13 of 18 numeric features have |skew| > 1 (rank_covis, rrf, cart_rate, order_rate, pop_cat_pct, item_seen, hours_since_last_seen, covis_max, covis_mean, covis_wsum, prefix_len, prefix_n_cart, prefix_n_order). 5 have more than half zeros (covis_max, covis_mean, covis_wsum, prefix_n_cart, prefix_n_order). Constant: item_seen; near-constant (> 99.9% one value): none. Every Shapiro test rejects normality (p < 0.05) on 5,000 rows, as expected for counts and ranks; the Q-Q plots show heavy right tails for the most skewed features. `category` has 80 levels, none missing, largest level 5.6%. Events: median 5 events per session (max 485), median 3 events per item (max 8,071).
Decision: Actions: mark the skewed features for the transform test in Step 8; add an 'is zero' flag for the zero-spike columns if Step 9 shows the zero group differs; drop the constant column(s); check the target rate of near-constant columns before dropping; keep ids out of the features.
Why: The guide's rules for the actions above were applied as written; the Shapiro p-values were read with the plots, not on their own (with millions of rows every test rejects).
Rejected: Nothing to reject: every option was run and describes a different aspect. Dropping near-constant columns straight away was rejected because a rare value can be the signal (checked against the target above).
Confidence: High
Affects later steps: Step 8 tests transforms on the skewed features; Step 7 uses the tails seen here; Step 13 handles category (80 levels) and aid.
Needs human input: no

### Step 7. Outliers
Kind: Diagnostic (7a) then Selection (7b)
Options run: 7a: IQR fences, z-score > 3, modified z-score > 3.5 (skipped where MAD = 0), 1st/99th percentile cut, Isolation Forest (200,000 rows), Local Outlier Factor (20,000 rows); impossible-value checks; flagged-row patterns; target rate of flagged rows. 7b (3-fold grouped CV, both families): keep as is, cap 1/99, cap 5/95, log1p of the skewed features, keep + outlier-count flag, remove flagged rows (in 3+ columns, training folds only, 1.8% of rows). 'Model the extreme segment separately' was considered and not run (reason below).
Observed: Flag shares differ a lot between methods (IQR 0-23%, z-score 0-2.8%, percentile 1%); the IQR share for the co-visitation columns (23%) is an artefact of zero inflation (77% zeros, so every non-zero value is beyond the fence). The modified z-score is undefined (MAD = 0) for 7 columns (rank_covis, item_seen, the three covis columns, prefix_n_cart, prefix_n_order). Multivariate: Isolation Forest flags 9.6% of rows and LOF 1.4%; they agree very little (Jaccard 0.05). Impossible values: none. cart_rate above 1 in 1,300 rows (max 2.78) is not an error: it is carts per click and exceeds 1 when an item is carted more than clicked. Every flagged pattern is a real extreme (long sessions, large co-visitation weights, high cart/order rates). Flagged rows carry the signal: target rate 1.99% for Isolation Forest flags vs 0.27% for the rest, and 0.88% vs 0.15% for rows outside the IQR fence in any column. 7b: linear PR-AUC log1p 0.0587, keep 0.0480, flag feature 0.0494, cap 1/99 0.0446, cap 5/95 0.0391, remove flagged 0.0465. Gradient boosting: keep 0.0382, log1p 0.0381, flag 0.0375, cap 1/99 0.0364, remove 0.0355, cap 5/95 0.0324.
Decision: Keep every row; no capping and no removal. Linear model: log1p of the skewed features (formalised in Step 8). Gradient boosting: keep as is. No outlier flag feature.
Why: Guide rules: errors become missing (none found); real extremes are kept for tree models, and for the linear model the choice among keep, cap and transform is by score: log1p beats keeping by 0.0107, above the noise of 0.0029. For the gradient boosting model 4 options are tied and the simplest, keep as is, wins. Never remove rows because of their target or more than 5% of rows.
Rejected: Cap at 1st/99th and cap at 5th/95th percentile: rejected, lower PR-AUC for both families (the extreme values ARE the signal, so clipping them loses information). Remove flagged rows: rejected, lower for both families and it discards positives. Keep + flag feature: tied with keeping as is, so not added (simplicity). 'Model the extreme segment separately': not run: the flagged population is not coherent (the methods agree at Jaccard 0.05, IQR flags 40% of rows), and the guide only asks for it when flagged rows form a coherent population.
Confidence: High
Affects later steps: Step 8 formally tests transforms (log1p is the linear favourite here); Step 9 must not read the outlier rows as noise: they are where the positives are. Informational note for the human (goes on the review list): outlier rows have a 2x-7x higher target rate.
Needs human input: no

### Step 8. Transformations and scaling
Kind: Selection
Options run: Transforms on the 12 skewed features: none, log1p, square root, Box-Cox (strictly positive columns only), Yeo-Johnson, quantile-normal; scalers: none, standardisation, min-max, robust. Linear model: all 24 transform x scaler combinations. Gradient boosting: the 6 transforms and the 3 scalers one at a time (9 combinations; trees ignore scaling, which the sanity check confirms). Target transform: not applicable (classification). Top linear options repeated over 3 fold seeds.
Observed: Mean absolute skew of the 12 skewed features: none 16.5, log1p 3.2, sqrt 3.5, Box-Cox 14.8 (only strictly positive columns qualify), Yeo-Johnson 1.7, quantile-normal 1.4. Linear PR-AUC: log1p + robust 0.0594, log1p + min-max 0.0589, log1p + standard 0.0582, quantile-normal + standard 0.0580, sqrt + robust 0.0561, versus no transform + standard scaling 0.0485. 8 linear options are tied within the noise (0.0019); repeating the top 3 over 9 folds gives the same order (log1p + robust 0.0596). Gradient boosting: all 9 variants are within noise (spread 0.0010 against a fold sd of 0.0026), so the tree model is insensitive to monotonic transforms, as expected.
Decision: Linear model: log1p + robust. Gradient boosting: no transform and no scaling (none + none).
Why: Guide rules: tree models get no transform unless clearly beneficial (nothing beats none by more than the noise); for the linear model pick the best score inside the noise threshold and tie-break by simplicity (a fixed rule such as log1p before a learned transform); with outliers kept in Step 7 prefer robust scaling among ties.
Rejected: Square root: lower skew reduction than log1p and lower PR-AUC. Box-Cox: cannot be applied to zero-valued columns, so only a few columns changed (skew 14.8, no gain). Yeo-Johnson and quantile-normal: reduce skew most (1.7 and 1.4) and score within noise of log1p, but they are learned per column and more complex, so the tie-break rejects them. Standard and min-max scalers: tied with robust, so rejected by the outlier rule. No transform: 0.0485, about 0.011 below the best (more than the noise).
Confidence: High for using log1p (it beats no transform by more than the noise); Medium for the scaler (three scalers are tied and the choice follows the guide's tie-break).
Affects later steps: The preprocessing spec carries one variant per family: linear = log1p on 12 skewed columns then robust scaling; gbm = raw. Steps 10, 13, 14 and 16 use the linear variant.
Needs human input: no

### Step 9. Features vs target
Kind: Diagnostic
Options run: Numeric: A Pearson, B Spearman, C mutual information (20-quantile-bin estimator on all 8,000,000 rows), D single-feature AUC and KS statistic, E target rate by decile (top 5). Categorical (category, 80 levels): A target rate per level with counts, B chi-square and Cramer's V, C weight of evidence and information value; D (ANOVA) not applicable to a binary target.
Observed: Strongest single features by AUC: rank_covis (0.182), rrf (0.811), covis_mean (0.801), covis_wsum (0.800). Non-linear effects: features where mutual information ranks 5+ places above Pearson are listed in the notebook. No feature is suspiciously strong (none above AUC 0.9 or below 0.1, no |r| above 0.9). category: Cramer's V 0.015, information value 0.049 (weak), with 0 of 80 levels under 30 positives. Features with essentially no signal: cart_rate, hours_since_last_seen, prefix_n_cart, prefix_n_order.
Decision: Rank features by all four measures; do not drop any yet (signal can appear in interactions). Candidate no-signal features are listed above. Keep the zero-inflated columns as they are; an `is zero` flag was not needed because the covis columns already separate at zero (see the decile table). Nothing goes to the leakage review beyond the routine screen in Step 12.
Why: Guide rules: rank by more than one measure and note disagreements; mark no-signal features but do not drop; send suspiciously strong ones to Step 12 (none). The single-feature AUC is preferred to Pearson for a 0.44% target, where correlations are tiny even for useful features.
Rejected: Dropping weak features now: rejected, the guide says signal can hide in interactions and Step 10 and 16 test that. Reading Pearson alone: rejected, it misses the monotone and non-linear effects (see the disagreement table).
Confidence: High
Affects later steps: Step 10 handles the redundancy among the top features; Step 12 screens them for leakage; Step 13 handles `category` (weak, 80 levels).
Needs human input: no

### Step 10. Features vs features
Kind: Diagnostic (findings) then Selection (redundancy)
Options run: A Pearson and Spearman matrices; B categorical association (not applicable: one categorical feature); C variance inflation factor; D hierarchical clustering on the stronger of |Pearson| and |Spearman|. Treatment test under the shared CV, both families: all features; keep covis_max and drop rank_covis, covis_mean, covis_wsum (the correlated group); drop rrf (derived); both drops.
Observed: Pearson finds no pair above 0.9 (the largest is 0.85), but Spearman finds six: covis_max, covis_mean and covis_wsum have rank correlations of 0.996-0.998 with each other and about -0.98 with rank_covis, so they form one group of four with the same ordering of candidates. VIF (5 = watch, 10 = problem): rrf 18.6, rank_covis 16.1, log_clicks 10.0, rank_pop 9.6, log_clicks_3d 5.0; rrf is an exact function of the two rank columns. Treatment test PR-AUC (linear / gbm): all features 0.0582 / 0.0375; keep covis_max, drop the other three 0.0463 / 0.0360; drop rrf only 0.0594 / 0.0373; both drops 0.0345 / 0.0246.
Decision: Drop `rrf` (derived from the two ranks) for both model families. Keep all four members of the co-visitation group.
Why: Guide rule: drop the extras only if the score is not worse by more than the noise (linear 0.0025, gbm 0.0024). Dropping rrf is not worse (linear +0.0012, gbm -0.0002) and rrf is an exact formula of two kept columns. Dropping the correlated group is worse for the linear model by 0.0119, more than the noise.
Rejected: Drop the covis group (keep covis_max): rejected for the linear model (worse by more than the noise, because rank_covis carries the pool-membership code and the ordering that covis_max lacks). Both drops: rejected, much worse for both families: the features compensate for each other. Dropping log_clicks (VIF 10.0) or log_clicks_3d (5.0): not tested, their correlation is below 0.9 and they measure different windows. Dropping by Pearson alone: rejected, it would have found no redundancy at all.
Confidence: Medium for the linear model (the two best options are tied and the guide rule picked the smaller set); Low for gradient boosting (dropping the correlated group is also within the noise, the choice of dropping only rrf follows the guide's advice to keep all members for trees). The gbm choice goes on the needs-human-review list.
Affects later steps: The preprocessing spec excludes item_seen (constant, Step 6) and rrf (redundant, this step). Step 16 therefore uses 16 numeric features plus category.
Needs human input: no

