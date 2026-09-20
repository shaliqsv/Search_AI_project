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

