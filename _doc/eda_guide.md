# EDA Guide for the Agent

## 1. Purpose and how to read this guide

You are performing exploratory data analysis on a tabular dataset the way a careful senior data scientist would. This guide defines the steps, the options to try inside each step, the evidence to collect, and the rule for choosing between options. You write and run all the code yourself. This guide contains none on purpose.

There are two kinds of steps, and each step below is labeled with its kind.

Diagnostic steps: the options are complementary lenses on the same question. Run all of them, then combine what you learn into a list of issues and actions.

Selection steps: the options are alternatives for handling the same problem. Run all of them under identical conditions, compare the results, and pick one using the decision rule written in that step.

The central rule of this guide: run every listed option before you choose. Do not skip an option because you expect it to lose. If you must skip one (it does not apply to the data types present, or the compute cost is prohibitive), say so in the log and give the reason. The person reading your log wants to see the evidence for the choice, including the options that lost.

---

## 2. Inputs

Required from the human. If any are missing, ask before starting.
- Dataset location
- Target column
- Task type (classification or regression)

Optional. If any are missing, make a reasonable assumption, write it in the log marked as "assumed", and continue.
- Row or entity identifier column
- Date or timestamp column
- Primary metric and the relative cost of each error type
- Columns that will not be available at prediction time
- Business rules about valid values
- Constraints (interpretability, latency, privacy)

If the task is unsupervised, skip every part of this guide that refers to the target.

---

## 3. Working principles

1. The raw data is read only. Every change is recorded as an entry in an ordered preprocessing specification and applied to a copy.
2. Evidence over habit. Choose the option the results support, not the one that is most common in tutorials.
3. No leakage inside comparisons. Anything that learns from data (imputers, scalers, encoders, cappers, resamplers, feature selectors) must be fit on the training portion of each cross-validation fold only, then applied to the validation portion.
4. Use the same yardstick for every option within a step: the same rows, the same folds, the same random seed, the same metric, and the same model settings.
5. Score with two model families wherever a comparison is empirical: a regularized linear model and a gradient boosting tree model, both at default settings. This is EDA, so do not tune hyperparameters. A choice that helps one family and hurts the other is allowed, and the preprocessing specification then carries one variant per family.
6. Meaningful difference. Two options are tied when the gap between their mean cross-validation scores is smaller than the larger of their two standard deviations across folds. For a close call, repeat the cross-validation with several different fold seeds before declaring a winner or a tie.
7. Tie-break by simplicity. Among tied options, choose the simpler one. From simplest to most complex: leave as is, a fixed rule, a statistic learned per column, a model-based method.
8. Nothing is dropped silently. Every removed row or column is counted in the log with the reason, and its share of the data is stated.
9. Keep the losing options' results. Save every comparison table, not only the winner.
10. Separate what you observed from what you infer. Write numbers first, interpretation second, and state how sure you are.
11. Holdout discipline. A final holdout is set aside in Step 0. Nothing that uses the target is computed on it. It is not touched during EDA.
12. Reproducibility. Fix the random seed at 42 everywhere, record library versions, and make sure the whole analysis reruns from the raw data with one command.
13. Large data. If the dataset has more than roughly one million rows, run expensive options (nearest-neighbor imputation, iterative imputation, local outlier factor, t-SNE, UMAP, permutation importance) on a stratified sample. Record the sample size, and if it is cheap to do so, confirm the winning option on the full data.
14. Comparison metric. Use the primary metric from the framing. If none was given, use ROC-AUC and PR-AUC for classification, and RMSE and MAE for regression, and report both.

---

## 4. Autonomy and escalation

Work autonomously by default. Apply the decision rule, log the decision with a confidence label, and move on. Confidence labels mean the following.

- High: the winner beats the alternatives by more than the noise threshold, and the result agrees across both model families.
- Medium: the winner is clear for one model family or by a modest margin.
- Low: the options are tied or the evidence conflicts, and you fell back on the simplicity tie-break or on judgment.

Stop and ask the human, with your evidence attached, when any of these occur.

1. A feature looks like it leaks the target (see Step 12), or you cannot tell whether a top feature is available at prediction time.
2. More than about 2 percent of rows have conflicting labels, or more than about 5 percent of a column's values are impossible under the stated business rules.
3. The minority class has fewer than about 100 rows in the development set.
4. The target definition appears to change over time.
5. A decision would remove more than about 5 percent of rows or any column the human named as important.
6. Two results contradict each other and you cannot explain why.
7. The framing information you assumed turns out to matter (for example, the presence or absence of a time order changes the split).

Everything logged with Low confidence is collected into a "needs human review" list in the final summary.

---

## 5. Deliverables and layout

Create these in the working directory.

- `eda_log.md`: one entry per step in the template below, written as you go, not at the end.
- `eda_summary.md`: the final report described in Step 17.
- `tables/`: every comparison table as a file, named by step and topic.
- `figures/`: every figure you rely on, named by step and topic.
- `preprocessing_spec.md`: an ordered, unambiguous list of every transformation chosen, with parameters, the columns it applies to, the model family it applies to, and the log entry that justifies it.
- A rerunnable script or notebook that reproduces every table, figure and the cleaned data from the raw data.
- The cleaned development dataset, plus a list of columns excluded and why.

---

## 6. Log entry template

Every step ends with an entry in this exact shape.

```
### Step N. Title
Kind: Diagnostic or Selection
Options run: (each option with its key numbers; mark any skipped, with the reason)
Observed: (facts and numbers only)
Decision: (the chosen option, or for diagnostic steps the list of actions)
Why: (the evidence that drove the choice, including the rule applied)
Rejected: (each alternative and why it lost)
Confidence: High, Medium or Low
Affects later steps: (what this changes downstream)
Needs human input: (yes or no, and the question if yes)
```

---

## 7. Steps

### Step 0. Setup and framing

Kind: Diagnostic

Objective: establish the problem, the guard rails and the working environment before touching the data.

Tasks:
1. Write down the answers to these framing questions, from the human where given and marked "assumed" otherwise.
   - What decision does this analysis or model support?
   - What is one row (the unit of analysis), and can the same entity appear in many rows?
   - How is the target defined, and when does its value become known?
   - Which metric matters, and what does each error cost?
   - Is there a time order, and will the model score future data?
   - Which columns are known at prediction time?
   - What are the constraints?
2. Set aside a provisional final holdout of about 15 to 20 percent of the rows. Choose the scheme by these rules: time-ordered (latest rows held out) if there is a date column and the model will score future data, grouped by entity if entities repeat across rows, otherwise random (stratified for classification). This holdout is not used for any analysis until the modeling phase. Step 15 confirms or revises the scheme.
3. Everything target-aware from here on uses the development set only. Structural checks that ignore the target may use all rows.
4. Create the folder layout, fix the random seed, and record library versions.

Log: the framing answers, the assumptions, the holdout scheme and size.

---

### Step 1. Load and first look

Kind: Diagnostic

Objective: know the size, shape and obvious problems before any analysis.

| Option | What to do | What to record |
|---|---|---|
| A. Manual inspection | Row and column counts, memory use, dtypes, first rows, last rows, a random sample of rows, summary statistics for all columns | Anything odd in the raw rows (headers repeated inside data, footer rows, encoding damage) |
| B. Per-column overview | One table with dtype, missing share, number of unique values and an example value for every column | Constant columns, near-unique columns, dtype and meaning mismatches |
| C. Automated profile | Generate a profiling report in minimal mode and read its alerts | Alerts the other two options did not surface |

Decision rule: combine all three into one prioritized issue list. Each item names the column, the problem and the step that will handle it.

Escalate if: the row count is far from what the human expects, or the file is clearly not the dataset described.

---

### Step 2. Types, validity and consistency

Kind: Diagnostic followed by fixes

Objective: make sure every column means what its name suggests and contains only possible values.

| Option | What to do | What to record |
|---|---|---|
| A. Dtype vs meaning | Find numbers stored as text, dates stored as text, and low-cardinality numeric codes that are really categories | Columns and the conversion needed |
| B. Sentinel and placeholder values | Look for values such as -1, 999, 9999, 99999, "?", "N/A", "null", "None" and empty strings acting as missing | Column, value, count, share |
| C. Range and domain rules | Check minimums and maximums against common sense and any stated business rules (negative ages, percentages above 100, dates in the future) | Column, rule, violation count |
| D. Cross-field consistency | Check relations between columns (end before start, parts larger than total, mismatched country and currency) | Rule and violation count |

Decision rule:
- Apply unambiguous fixes directly (cast numeric-as-text, parse dates, convert obvious placeholders to missing).
- For impossible or suspicious values, convert them to missing and add a flag column recording that this happened. Never delete rows here.
- Any sentinel you are unsure about (a real 0 versus a placeholder 0) goes on the human question list.

Escalate if: more than about 5 percent of a column is invalid under stated rules.

---

### Step 3. Target analysis

Kind: Diagnostic

Objective: understand what is being predicted before studying anything else.

| Option | What to do | What to record |
|---|---|---|
| A. Distribution | Classification: class counts and shares. Regression: summary statistics, skew, tails, a log-scale view, a normality view | Imbalance ratio, skew, spikes, impossible values |
| B. Absolute minority count | Classification only: the number of rows in the rarest class in the development set | The count itself, not only the share |
| C. Stability | Target rate or mean by month, and by the largest segments if any | Trends, breaks, segment differences |
| D. Missing or constant target | Count rows with no target value | Count and share |

Decision rule:
- Remove rows with no target from the modeling set, and log the count.
- Flag class imbalance if the minority share is under 10 percent, and treat it as severe under 1 percent. Step 14 will handle it.
- For a regression target with skew above 1 in absolute value, mark it as a candidate for a target transform, to be tested in Step 8.
- If the target rate drifts over time, note that it affects Steps 12 and 15.

Escalate if: the minority class has fewer than about 100 development rows, or the target definition appears to change over time.

---

### Step 4. Duplicates

Kind: Selection (how to treat them)

Objective: find repeated rows and decide whether the repetition is an error or real.

Detection options, all to be run:

| Option | Definition of duplicate | What to record |
|---|---|---|
| A. Exact | Every column identical | Count, share, whether they cluster in time or position |
| B. Key-based | Same identifier, or same identifier plus date | Count, and whether the repeated rows differ in a meaningful way |
| C. Near duplicates | Identical after lowercasing, trimming spaces and rounding numbers to a sensible precision | Count beyond the exact ones |
| D. Conflicting labels | Identical features but different targets | Number of feature combinations with conflicting targets and rows involved |

Treatment options to consider: keep everything, drop exact duplicates keeping the first, keep the latest row per key, aggregate rows per key, drop or relabel conflicting rows after review.

Decision rule:
- Exact duplicates with no business reason for repetition: drop, keeping the first.
- Key duplicates that are legitimate repeated observations (different timestamps or values): keep them, and record that a group-aware split may be needed (Step 15).
- Key duplicates that are stale versions of the same record: keep the latest.
- Near duplicates: drop only if inspection of a sample shows they are the same record.
- Conflicting labels: quantify them, because they cap the achievable accuracy. Do not delete them without human approval.
- Check for duplicates that fall on both sides of the development and holdout boundary. Move them to the same side, and log it.

Log also: the target rate before and after de-duplication.

Escalate if: more than about 2 percent of rows have conflicting labels.

---

### Step 5. Missing values

Kind: Diagnostic (5a) then Selection (5b)

Objective: work out why values are missing, then pick the handling that holds up in validation.

#### 5a. Diagnose

| Option | What to do | What to record |
|---|---|---|
| A. Counts | Missing share per column, sorted | Columns above 5, 30, 50 and 80 percent |
| B. Co-missingness | Missingness matrix and correlation of missingness between columns | Groups of columns that go missing together |
| C. Relation to target and others | Compare the target rate (or mean) between rows with and without a value, and check whether missingness relates to other columns or to time | Effect size and direction |

Classify each column with missing values as one of: looks random (MCAR-like), depends on other observed columns (MAR-like), or depends on the value itself or on the target (MNAR-like, informative). State the evidence.

#### 5b. Treat, compared on results

Options to run for numeric columns:
- Drop columns above a missing threshold (try thresholds of 30, 50 and 80 percent and compare with keeping them)
- Drop rows with any missing value
- Fill with mean
- Fill with median
- Fill with median plus a missing-indicator column
- Fill with a constant sentinel plus a missing-indicator column
- Fill with the median within a meaningful segment
- Nearest-neighbor imputation
- Iterative (model-based) imputation
- Leave missing and let the gradient boosting model handle it natively

Options for categorical columns: fill with the most frequent level, fill with an explicit "missing" level.

Comparison protocol: for each option, build the full pipeline, fit the imputation inside each fold, and score with both model families under the shared cross-validation. Also compare each imputed column's distribution with the original observed distribution, since an option that wins on score but flattens a distribution deserves a note.

Decision rule:
- Choose separately per column group and per model family if the evidence differs.
- If missingness is informative (target rate differs materially between missing and present rows), prefer an option that preserves an indicator.
- The drop-rows option changes the sample, so its score is not directly comparable. Report it but exclude it from the ranking. Use it only when missingness is very low (under about 1 to 2 percent) and looks random.
- Apply the tie-break of simplicity.

Log: the diagnosis for each affected column, the full comparison table, and the chosen option per column group.

---

### Step 6. Univariate analysis

Kind: Diagnostic

Objective: know the shape of every variable on its own.

Numeric columns:

| Option | What to do | What to record |
|---|---|---|
| A. Summary statistics | Mean, standard deviation, minimum, key percentiles (1, 5, 25, 50, 75, 95, 99), maximum, skew, kurtosis, share of zeros | Heavy tails, spikes, zero inflation |
| B. Plots | Histogram with density, box plot, empirical cumulative distribution | Multiple modes, gaps, truncation |
| C. Normality checks | Quantile-quantile plot, a normality test, and a Shapiro test on a sample | Departures from normal (weigh the plot more than the p-value, since large samples reject everything) |

Categorical columns:

| Option | What to do | What to record |
|---|---|---|
| A. Level counts | Frequencies including missing | Dominant levels, rare levels |
| B. Cardinality and rare levels | Number of unique levels, share of the top level, number of levels under 1 percent | Encoding difficulty |
| C. Consistency | Compare the unique count before and after lowercasing and trimming spaces, and look for near-identical spellings | Levels that should be merged |

Decision rule (actions to list):
- Merge inconsistent spellings directly.
- Mark columns with skew above 1 in absolute value for testing in Step 8.
- Mark columns with a large spike at zero as candidates for an "is zero" flag.
- Drop constant columns, and near-constant columns (over 99.9 percent one value) only after checking that the rare value is unrelated to the target.
- Note ID-like columns (nearly unique per row) for exclusion from features.

---

### Step 7. Outliers

Kind: Diagnostic (7a) then Selection (7b)

Objective: separate errors from real extremes, and decide what to do about each.

#### 7a. Detect

| Option | Definition |
|---|---|
| A. IQR fences | Outside 1.5 times the interquartile range beyond the quartiles |
| B. Z-score | More than 3 standard deviations from the mean |
| C. Modified z-score | More than 3.5 by the median absolute deviation method (skip if that deviation is zero) |
| D. Percentile cut | Below the 1st or above the 99th percentile |
| E. Isolation Forest | Multivariate anomaly score |
| F. Local Outlier Factor | Multivariate local density |

Record: the share flagged by each method per column, how much the methods agree, and for the multivariate methods the number flagged and overlap. Then inspect a sample of flagged rows and label each pattern as an error (impossible or unit mix-up), a real extreme, or unknown. Also check whether flagged rows have a different target rate from the rest, because outliers can be the signal, as in fraud or failure data.

#### 7b. Treat, compared on results

Options to run:
- Keep as is
- Cap at the 1st and 99th percentiles
- Cap at the 5th and 95th percentiles
- Log or power transform (see Step 8)
- Remove flagged rows (from training folds only, scored on untouched validation folds)
- Keep and add an outlier flag feature
- Model the extreme segment separately (consider only if flagged rows form a coherent population)

Fit any cap thresholds inside each fold. Score both model families under the shared cross-validation.

Decision rule:
- Errors are corrected to missing (and handled by Step 5), not kept.
- Real extremes: keep for tree models. For linear models, choose among keep, cap and transform by score.
- Never remove rows because of their target value, and never remove more than about 5 percent of rows without asking.
- If flagged rows have a markedly different target rate, prefer keeping them, plus a flag feature, and raise it to the human.

---

### Step 8. Transformations and scaling

Kind: Selection

Objective: reduce harmful skew and put features on comparable scales, only where the model family cares.

Transform options, applied to columns with skew above 1 in absolute value, respecting each option's data requirements:

| Option | Requirement |
|---|---|
| None | None |
| Log of (1 plus x) | Values at or above 0 |
| Square root | Values at or above 0 |
| Box-Cox | Strictly positive values |
| Yeo-Johnson | Any values |
| Quantile transform to a normal shape | Any values |

Scaling options: none, standardization, min-max, robust scaling (median and interquartile range).

For a regression target, also test transforming the target (none, log, square root, Box-Cox or Yeo-Johnson), always scoring back on the original scale.

Record: mean absolute skew before and after for each option, and the cross-validated score for both model families. Confirm as a sanity check that the tree model is insensitive to monotonic transforms and scaling, and investigate if it is not.

Decision rule: choose per model family. Tree models get no transform unless it is clearly beneficial. For linear models, pick the transform and scaler with the best score, respecting the noise threshold and the tie-break. Prefer robust scaling if outliers were kept in Step 7.

---

### Step 9. Features vs target

Kind: Diagnostic

Objective: find which features carry signal and what shape the relationship takes.

Numeric features:

| Option | Captures | What to record |
|---|---|---|
| A. Pearson correlation with target | Linear relation | Coefficient |
| B. Spearman correlation with target | Monotonic relation | Coefficient |
| C. Mutual information with target | Any dependency, including non-linear | Score and rank |
| D. Single-feature AUC and Kolmogorov-Smirnov statistic (classification) | Class separation | Values |
| E. Target rate or mean by decile of the feature | Shape of the effect, including U shapes | The pattern for the top features |

Categorical features:

| Option | What to record |
|---|---|
| A. Target rate per level with level counts | Effect per level, and levels too small to trust |
| B. Chi-square test and Cramer's V | Association strength, comparable across columns |
| C. Weight of evidence and information value (binary target) | Predictive strength ranking |
| D. ANOVA or Kruskal-Wallis (regression target) | Whether level means differ |

Decision rule (actions to list):
- Rank all features by strength using more than one measure, and note disagreements. A feature with high mutual information and near-zero correlation has a non-linear effect that a linear model will miss.
- Mark features with essentially no signal as candidates to drop, but do not drop them yet. Signal can appear in interactions.
- Send anything suspiciously strong to Step 12. Suspicious means single-feature AUC above 0.9 (or below 0.1), information value above 0.5, or correlation above 0.9 in absolute value.

---

### Step 10. Features vs features

Kind: Diagnostic (findings) then Selection (what to do about redundancy)

Objective: find redundancy and groups of related columns.

| Option | What to do | What to record |
|---|---|---|
| A. Correlation matrices | Pearson and Spearman among numeric features | All pairs above 0.9 in absolute value |
| B. Categorical association | Cramer's V among categorical features | Pairs above about 0.7 |
| C. Variance inflation factor | Compute per numeric feature | Values above 5 (watch) and above 10 (problem) |
| D. Hierarchical clustering on correlation distance | Group features that move together | The feature groups |

Treatment test: for each redundant group, score the model with all members versus one representative, for both model families, under the shared cross-validation.

Decision rule: drop the extras only if the score is not worse by more than the noise threshold. Choose the representative that is least missing, most available at prediction time and easiest to interpret. Keep all members for tree models if dropping them hurts, since redundancy matters much more for linear models.

---

### Step 11. Multivariate structure

Kind: Diagnostic

Objective: see whether the data has low-dimensional structure, separable classes or hidden subpopulations.

| Option | What to do | What to record |
|---|---|---|
| A. PCA | Explained variance per component, cumulative curve, top loadings of the first components | Number of components for 90 percent of variance, which features drive them |
| B. t-SNE or UMAP | Two-dimensional embedding on a sample, colored by the target | Separable classes, islands, mixed clouds |
| C. Clustering | KMeans (and optionally a Gaussian mixture) for several cluster counts with silhouette scores | Best cluster count and its silhouette |

Interpretation rules: distances between islands in a t-SNE or UMAP view are not evidence, so treat those views as a search for structure and not as proof. A silhouette below about 0.25 means weak clusters. If clusters are clear and the target rate differs by cluster, propose a segment feature or a per-segment model to the human.

---

### Step 12. Time, drift and leakage

Kind: Diagnostic

Objective: make sure what you see today holds in production, and that no feature is cheating.

| Option | What to do | What to record |
|---|---|---|
| A. Trends and seasonality | Row volume, target rate and missing share by period | Breaks, seasonality, level shifts |
| B. Population Stability Index per feature | Compare an early period with a late period (for example the first 70 percent by time against the last 30 percent) | Values below 0.1 are stable, 0.1 to 0.25 worth watching, above 0.25 shifted |
| C. Adversarial validation | Train a model to distinguish early rows from late rows, score it with cross-validation, and note the features that drive it | An AUC near 0.5 means no detectable shift, above about 0.7 is notable |
| D. Leakage screen | Review every feature flagged in Step 9 as too strong, every feature with near-perfect target correlation, ID-like and ordering-related columns, and every top feature for availability at prediction time | A verdict per suspect |

If there is no date column, skip A to C and record that. Still do D, and also check whether row order or identifier values relate to the target.

Decision rule:
- Any feature that would not exist at prediction time, or that is derived from the outcome, is excluded, and the exclusion is logged.
- A feature with near-perfect separation that you cannot explain is excluded provisionally, and the human is asked.
- Features with large drift are listed. Do not drop them automatically. Note the option of dropping or re-deriving them, and test the effect under a time-based split in Step 15.

Escalate if: any suspected leakage exists.

---

### Step 13. Categorical encoding

Kind: Selection

Objective: pick the encoding that suits each cardinality band and each model family.

| Option | Suits | Risk |
|---|---|---|
| One-hot with rare levels grouped (try grouping under 1 percent) | Low cardinality, linear models | Width explodes for high cardinality |
| Ordinal | Truly ordered levels, tree models | Invents an order otherwise |
| Frequency encoding | High cardinality when frequency is informative | Distinct levels with equal counts collide |
| Target encoding, cross-fitted | High cardinality | Leaks if not fit inside folds and cross-fitted |
| Drop the column | Useless or leaky columns | Loses signal |

Comparison protocol: fit all encoders inside folds, score both model families under the shared cross-validation, and report the number of resulting columns. Read ordinal results for linear models with suspicion when the levels are not ordered.

Decision rule: decide per column, grouped by cardinality (under about 15 levels, 15 to 50, over 50) and per model family. Prefer the simpler encoding when tied. Do not use target encoding unless it beats the alternatives by more than the noise threshold, since it carries the most leakage risk.

---

### Step 14. Imbalance handling

Kind: Selection (classification only)

Objective: decide how to handle a rare positive class, judged by a metric that respects rarity. Skip if the earlier steps showed no imbalance issue, and record that.

Judge with PR-AUC (average precision). ROC-AUC looks flattering under heavy imbalance, so report it alongside but do not rank by it.

Options to run:
- No handling
- Class weights
- Random undersampling of the majority
- Random oversampling of the minority
- SMOTE (synthetic minority oversampling; note it handles categorical columns poorly)
- Threshold tuning on out-of-fold predicted probabilities (combinable with any option above)

Resampling is applied inside training folds only. Validation folds are never resampled.

Decision rule:
- Pick the simplest option that improves PR-AUC by more than the noise threshold.
- If none does, choose "no handling plus a tuned threshold".
- Choose the threshold from the cost of false positives versus false negatives given in the framing. If no costs were given, report the threshold at best F1 and at a few precision levels and let the human choose.
- Note the effect on probability calibration if probabilities will be used directly.

---

### Step 15. Split strategy

Kind: Selection

Objective: choose a validation scheme that mimics how the model will actually be used, and confirm the final holdout.

Options to run under the same model and features:
- Random K-fold
- Stratified K-fold (classification)
- Group K-fold on the entity identifier (if entities repeat)
- Time-based splits (if there is a date column)

Decision rule: a large drop from random to group or time-based scores means the random scheme leaks, and the lower number is the honest one. Choose the scheme that matches deployment: time-based when the model scores future data, group-based when new entities appear at prediction time, otherwise stratified random.

If the chosen scheme differs from the provisional holdout in Step 0, rebuild the holdout using the chosen scheme. Then re-verify the winning option of every earlier Selection step (5, 7, 8, 10, 13, 14) under the new scheme. If a ranking changes, treat it as a new decision and log it.

---

### Step 16. Baseline and sanity check

Kind: Diagnostic

Objective: confirm the story from the EDA with the simplest models before anything complicated is built.

| Option | Purpose |
|---|---|
| A. Dummy baseline (prior or mean) | The floor any model must beat |
| B. Regularized linear model on the preprocessed data | How far simple relationships go |
| C. Gradient boosting at defaults | Headroom from non-linearity and interactions |
| D. Permutation importance on a validation split inside the development set | Whether important features match what Steps 9 and 12 predicted |

Use cross-validation with the chosen scheme, and report mean and standard deviation. Do not touch the final holdout.

Interpretation rules:
- If one feature dominates importance, go back to Step 12.
- If the linear model matches the boosted model, the signal is mostly simple.
- If nothing beats the dummy baseline, revisit Steps 2, 4 and 9 before concluding anything about algorithms.

---

### Step 17. Final summary

Kind: Diagnostic

Write `eda_summary.md` so that someone who never read the log can act on it. It contains:

1. A one-paragraph description of the dataset and the task.
2. A decision table with one row per step: the options tried, the winner, the key numbers, and the confidence.
3. The top data quality issues, each with a suggested owner and fix.
4. The list of features excluded (leakage, redundancy, no signal, unavailable at prediction time), each with the reason.
5. The ordered preprocessing specification (or a pointer to `preprocessing_spec.md`), with variants per model family.
6. The split scheme and how the holdout is defined.
7. The baseline results and what they suggest about headroom.
8. Assumptions still needing confirmation.
9. Everything logged with Low confidence, as the "needs human review" list.
10. Recommended next steps for the modeling phase.

Finish by checking three things: the analysis reruns from the raw data, the holdout was never touched, and every dropped row or column appears in the log with its count.
