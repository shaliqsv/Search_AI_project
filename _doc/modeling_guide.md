# Modeling Guide for the Agent: OTTO Search and Ranking (Level 0)

This guide covers the modeling work that ends with a frozen model bundle, which is the Level 0 outcome in the project plan. You write and run all code yourself. This guide has none.

The project plan (`plan.md`) is the source of truth. This guide refers to its decisions by ID (D4, D12, and so on) and does not repeat them. If this guide and the plan disagree, the plan wins. Do not change an Accepted decision without the human's approval. Provisional decisions are yours to settle with evidence. Record every new or changed decision in `docs/decisions/` in the plan's format: decision, why, alternatives considered, status.

Each step is Diagnostic (run all the options, combine the findings) or Selection (run all the options under identical conditions and pick by the stated rule). Run every listed option before choosing. If you skip one, log why.

Out of scope here: Airflow, MLflow automation, CI/CD, Terraform, canary deployment and hand-deploying the container. Those belong to the Level 0 deployment, Level 1 and Level 2 work.

## Inputs

- `plan.md`
- The session sample in parquet and any EDA outputs already produced
- AWS region choice and Bedrock access
- The human, for hand-labeling the judge calibration set (Step 9)

## Rules

1. Time and leakage. Split by time only. Anything learned from data (categories, co-visitation counts, popularity statistics, item features, embeddings, calibration) uses the training window only. Features are computed point in time, as of the day before each session (D25). Weeks 3 and 4 are Level 1 arrivals, so modeling never touches them.
2. The ranking query is a documented leak (D5). Never present ranking results as evidence of real query understanding.
3. The test slice is used once, in Step 11, after everything is frozen. If anything changes after that, it is spent, and the summary must say so.
4. One yardstick. All methods are scored on the same sessions with the same metric code. The two rerankers use the same candidate pool (D8). Compare methods with a paired bootstrap over sessions. If the 95 percent interval of the difference includes zero, it is a tie, and ties go to the simpler method. Complexity ladder: co-visitation, LightGBM, Two-Tower, DCN-V2 with MMoE.
5. Freeze before comparing. Gain values, the K values, the candidate pool and the benchmark set are fixed before methods are compared. Changing one afterward means rerunning every method.
6. Equal, bounded tuning. Give each model the same modest budget (same number of trials or same time), and record it. This is a learning project, so stop at clear diminishing returns.
7. Ablations are deliverables. Each one named in a step below is run and reported, not treated as an extra.
8. Suspect leakage when a result looks too good, for example a learned model beating co-visitation by a huge margin, or a metric near a ceiling. Stop and check for future information in the features or categories.
9. Cost. The whole project targets tens of dollars. Log every Bedrock call (calls, tokens, dollars), cache by the keys in D19, develop on small subsets, and confirm the budget alarm exists.
10. Reproducibility. Seed 42, data snapshot ID and code commit SHA on every run. Training runs as scripts with a config file so they work on the Mac and on Colab, with checkpoints saved to S3 (D21). Everything reruns from one command.
11. Manual-steps log. From day one, write down every manual step and how long it took. This is Level 0 evidence for the MLOps tab.
12. Parked, so not built: the LLM ranker (D14), SASRec, cross-session personalization, price signals. If one seems needed, ask.

## When to stop and ask

Stop and ask the human, with evidence, if:
- You suspect leakage.
- A component on the live path (Two-Tower, DCN-V2 with MMoE) loses to a simpler method beyond noise. Report it. Do not swap components silently.
- CPU training exceeds about an hour for a neural model (a spot-training decision, D21).
- The Bedrock model is unavailable in the chosen region, or costs threaten the budget.
- The OTTO license terms are unclear about publishing derived data.
- Judge calibration labels are needed, or the results suggest revisiting D15 (whether the main pipeline should train on simulated exposure).
- Anything requires changing an Accepted decision.

Otherwise decide, log it, and give a confidence label. High means the winner beat the alternatives beyond noise and agrees across metrics or slices. Medium means clear on one metric or by a modest margin. Low means a tie or conflicting evidence, decided by the ladder. Collect all Low items in a "needs review" list.

## Deliverables

- `model_log.md`, with one entry per step
- `experiments.csv`: id, step, description, features, model, parameters, seeds, metrics with intervals, fit and prediction time, size, notes
- `docs/decisions/`
- `tables/` and `figures/`
- The frozen bundle in `artifacts/`
- The manual-steps log
- `model_summary.md`
- Code in the repository layout from the plan (Section 8)

Log entry format for each step: step and kind, candidates run (with numbers, experiment ids and reasons for skips), observed, decision, why, rejected options and why, confidence, effects on later steps, needs human input.

---

## Steps

### Step 0. Setup and checks (Diagnostic)

- Read the plan in full. List the Provisional decisions this phase settles: D4, D8, D15, D18 and D21. D25's shared feature implementation also starts now.
- Check the sample: session and event counts, event-type mix, sessions with carts and orders, session length and item frequency distributions, week-over-week shifts, duplicates. State how many cart and order positives each head needs for stable training and evaluation. If the sample has too few, oversample sessions containing them and record the reweighting (D3).
- Confirm the OTTO license terms before anything derived leaves private storage.
- Confirm the region and Bedrock model availability (inference profiles if required). Put model IDs in configuration.
- Build a hello-world arm64 image that imports faiss, onnxruntime and lightgbm. If a wheel is missing, use conda-forge or a source build, and log it.
- Create the experiment table, the Bedrock cost log and the manual-steps log.

### Step 1. Splits and evaluation harness (Selection)

- Split by time within weeks 1 and 2: a training window, a validation slice, and a test slice on the days right after. Design the rule so it can be applied unchanged to each Level 1 arrival later. Propose exact day boundaries and a rule for sessions that cross a boundary, and record them as a Provisional decision (D3).
- Build the harness before any model: NDCG@10 with graded gains, recall@K at 20, 50, 100 and 200, OTTO weighted recall@20 (0.10, 0.30, 0.60), bootstrap 95 percent intervals over sessions, paired bootstrap for differences, and slices by category, session length and item popularity bucket. Verify it on toy cases against scikit-learn or ranx (D7).
- Gains: score the baselines under the provisional gains (click 1, cart 2, order 3) and at least two alternatives. Confirm the method ranking is stable, then freeze one set.

### Step 2. Categories, queries and the benchmark set (Selection)

- Build the co-visitation graph from the training window only, embed with truncated SVD, and cluster with k-means at K from about 50 to 100. Choose K by stability across seeds and size balance. Compare quickly against Leiden community detection. Name clusters from the fictional taxonomy, aligned by hierarchical clustering order (D4). Label the categories as synthetic everywhere.
- Ranking queries: the category of the session's held-out next click plus the session prefix. Write down the leak (D5).
- Classifier evaluation set: paraphrased, misspelled, multi-intent and out-of-scope terms per category. Generate it with a different model or prompt style than the classifier under test, and hand-check a sample. Split it into a dev part (for calibration) and a held-out part.
- Benchmark set (D9): about 300 (category query, session prefix) pairs from the test slice, stratified by category and session length. Version and freeze it.

### Step 3. Baselines (Selection)

- Co-visitation (D10): item-to-item counts with event-type weights and time decay, producing candidates and a score. Tune the weights and decay on validation within the standard budget. This is the bar the learned models must clear.
- BM25 (D10): implement from scratch and check against a library on toy cases. Documents are the category name tokens plus the names of the item's most co-visited categories. Tune lightly. Expect behavior close to set membership with ties, and report that as a finding.
- Record recall@K and NDCG@10 for both. The real-text BM25 side lab is optional, time-boxed and outside the product.

### Step 4. Two-Tower with FAISS (Selection)

- Query tower (D12, D6): the classifier's top-3 category distribution as a confidence-weighted mix of category embeddings, plus a session encoder that pools item embeddings weighted by recency and event type. Item tower: item embedding, category and popularity buckets.
- Decide how to feed the category distribution in training: the true category, or the true category with simulated classifier confusion. Log the choice. The real classifier enters in Step 10.
- Restrict the catalog to items above a minimum frequency, chosen from the recall and coverage tradeoff.
- Train with sampled softmax, in-batch negatives and logQ correction. Ablation: add same-category hard negatives.
- ANN: use exact search as ground truth, then test IVF-PQ and HNSW at several settings. Choose by measured recall against exact search, latency, and index size relative to the Lambda image budget.
- Size the model so a full CPU training run in the training image takes about an hour, and measure it early (D21).

### Step 5. Candidate pool (Selection)

- Options: Two-Tower only, co-visitation only, and their union, each at several pool sizes.
- Record recall@K (overall and by event type), latency and pool size.
- Choose the smallest pool whose recall is within noise of the best, then freeze it (D8). Both rerankers use it.

### Step 6. LightGBM LambdaMART (Selection)

- Features (D11): category confidence, co-visitation scores against session items (max and mean), item popularity (global, within category, recent), item click, cart and order rates, recency, session length, and Two-Tower similarity. All are computed point in time in the shared feature implementation used later for serving (D25).
- Train with the lambdarank objective on graded labels. Ablations: the binary-logloss version, and with versus without Two-Tower similarity.
- Add a propensity-weight column that defaults to 1 (D15), and confirm that all-ones weights reproduce the unweighted results exactly.
- Tune within the standard budget.

### Step 7. DCN-V2 with MMoE (Selection)

- Build DCN-V2 cross layers and an MMoE with click, cart and order towers (D13). The loss is a weighted sum of per-task binary cross-entropy. Include the same propensity-weight column defaulting to 1.
- Options to compare: the number of cross layers, the number of experts, task loss weights, and the final blend weights (OTTO's 0.10, 0.30, 0.60 by default, tuned on validation).
- Ablations: single-task models, and a shared-bottom multi-task model as the baseline MMoE is meant to beat. Watch for seesaw effects, since orders are rare. If they appear, note PLE as the follow-up and do not build it.
- Measure CPU training time against the one-hour target.

### Step 8. Position-bias study (Selection)

- Build the semi-synthetic simulation (D15): a logging policy (popularity plus noise, or co-visitation based), ranked impressions, and clicks from a position-based examination model with known propensities, using true relevance from held-out behavior.
- Compare a naive model against an IPW model, with clipping and self-normalization, judged against the known ground truth.
- Sweep the strength of the bias, the clipping thresholds and several seeds. Plot bias against variance.
- Real OTTO labels stay unweighted in the main models. After the results, ask the human whether the main pipeline should train on simulated exposure.

### Step 9. Claude roles (Selection)

Use Bedrock for every call (D16). Pick the smallest model that passes each role's evaluation, and fix a pass criterion in advance.

Classifier (D17):
- Return the top 3 categories with confidences and an out-of-scope flag. Evaluate on the held-out set for top-1 and top-3 accuracy, expected calibration error, and out-of-scope precision and recall.
- Calibrate confidences on the dev part of the query set.
- Baselines: sentence-embedding nearest category, and TF-IDF. Keep the LLM only if it beats them by the pre-stated margin, given its cost and latency.
- Decide what happens below the confidence threshold: blend the top categories, or ask the user to refine.
- Cache by normalized query.

Explainer (D18):
- Give Claude only the numeric payload per result and require it to cite only what the payload contains.
- Build the faithfulness test, which parses numbers and feature names from the output and checks them against the payload. Record fixtures for CI, and prepare a scheduled live run.
- Build a templated explanation as the fallback and as a baseline.
- Attribution for DCN-V2: compare leave-one-feature-group-out deltas against integrated gradients, on sanity checks (do removals move the score as claimed), agreement, stability and latency for the top few results. Choose one.

Judge (D19):
- Score each method's top 10 from 0 to 4 with method identity hidden and method order randomized. The judge must not see method scores or co-visitation counts. One call per (pair, method) returns ten scores, about 1,500 calls in total. Cache by pair, method, model version and prompt version.
- Prepare a labeling sheet of 150 to 200 pairs for the human. Compute weighted kappa and Spearman against the judge, and test for position and verbosity effects.
- Write down that the judge mainly measures category coherence with the session, not true product relevance.

### Step 10. Comparison and analysis (Diagnostic)

- Score all five methods: NDCG@10 with intervals on the full held-out window, recall@K for the retrieval-style methods, and judge scores on the benchmark set.
- Slice by category, session length and item popularity bucket. Look for popularity bias in particular.
- Tabulate all ablations from Steps 4 to 8.
- Run the live path end to end with the real classifier in the loop on a subset (D5), and quantify how classifier errors propagate.
- Write the disagreement analysis (D19): the cases where NDCG and the judge diverge, grouped by cause, including artifacts of the synthetic categories.
- Measure the live path's latency, model sizes and Bedrock cost per query.
- Report honestly. If co-visitation or LightGBM beats a learned method, say so, and apply the stop-and-ask rule for live-path components.

### Step 11. Freeze and Level 0 exit (Diagnostic)

- Write down the expected test-slice range from validation, then score the frozen models on the test slice once and report the results with intervals.
- Build the bundle: models, ONNX exports of the neural models with output parity checked against the training framework, the FAISS index, item feature tables, the category map, prompts and configuration, and a manifest (model versions, feature schema hash, data snapshot ID, code commit SHA, metrics) (D23, D26).
- Run a feature parity test: the training path and the serving path must produce identical features on a fixture (D25).
- Upload the bundle to S3 by hand and time it. Finish the manual-steps list with timings.
- Produce the versioned exports the frontend will read: metrics, per-query results and judge scores (D33).

Hand-deploying the container is the next piece of Level 0 work, outside this guide.

### Step 12. Summary (Diagnostic)

Write `model_summary.md`: a paragraph on the approach, a decision table (step, candidates, winner, key numbers, confidence), the five-method table, the ablations, the position-bias findings, the Claude role evaluations, and the limits stated plainly (synthetic categories, the D5 leak, the judge measuring coherence). Add the cost spent, the Provisional decisions settled with links to their records, the needs-review list, and anything Level 1 needs to know.

Final check: the test slice was used once (or the summary says it was spent), everything reruns from one command, every experiment is in the table, and the manual-steps log is complete.