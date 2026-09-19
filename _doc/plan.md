# Search and Ranking on OTTO: Project Plan

Version 0.1, dated September 19, 2026. This is a living document. Every decision has a status (Accepted or Provisional), the reasoning behind it, and the alternatives that were considered, so later changes can be recorded in the same format. Prices quoted are rough estimates from memory and should be checked against current AWS pricing before you rely on them.

## 1. Purpose and constraints

The project has two goals, in this order. The first is to understand search and ranking properly by building the whole stack once: candidate retrieval, learning to rank, multi-task ranking, evaluation, position-bias correction, and LLM-based query understanding, explanation and judging. The second is a portfolio piece for senior search and ranking data scientist roles. It shows one frozen model set moving through three MLOps maturity levels (manual, automated pipeline, CI/CD), and a frontend shows both how well the ranking works and how it was deployed.

| Constraint | Value | What it forces |
|---|---|---|
| Cost | Minimise. Target: tens of dollars for the whole build | No always-on managed services, no GPU hosting, no idle endpoints |
| Compute | 8GB M3 Mac, Colab as backup | Sampled data, small models, out-of-core tooling |
| Time | 10 to 15 hours a week, about 28 weeks (280 to 420 hours) | A frozen model set, a cut order, a mid-project checkpoint |
| Skills | Comfortable with Docker, Airflow and infrastructure as code | Levels 1 and 2 are about doing it well, not learning the basics |
| Delivery | Docker-first, deployed to AWS | The image is the unit of shipping. AWS is the deployment target |

## 2. What gets built

The live search path is:

```
raw search term
  -> Claude classifier (Bedrock): top categories with confidence, out-of-scope flag
  -> Two-Tower query tower (category distribution + session encoder)
  -> FAISS ANN search over item embeddings: candidate pool
  -> DCN-V2 + MMoE reranker: click, cart and order probabilities
  -> blended score, top 10
  -> Claude explainer (Bedrock): short text grounded in the numeric scores
```

Alongside it, a methods comparison tab runs the same query and session through five methods: BM25 (a floor), co-visitation (the strong non-learned baseline), LightGBM LambdaMART, Two-Tower, and DCN-V2 + MMoE. Each method gets two numbers: NDCG@10 against OTTO's held-out behavior, and a Claude judge score from 0 to 4. The interesting cases are where the two disagree, and the write-up analyses those cases explicitly.

An architecture tab explains each platform choice and lists what was deliberately left out, with reasons and rough prices. A position-bias study (D15) runs as a controlled simulation. The MLOps tab shows what each maturity level measurably changed.

The three phases are MLOps maturity levels applied to the same frozen models:

| Level | What it contains | Exit criteria |
|---|---|---|
| 0, manual | Notebooks and scripts, hand-run steps, artifacts uploaded to S3 by hand, container deployed by hand once | Frozen model bundle in S3. Written list of every manual step with timings. Container deployed by hand |
| 1, pipeline | Airflow DAG: partition arrives, validate, build features, train, evaluate, gate, register, deploy. Scheduled and data-triggered | Two consecutive weekly arrivals give one promotion and one rejection, both explained by logged metrics. The same commit reproduces the same metrics |
| 2, CI/CD | GitHub Actions tests and builds on every push, infrastructure as code, canary deployment with alarm rollback, retraining on code or drift triggers | Failure injection works both ways (bad model rejected, degraded deployment rolled back). A feature-definition change reaches production with no manual step |

Measured differences between levels, recorded as you go and shown in the MLOps tab: number of manual steps, time to retrain, time to deploy, time to roll back, reproducibility (same commit, same metrics), and automated test count.

## 3. Data reality and its consequences

OTTO is a session-based e-commerce dataset. As publicly described, it has roughly 12.9M sessions, 220M events and 1.8M items over four labeled weeks. Each event is an item ID (aid), a timestamp and a type (click, cart, order). The data lacks several things the original brief assumed, and most scoping choices in this document follow from that.

| What OTTO lacks | Consequence | How the plan handles it |
|---|---|---|
| Item text (titles, descriptions) | BM25 has almost nothing to index. The classifier and the judge have no real content. Text-based methods cannot show their strengths | Synthetic category names (D4), BM25 kept only as a floor (D10), stated on the architecture tab |
| Real queries | No query text and no query-item relevance labels | Queries constructed from categories (D5). Classifier evaluated on generated queries |
| Displayed positions or impressions | Position bias cannot be estimated from the data | Controlled simulation with known propensities (D15) |
| Cross-session user IDs | Personalization can only use the current session | Session-only personalization (D6) |
| Prices or marketplace signals | No price, seller or availability features | Left out (Section 7) |
| Ground-truth categories | Categories are derived from the same co-occurrence signal the models learn from, which risks circular results | Built from the training window only, leaks documented (D4, D5) |

## 4. Deviations from the original brief

| Original brief | This plan | Why (decision) |
|---|---|---|
| Three product phases | Three MLOps maturity levels | D1 |
| SageMaker-native build | Docker-first, AWS as deployment target | D20 |
| SageMaker Pipelines | Airflow in local Docker Compose | D22 |
| SageMaker Feature Store | Parquet offline, DynamoDB online, one shared feature implementation | D25 |
| SageMaker real-time endpoints with guardrails | Lambda container with CodeDeploy canary and alarm rollback | D26, D30 |
| LoRA LLM ranker on an LMI container | Dropped. Co-visitation takes its place in the comparison | D14 |
| IPW applied to training labels | Simulation study plus propensity-weight support in the pipeline, real OTTO labels trained unweighted | D15 |
| Claude via API | Claude via Amazon Bedrock | D16 |
| Model Registry gate | MLflow registry with a paired-bootstrap gate | D23, D27 |

## 5. Decisions at a glance

| ID | Decision | Status |
|---|---|---|
| D1 | Phases are MLOps maturity levels 0, 1, 2 | Accepted |
| D2 | OTTO as the dataset | Accepted |
| D3 | Session sample in parquet, DuckDB or Polars, Kaggle for sampling | Accepted |
| D4 | Synthetic categories from co-visitation, named from a fictional taxonomy | Provisional (week 2) |
| D5 | Two separate query constructs: ranking evaluation and classifier evaluation | Accepted |
| D6 | Session-only personalization | Accepted |
| D7 | NDCG@10, recall@K, OTTO weighted recall, bootstrap CIs, temporal split | Accepted |
| D8 | Five methods, shared candidate pool for the rerankers | Provisional (week 9) |
| D9 | Frozen benchmark set of about 300 pairs for judge and comparison tab | Accepted |
| D10 | Co-visitation and BM25 baselines | Accepted |
| D11 | LightGBM LambdaMART | Accepted |
| D12 | Two-Tower with FAISS | Accepted |
| D13 | DCN-V2 with MMoE | Accepted |
| D14 | LLM ranker dropped | Accepted |
| D15 | Position bias via simulation, IPW support in the pipeline | Provisional (week 14) |
| D16 | Claude through Bedrock, model per role chosen by evaluation | Accepted |
| D17 | Classifier with calibration, abstention and an embedding baseline | Accepted |
| D18 | Explainer with numeric payload and faithfulness test | Provisional (week 15) |
| D19 | Judge: batch, cached, calibrated against your labels | Accepted |
| D20 | Docker-first, arm64 end to end | Accepted |
| D21 | Training compute: Mac and Colab at Level 0, CPU-sized runs at Level 1 | Provisional (weeks 10 to 12) |
| D22 | Airflow in local Docker Compose | Accepted |
| D23 | MLflow for tracking and registry | Accepted |
| D24 | pandera for data validation | Accepted |
| D25 | Feature management: Parquet plus DynamoDB, one shared implementation | Provisional |
| D26 | Serving on Lambda (arm64 container) | Provisional (week 22) |
| D27 | Promotion gate with paired bootstrap | Accepted |
| D28 | GitHub Actions CI with OIDC to AWS | Accepted |
| D29 | Terraform for infrastructure as code | Provisional (week 22) |
| D30 | CodeDeploy canary with alarm-driven rollback | Accepted |
| D31 | Monitoring, drift and retraining triggers | Accepted |
| D32 | Live endpoint protection | Accepted |
| D33 | Static frontend with an optional live mode | Accepted |
| D34 | Cost controls and teardown | Accepted |

## 6. Decision records

### 6.A Framing and data

#### D1. Phases are MLOps maturity levels

**Decision.** The three phases are MLOps maturity levels 0 (manual), 1 (automated pipeline) and 2 (CI/CD), applied to one frozen set of models.

**Why.** The goal is understanding, and each level adds engineering rather than new modeling, so the model count does not multiply the work or the bill. The progression also gives a before-and-after story with measurable differences. You cannot appreciate or measure what automation fixes unless you have first done the work by hand and logged the pain.

**Alternatives considered.** Product phases (core ranking, then comparison tab, then extras), as in the original brief. Rejected because it front-loads breadth and leaves MLOps as an afterthought. Building straight to Level 2. Rejected because the manual pain is the evidence for why the automation exists. Stopping at Level 1. Rejected because you asked for all three levels.

#### D2. OTTO with synthetic categories

**Decision.** Use the OTTO session dataset. Categories and queries are synthetic (D4, D5).

**Why.** It has real behavior with clicks, carts and orders (needed for the multi-task ranker), enough scale for real retrieval and ranking work, four weeks of time structure (needed for the drift and retraining story), and a well-known benchmark with published solutions to compare against.

**Alternatives considered.** The following details are from memory and should be verified before relying on them. H&M has text, images and transactions but no clicks, sessions or queries. Amazon ESCI has real queries and graded relevance but no sessions, clicks or positions. Diginetica and the Coveo data challenge have queries and sessions, but with hashed tokens that no LLM can read. Instacart has product names but reorder-based behavior with no clicks. Amazon Reviews has text and ratings but no sessions. Stitching two datasets together was rejected because the join would be artificial and would add complexity without teaching more about ranking. The cost of choosing OTTO is the loss of real query text. That cost is accepted and documented in Section 3.

#### D3. Session sample, out-of-core tooling

**Decision.** Sample whole sessions (by hash of session ID, for reproducibility) into parquet, targeting 1 to 2 million sessions. Do the sampling in a Kaggle notebook with the dataset attached and download only the sample. Query with DuckDB or Polars. Use only the four labeled training weeks, since the public test week has hidden labels. Provisional layout: weeks 1 and 2 as the initial training window, week 3 as the first weekly arrival at Level 1, week 4 as the second, with evaluation always on the days right after a training window.

**Why.** 8GB of RAM cannot hold 220M events in pandas. Sampling by session keeps sessions intact, which matters because the models learn from session context. Doing the sampling on Kaggle avoids downloading the full dataset to the Mac. Check that the sample has enough cart and order events to train those heads. If it does not, oversample sessions containing carts or orders and record the reweighting, because it changes base rates.

**Alternatives considered.** The full dataset on Athena, Glue or EMR (cost, and scale beyond a sample teaches little more). Local Spark (heavy on 8GB). Random event sampling (breaks sessions). A random split instead of a temporal one (leaks the future). Also check the dataset license and terms before publishing any derived sample.

#### D4. Synthetic categories

**Decision.** Build an item co-visitation graph from the training window only. Embed items with truncated SVD of that matrix, then cluster with k-means into roughly 50 to 100 categories, choosing K by stability across seeds and size balance. Name the clusters from a fictional fashion-and-home taxonomy generated once with Claude. Align it with the cluster hierarchy: order clusters by hierarchical clustering of their centroids, order taxonomy leaves by tree position, and pair them, so related names land on related clusters. Label these categories as synthetic everywhere in the UI.

**Why.** The classifier needs human-readable names, and OTTO has none. K in the 50 to 100 range gives a classifier something meaningful to discriminate while keeping thousands of items per category in the sample. Clustering on the training window only prevents the categories from encoding future behavior. The names carry no true semantics, which is why the classifier is evaluated as a pipeline component (D17), not as evidence about real query understanding.

**Alternatives considered.** Opaque labels such as cluster_017 (an LLM cannot classify a term against them). Names generated from item statistics (nothing semantic to go on). A real taxonomy from another dataset (no join key). Clustering on all weeks (leakage). Graph community detection such as Leiden (a valid alternative, worth a quick comparison against k-means). Item2vec embeddings instead of SVD (similar in spirit, more tuning). Much finer or coarser K (too sparse, or too vague to be a search box).

**Revisit.** Week 2, after inspecting cluster sizes and stability.

#### D5. Two separate query constructs

**Decision.** Build two distinct kinds of query and never mix them. For ranking evaluation, a query is the category of the session's held-out next click, paired with the session prefix. This is a documented leak, used as a proxy for intent, and the model is evaluated on ranking within that category. For classifier evaluation, generate a separate held-out set of paraphrased, misspelled, multi-intent and out-of-scope search terms per category. For a subset of the ranking evaluation, run the classifier in the loop (its predicted distribution feeds retrieval) to measure how classifier errors propagate.

**Why.** OTTO has no queries, so something must stand in. Deriving the category from the label makes ranking easier than real search, and saying so is better than hiding it. The classifier needs text queries that ranking evaluation cannot supply. Testing with the classifier in the loop shows the end-to-end cost of query understanding errors.

**Alternatives considered.** Dropping queries entirely and treating the task as pure session recommendation (rejected: the product is search). Using the session's category mixture as a soft query (less clean to evaluate). Using the classifier output for all evaluation (couples two error sources and makes debugging harder).

#### D6. Session-only personalization

**Decision.** The user side of every model is the current session prefix: recent items, event types and recency. Start with pooled item embeddings weighted by recency and event type.

**Why.** OTTO sessions are anonymous, with no cross-session identity, so nothing else is possible. Simple pooling keeps the Two-Tower cheap enough to train on a Mac-sized budget.

**Alternatives considered.** Cross-session history (data does not exist). No personalization (throws away the main signal in the data). A transformer session encoder such as SASRec (stronger and a good extension, but more compute and tuning, deferred).

### 6.B Evaluation

#### D7. Metrics and splits

**Decision.** Primary metric is NDCG@10 against held-out behavior. Also report recall@K for retrieval stages (K at 20, 50, 100, 200), and OTTO's weighted recall@20 (weights 0.10 for clicks, 0.30 for carts, 0.60 for orders). Use graded gains for NDCG (provisionally click 1, cart 2, order 3). Report 95 percent bootstrap confidence intervals over sessions, and slice results by category, session length and item popularity bucket. Split by time only. Build and test the evaluation harness first, checking it against a library (scikit-learn or ranx) on toy cases.

**Why.** Confidence intervals make the promotion gate (D27) statistically meaningful. Slices expose popularity bias, a common failure of behavior-derived labels. Building evaluation before models prevents fitting the metric to the model.

**Alternatives considered.** MAP or MRR (less informative for graded multi-event relevance). AUC (not a ranking metric). Random splits (time leakage). Single point estimates (cannot tell noise from a real gain).

**Revisit.** Week 3, for the gain values.

#### D8. Five methods and a shared candidate pool

**Decision.** Compare BM25, co-visitation, LightGBM LambdaMART, Two-Tower and DCN-V2 + MMoE. The three retrieval-style methods (BM25, co-visitation, Two-Tower) are scored on their own top results and by recall@K. The two rerankers (LightGBM, DCN-V2 + MMoE) rerank the same candidate pool, built from the union of Two-Tower and co-visitation candidates. The live path's pool is finalized once recall is measured.

**Why.** If each method used its own retrieval, differences in NDCG would confound retrieval quality with ranking quality. A shared pool makes the reranker comparison attributable to the ranker.

**Alternatives considered.** End-to-end for every method (confounded). Rerankers on Two-Tower candidates only (favors the deep path unfairly). Two-Tower candidates only in the live path (simpler, but check whether adding co-visitation raises recall).

**Revisit.** Week 9, after Two-Tower recall is known.

#### D9. Frozen benchmark set

**Decision.** NDCG@10 is computed on the full held-out window. The judge and the comparison tab use a frozen, versioned benchmark of about 300 (category query, session prefix) pairs, stratified by category and session length, drawn from the held-out window. Free-text search exists only on the live tab.

**Why.** The judge runs as a cached batch job, so it needs a finite set. The comparison tab needs ground truth, and no ground truth exists for arbitrary text. A frozen set makes the numbers reproducible across model versions.

**Alternatives considered.** Judging live (cost, latency, nondeterminism). Arbitrary queries in the comparison tab (no ground truth). A much larger judged set (the hand-labeling burden for calibration grows, and the judge sees little content per item anyway).

### 6.C Models

#### D10. Co-visitation and BM25 baselines

**Decision.** Build co-visitation as a proper baseline: item-to-item counts from the training window with event-type weights and time decay, giving candidates and a score. Build BM25 as a floor, implemented from scratch once and checked against a library. Its documents are synthetic: the category name tokens plus the names of the item's most co-visited categories. Add a small side lab that applies BM25 to a real text dataset, outside the product, so the method is learned where it works.

**Why.** Co-visitation is strong on OTTO, so it is the honest bar the learned models must clear. BM25 on synthetic documents will behave close to set membership with ties, which is a real and explainable result. The side lab keeps the learning goal intact.

**Alternatives considered.** Dropping BM25 (loses the lexical baseline and the teaching moment). OpenSearch for BM25 (cost). LLM-generated item descriptions to enrich BM25 (circular, adds cost and complexity, noted as a possible future extension).

#### D11. LightGBM LambdaMART

**Decision.** Train LightGBM with the lambdarank objective on graded labels, evaluated on NDCG@10. Features include category confidence, co-visitation scores against session items (max, mean), item popularity (global, within category, recent), item click, cart and order rates, recency, session length, and Two-Tower similarity when available. Keep a binary-logloss version as an ablation.

**Why.** It is the standard industry GBDT baseline, strong on tabular features, cheap on CPU, and the best way to learn the lambda gradients and listwise objectives. It also makes the comparison against the deep ranker meaningful.

**Alternatives considered.** XGBoost ranking (comparable, no need for both). CatBoost YetiRank (interesting, lower priority). Pointwise classification only (loses list structure, kept as the ablation).

#### D12. Two-Tower with FAISS

**Decision.** The query tower takes the classifier's category distribution (top 3, as a confidence-weighted mix of category embeddings) plus a session encoder (D6). The item tower takes an item embedding, category and popularity buckets. Train with sampled softmax using in-batch negatives and logQ correction, and add same-category hard negatives as an ablation. Restrict the catalog to items above a minimum frequency in the sample. Serve with FAISS in process (IVF-PQ or HNSW, chosen by measured recall, latency and size), using exact search as the ground truth for ANN recall.

**Why.** It gives the full retrieval story: negative sampling and its bias, ANN tradeoffs and recall measurement. In-process FAISS costs nothing and fits in a Lambda image at this scale.

**Alternatives considered.** OpenSearch k-NN (managed cost, roughly a few hundred dollars a month for serverless, far less but single-node for the smallest provisioned domain). pgvector on Aurora or RDS (always-on cost). MemoryDB (always-on cost). ScaNN (unnecessary at this scale). Annoy (works, less flexible). Exact search only (fine at this scale, but gives no ANN lesson).

#### D13. DCN-V2 with MMoE

**Decision.** DCN-V2 cross layers for explicit feature interactions and MMoE with three task towers (click, cart, order). Inputs are session, item, category and Two-Tower features plus co-visitation features. Loss is a weighted sum of per-task binary cross-entropy. The final score blends task probabilities using OTTO's weights as the default, tunable on validation. Keep single-task models as an ablation to show whether multi-task helps.

**Why.** It matches the brief, and MMoE suits tasks with different sparsity (orders are rare) and correlation. DCN-V2 is well documented in industry practice, so results can be discussed against published lessons.

**Alternatives considered.** DeepFM, xDeepFM or DCN v1 (older or less relevant). DIN or transformer rerankers (a sequence-modeling extension, deferred). Shared-bottom multi-task (the baseline MMoE is meant to beat). PLE (a natural extension if MMoE shows seesaw effects). Separate single-task models (kept as the ablation).

#### D14. LLM ranker dropped

**Decision.** The LoRA-fine-tuned LLM ranker from the brief is not built. Co-visitation takes its place as the fifth method.

**Why.** OTTO has no item text, so a pretrained LLM's world knowledge goes unused and it becomes an expensive sequence model over ID tokens. Hosting cost is high (a single always-on GPU instance is roughly $1,000 a month, so scale-to-zero or cached outputs would be needed). A custom catalog-aware head may not run directly on vLLM, so serving is a technical risk. It also takes a large share of a 280 to 420 hour budget.

**Alternatives considered.** Keeping it as a core method (cost and time). An offline-only demo with cached outputs (possible, but weak evidence). A SASRec-style transformer as a cheaper sequence-model extension (a good future addition, and the right control if an LLM ranker is ever revisited). Deferring until after Level 2 (you chose to skip it entirely).

### 6.D Position bias

#### D15. Position bias by simulation

**Decision.** Because OTTO has no positions, run position-bias correction as a semi-synthetic study. Build a logging policy (for example popularity plus noise, or co-visitation based), generate ranked impressions, and simulate clicks from a position-based examination model with known propensities and true relevance taken from held-out behavior. Train a naive model and an IPW model, evaluate against known ground truth, use clipping and self-normalization, and plot the bias and variance tradeoff. The main pipeline trains on real OTTO labels with a propensity-weight column that defaults to 1, so the code supports IPW without pretending OTTO contains positions.

**Why.** IPW needs propensities. OTTO offers none, so applying IPW to raw OTTO would mean IPW with invented propensities and no way to check it. A simulation gives ground truth, which lets you show that the correction works, which is stronger evidence than applying it blindly. Training the main models on real labels keeps the evaluation about real behavior instead of a simulation.

**Alternatives considered.** IPW on OTTO with assumed propensities (unverifiable). Estimating propensities with EM or intervention harvesting (needs position variation OTTO does not have). Training the main pipeline on simulated impressions and clicks (throws away real behavior and turns the whole evaluation into a simulation). Skipping bias correction (loses a topic senior interviewers ask about). This is a deviation from the original brief, which applied IPW to training labels directly.

**Revisit.** Week 14, if you would prefer the main pipeline to train on simulated exposure.

### 6.E Claude roles

#### D16. Claude through Bedrock

**Decision.** All Claude calls go through Amazon Bedrock. Pick the smallest model that passes each role's evaluation (a Haiku-class model for the classifier and explainer, a stronger model for the judge is the starting guess). Model IDs live in configuration, not code. Choose one AWS region for everything and check Bedrock model availability there first, using inference profiles if required.

**Why.** IAM authentication works the same from a laptop and from Lambda, spend shows up in the same AWS bill and budget alarms, batch inference exists for the judge, and it keeps "runs on AWS" literally true.

**Alternatives considered.** The Anthropic API directly (simple, but adds secret management and a second bill). Open-weight models on Bedrock or locally (cheaper, but the roles need strong instruction following). Dropping the Claude roles (you chose to keep them all).

#### D17. Query classifier

**Decision.** Return structured output: the top 3 categories with confidences and an out-of-scope flag. Evaluate on the held-out synthetic query set from D5: top-1 and top-3 accuracy, expected calibration error, and out-of-scope precision and recall. Compare against a sentence-embedding nearest-category baseline and TF-IDF. The LLM is kept only if it beats the baselines by enough to justify its cost and latency. Cache by normalized query. Below a confidence threshold, blend the top categories in the query tower or ask the user to refine.

**Why.** The classifier is the entry point, and calibrated confidence is what makes downstream blending sensible. An embedding baseline is cheap and forces an honest answer to whether the LLM is needed.

**Alternatives considered.** Embedding-only classification (kept as the baseline). A fine-tuned small classifier (needs labeled data the project has to synthesize anyway). Keyword rules (brittle). Free-form LLM output without a schema (hard to evaluate).

#### D18. Explainer

**Decision.** Claude receives only a numeric payload for each result (category confidence, co-visitation score, click, cart and order probabilities, popularity rank, and group-level attribution values) and must cite only what is in the payload. An automated faithfulness test parses numbers and feature names from the output and checks them against the payload. In CI the test runs on recorded fixtures. A scheduled job runs it against live Bedrock output. Explanations load after the results render.

**Why.** Grounding is the point of the feature, and it is only real if it is tested. CI should not call Bedrock on every push (cost and flakiness).

**Alternatives considered.** Templated explanations with no LLM (deterministic and faithful, less natural, kept as the fallback and as a baseline). SHAP for every model (exact for LightGBM, awkward for DCN-V2). Free-form explanation with full context access (invites unfaithful claims).

**Revisit.** Week 15. The attribution method for DCN-V2 is undecided. Candidates are leave-one-feature-group-out deltas and integrated gradients, computed only for the top few results to keep latency down.

#### D19. Judge

**Decision.** Claude scores each of a method's top 10 items from 0 to 4 for fit with the session intent, seeing the query, the session's recent items with their category names, and each candidate's category name. The method identity is hidden and the order of methods is randomized. One call per (benchmark pair, method) returns ten scores, so a full run is about 1,500 calls. Results are cached by (pair, method, model version, prompt version). Calibrate against your own labels on 150 to 200 pairs (weighted kappa and Spearman) and test for position and verbosity effects. Write a disagreement analysis of cases where NDCG and the judge diverge, sorted by cause.

**Why.** The judge is the independent axis in the comparison. Caching and batching keep cost to a few dollars and make results reproducible. The judge must not see method scores or co-visitation counts, which would favor the co-visitation method and make the comparison circular.

**Limits to state openly.** Because items are IDs with synthetic category names, the judge mainly measures category coherence with the session, not true product relevance. Disagreements will partly be artifacts of that, and the write-up should say so.

**Alternatives considered.** Live judging (cost, latency, nondeterminism). Pairwise preferences (more reliable, but 10 pairs per query across five methods). Averaging several judge models (a second judge on a subset is a good later check). Human labels only (too slow to cover five methods). A single listwise score per method (coarser and harder to compare to NDCG).

### 6.F Platform and MLOps

#### D20. Docker-first, arm64 end to end

**Decision.** One repository with multi-stage Dockerfiles: a training image and a serving image, plus Docker Compose for local Airflow and MLflow. Build arm64 images everywhere, matching the Mac, Lambda on Graviton and Fargate on Graviton. Use a dev container for day-to-day work.

**Why.** The image is what moves between your laptop and AWS, which keeps behavior identical and makes Level 2 deployment a matter of pushing the same artifact. Graviton is also cheaper.

**Alternatives considered.** Conda environments only (no shippable artifact). amd64 images built through emulation (slow, and a mismatch with the Mac). Kubernetes with kind (overkill for one service). Check in week 1 that arm64 wheels exist for faiss, onnxruntime and lightgbm, and use conda-forge or a source build if one is missing.

#### D21. Training compute

**Decision.** At Level 0, prototype on the Mac (CPU or PyTorch MPS) and run full neural training on Colab, written as scripts with a config file and checkpoints saved to S3. At Level 1 the pipeline must run unattended, so Colab cannot be a pipeline step. Size the neural models so a full training run fits on CPU in the Docker training image in about an hour. If that is not possible, add an Airflow task that launches a SageMaker spot training job or an EC2 spot instance.

**Why.** Colab is free and good for exploration, but it cannot be automated reliably. Docker on a Mac cannot use MPS, so the automated path is CPU unless it uses cloud compute. Small models on a sample are a reasonable match for this project's goals.

**Alternatives considered.** Automating Colab (fragile). Always-on GPU (cost). SageMaker training from the start (pay per run, and acceptable as the fallback, but it adds SageMaker coupling early). Kaggle kernels (free GPU hours, but not part of an AWS pipeline).

**Revisit.** Weeks 10 to 12, once real training times are known.

#### D22. Airflow in local Docker Compose

**Decision.** Run Airflow locally in Docker Compose (or on a small EC2 instance started on demand). The DAG is triggered by a schedule and by a new weekly partition landing in S3 (a sensor).

**Why.** You asked for Airflow at Level 1, and managed Airflow is out of budget (MWAA starts at roughly $300 to $400 a month). OTTO's four weeks give a natural data-arrival simulation.

**Alternatives considered.** MWAA (cost). SageMaker Pipelines (the original plan, but it couples to SageMaker and does not match the Airflow requirement). Step Functions (cheap and AWS-native, but it is not Airflow). Dagster or Prefect (good tools, not what you asked for). Cron plus a Makefile (that is Level 0.5, not Level 1). GitHub Actions cron (weak data-dependency semantics).

#### D23. MLflow for tracking and registry

**Decision.** Run an MLflow server in Docker with a Postgres or SQLite backend and an S3 artifact store. Use registry aliases (champion, challenger). Each model bundle carries a manifest with model versions, feature schema hash, data snapshot ID, code commit SHA and metrics.

**Why.** It is free and portable, and the manifest gives end-to-end lineage from data to deployment, which the MLOps tab displays.

**Alternatives considered.** SageMaker Model Registry (AWS-native, but tied to SageMaker). Weights and Biases (SaaS, free tier, outside AWS). DVC (a good option for data versioning, may be added). Plain S3 with JSON manifests (simple, reinvents the registry).

#### D24. pandera for data validation

**Decision.** Validate with pandera at three points: raw ingest, after feature building, and the assembled training set. Also check model outputs (score range, no NaN, expected top-k size).

**Why.** It is lightweight, Python-native, and easy to run in unit tests and CI.

**Alternatives considered.** Great Expectations (heavier). Deequ (needs Spark). Soda (extra service). No validation (the original brief asked for a gate).

#### D25. Feature management

**Decision.** Offline features are stored as Parquet in S3 and computed point-in-time (as of the day before each session) to avoid leakage. Item features for serving live in DynamoDB on-demand, refreshed by the weekly pipeline. Training and serving call one shared feature implementation, and CI runs a parity test computing features both ways on a fixture. Session features are computed per request.

**Why.** Training-serving skew and point-in-time correctness are the real lessons of a feature store, and both can be demonstrated without a managed store. DynamoDB on-demand costs pennies at this scale.

**Alternatives considered.** SageMaker Feature Store (originally planned, adds online store cost and coupling). Feast (needs infrastructure). Redis or ElastiCache (always-on cost). Baking item features into the model bundle (simplest, cheapest and immutable, and the fallback if DynamoDB adds no value). Computing everything at request time (too slow).

#### D26. Serving on Lambda

**Decision.** Serve from an arm64 Lambda container image behind a Function URL, with the model bundle baked into the image, neural models run through ONNX Runtime, LightGBM used natively, and FAISS loaded at init. Use 2 to 4GB of memory and no provisioned concurrency. The frontend defaults to static replay (D33), so cold starts affect only live mode.

**Why.** Pay-per-request pricing keeps idle cost at zero, and the free tier covers demo traffic. Baking the bundle into the image makes each version immutable, so a canary alias maps cleanly to an image version.

**Alternatives considered.** SageMaker real-time endpoints (roughly $80 a month for one small always-on instance, and more for GPU). SageMaker Serverless Inference (CPU only, cold starts, SageMaker coupling). ECS Fargate (roughly $10 to $15 a month for the smallest always-on task, no cold starts, canary via CodeDeploy blue/green, and the switch if Lambda cold starts hurt). App Runner (similar to Fargate, less control). EC2 (manual operations). EKS (overkill).

**Revisit.** Weeks 22 and 23, after measuring cold start and image size.

#### D27. Promotion gate

**Decision.** A candidate model is promoted only if it passes validation, beats the champion on the primary metric with the lower bound of a paired-bootstrap 95 percent interval on the per-session difference above zero, does not regress beyond tolerance on guardrails (recall@20, NDCG on the smallest categories, p95 latency, bundle size), and passes a smoke test. A statistical tie means no promotion, and the reason is recorded.

**Why.** A point-estimate comparison promotes noise. Guardrails stop a model from winning on the headline metric by damaging something else. The logged reason for every rejection is evidence for the MLOps tab.

**Alternatives considered.** Point-estimate greater-than (noisy). Fixed absolute thresholds (do not track drift). Manual approval (breaks the Level 2 goal). Online A/B testing (there is no traffic).

#### D28. GitHub Actions CI with OIDC

**Decision.** On every push: lint and type checks, unit tests, pandera schema tests, DAG import tests, faithfulness tests on recorded fixtures, and a container build with a smoke test against a tiny fixture bundle. On merge to main: push to ECR, run terraform plan and apply, and start the CodeDeploy deployment. GitHub authenticates to AWS with OIDC and a least-privilege role, with no long-lived keys. Use ARM runners or emulation for arm64 builds.

**Why.** The free tier for public repositories covers this, and OIDC removes the most common credential leak. The smoke test catches broken images before deployment.

**Alternatives considered.** CodePipeline and CodeBuild (AWS-native, more setup, small cost). Jenkins (self-hosted operations). GitLab CI (a different platform). Pre-commit only (not CI). Check current runner pricing and arm64 availability for your repository visibility.

#### D29. Terraform for infrastructure as code

**Decision.** Terraform, with remote state in S3 and a plan step in CI.

**Why.** It is widely used and portable across clouds, and the plan and apply flow reviews well in pull requests.

**Alternatives considered.** CDK in Python (natural for a data scientist, generates CloudFormation, heavier for small stacks). SAM (good for Lambda, narrow otherwise). Raw CloudFormation (verbose). Pulumi (smaller community). Console clicking (defeats Level 2).

**Revisit.** Week 22. The choice is close, and switching costs little if you prefer CDK.

#### D30. CodeDeploy canary with alarm rollback

**Decision.** Deploy with CodeDeploy for Lambda, shifting alias traffic (for example 10 percent for 5 to 10 minutes, then the rest). A BeforeAllowTraffic hook runs smoke and golden-set checks, including top-k overlap with the champion and latency. CloudWatch alarms on error rate, p95 latency and quality proxies (score distribution shift, empty-result rate) trigger automatic rollback. A replay job generates OTTO-like requests during the canary window. Demonstrate failure injection both ways.

**Why.** It gives real canary and auto-rollback behavior, which is the point of the original guardrails, at essentially no cost (CodeDeploy for Lambda has no additional charge as far as I know, so check).

**Alternatives considered.** SageMaker deployment guardrails (originally planned, tied to SageMaker endpoints). All-at-once deploys with manual rollback (no guardrails). ECS blue/green (a valid path if you move to Fargate). Argo Rollouts or Flagger (need Kubernetes).

#### D31. Monitoring, drift and retraining triggers

**Decision.** Track input feature drift (PSI and KS tests), score distribution, top-k overlap with the champion, latency and error rate. Compute drift in the pipeline on each new week and publish serving metrics as CloudWatch custom metrics. Retraining triggers are: a new partition (Level 1), a code or feature-definition change through CI (Level 2), and a drift threshold breach (Level 2, which triggers retraining and never auto-promotes). Since there are no live users, drift is simulated by replaying later OTTO weeks, and the write-up says so.

**Why.** Lightweight custom metrics cover what a portfolio needs and keep the design portable. Separating triggers from promotion keeps a drift alarm from shipping an unvetted model.

**Alternatives considered.** SageMaker Model Monitor (endpoint-coupled). Evidently (good for offline reports, optional add-on). WhyLabs or Arize (SaaS). No monitoring (the brief listed it).

#### D32. Live endpoint protection

**Decision.** A public Function URL that calls Bedrock is a cost-abuse risk. Live mode requires an API key checked in code, the Lambda has a reserved concurrency cap, responses are cached, and a budget alarm notifies you. Keep a kill switch script that sets reserved concurrency to zero.

**Why.** A single misuse of a public LLM-backed endpoint can cost more than the whole rest of the project.

**Alternatives considered.** API Gateway REST with usage plans (roughly $3.50 per million requests, a good upgrade if abuse becomes real). API Gateway HTTP API (cheaper but has no usage plans). WAF (a monthly base cost that is too high for this project). No protection (unacceptable).

### 6.G Frontend and cost

#### D33. Static frontend with an optional live mode

**Decision.** A static single-page app (Vite with React, or plain HTML and JavaScript) served from S3 through CloudFront or from GitHub Pages. It reads versioned JSON exports produced by the pipeline (metrics, per-query results, judge scores, model lineage, canary events, per-level measurements). A toggle enables live mode against the Lambda. Tabs: Search, Methods, MLOps, Architecture.

**Why.** It costs almost nothing, is always up, and works during interviews even if the backend is torn down. Reading pipeline exports also demonstrates that the deployment produces its own evidence.

**Alternatives considered.** Streamlit or Gradio (need a server, or a non-AWS host). Next.js with server rendering (server cost). Grafana (excellent for operations, weak for telling the story). QuickSight (cost).

#### D34. Cost controls and teardown

**Decision.** Set budget alarms at low thresholds on day one. Use cost allocation tags. Keep nothing running that is not needed. Set short CloudWatch log retention. Add S3 and ECR lifecycle rules. Avoid NAT gateways (Lambda needs no VPC, and one NAT gateway costs roughly $32 a month idle). Script a one-command teardown with terraform destroy that keeps the state and artifact buckets.

**Why.** Idle resources, NAT gateways and log storage are the usual sources of surprise bills. Teardown also makes the project reproducible for a reviewer.

**Alternatives considered.** Relying on remembering to delete things (fails eventually). Hard spending limits (AWS budgets alert rather than stop, so alarms plus a kill switch are the practical approach).

## 7. Deliberately left out

Each item belongs on the architecture tab with its reason and price.

| Left out | Reason | Rough cost avoided |
|---|---|---|
| SageMaker real-time endpoints, Pipelines, Feature Store, Model Monitor | Cost and coupling. Lambda, Airflow, DynamoDB and custom monitoring teach the same ideas | About $80 a month per small endpoint, plus store and monitoring charges |
| Managed Airflow (MWAA) | Local Docker Compose covers Level 1 | Roughly $300 to $400 a month |
| OpenSearch | FAISS in process covers ANN, and BM25 is a floor on synthetic text | A few hundred a month for serverless |
| GPU serving and the LLM ranker | No item text, high cost, serving risk (D14) | Roughly $1,000 a month for one GPU instance |
| Cross-session personalization | No user identity in the data | Not applicable |
| Price, seller and availability signals | Not in the data | Not applicable |
| Online A/B testing and interleaving | No traffic. Offline evaluation and canary replay only | Not applicable |
| Real query text and query understanding at scale | No queries in the data. Classifier tested on generated queries | Not applicable |
| Streaming features (Kinesis, Flink) | Session features are computed per request. Streaming adds cost without a real-time source | Tens to hundreds a month |
| Kubernetes | One service does not need it | Cluster and node costs |
| NAT gateways, WAF, provisioned concurrency | Idle costs outside the project's needs | Roughly $32 a month per NAT gateway, plus WAF and Lambda provisioned charges |

## 8. Repository layout

```
search-ranking/
  README.md
  plan.md
  docs/decisions/            (new or changed decisions, same format as Section 6)
  notebooks/                 (Level 0 exploration)
  src/ranking/
    data/                    (sampling, splits, categories, query construction)
    features/                (one implementation shared by training and serving)
    eval/                    (metrics, bootstrap, benchmark set)
    models/                  (covis, bm25, lgbm, two_tower, dcn_mmoe)
    bias/                    (click simulator, IPW study)
    llm/                     (classifier, explainer, judge, prompts, fixtures)
    serving/                 (handler, retrieval, rerank)
  pipelines/airflow/dags/
  infra/terraform/
  frontend/
  docker/                    (Dockerfile.train, Dockerfile.serve, docker-compose.yml)
  tests/
  .github/workflows/
```

## 9. Timeline and milestones

About 28 weeks at 10 to 15 hours a week. Treat the dates as estimates.

| Weeks | Milestone | Done when |
|---|---|---|
| 1 to 2 | Setup and data: repo, dev container, budget alarm, arm64 check, OTTO sample, temporal split, clustering, query construction | Sampling is scripted and reproducible, a leakage checklist passes, cart and order rates are checked |
| 3 to 4 | Evaluation harness and baselines: popularity, co-visitation, BM25 floor | Metrics match a library on toy cases, every baseline has a score with a bootstrap interval |
| 5 to 6 | LightGBM LambdaMART over the shared candidate pool | It beats the baselines, and you can explain the lambda gradients and feature importances |
| 7 to 9 | Two-Tower with FAISS | Recall@K at several K, negative-sampling ablation, ANN recall against exact search |
| 10 to 12 | DCN-V2 with MMoE on click, cart and order | Compared to LightGBM per task, with a single-task ablation, CPU training time measured |
| 13 to 14 | Position-bias study | Naive model shows the bias, IPW recovers the true ordering, clipping tradeoff plotted |
| 15 to 17 | Claude roles and manual deploy: classifier evaluation, explainer faithfulness test, judge calibration, hand-deployed container | Classifier calibration and judge agreement with your labels are reported, Level 0 exit criteria met |
| 18 to 21 | Level 1: Airflow DAG, pandera gates, MLflow registry, promotion gate, weekly trigger | Consecutive weeks give one promotion and one rejection, reproducibly |
| 22 to 25 | Level 2: Actions CI, Terraform, Lambda deployment, CodeDeploy canary, failure injection | Both failure cases are demonstrated and a feature change deploys hands-free |
| 26 to 28 | Frontend: Search, Methods, MLOps, Architecture tabs | Static site reads pipeline exports, live toggle works |

Checkpoint at the end of week 14. If Level 0 modeling looks likely to run past week 17, trim rather than push. Cut in this order: the standalone BM25 side lab, explainer polish, the frontend's live toggle, then shrink the position-bias study to one clean simulation. Protect the evaluation harness, the promotion gate and the failure-injection demo, because they carry the most signal.

Two habits to keep from week 1. First, log every manual step and how long it took, because those numbers cannot be reconstructed later and they feed the MLOps tab. Second, keep a decisions log in docs/decisions with a short entry every time something is chosen or cut, including the reason and the price where relevant.

## 10. Rough cost picture

These are estimates for a lean build with nothing left running. Verify with the AWS Pricing Calculator.

| Component | Lean choice | Rough cost | Rejected alternative | Its rough cost |
|---|---|---|---|---|
| Storage, registry, features | S3, ECR, DynamoDB on-demand | Pennies to free tier | SageMaker Feature Store online store | Ongoing per-unit charges |
| Serving | Lambda arm64 | Free tier at demo traffic | SageMaker real-time endpoint | About $80 a month per small instance |
| ANN | FAISS in process | Zero | OpenSearch serverless | A few hundred a month |
| Orchestration | Airflow in local Docker | Zero | MWAA | Roughly $300 to $400 a month |
| Training | Mac and Colab free tier | Zero | On-demand GPU instances | Hourly charges, spot much cheaper |
| Claude | Bedrock, cached and batched | A few dollars total | Live judging | Unbounded with traffic |
| Frontend | S3 with CloudFront, or GitHub Pages | Free tier | Always-on app server | Monthly instance cost |
| Deployment | CodeDeploy for Lambda | No additional charge (verify) | SageMaker guardrails | Tied to endpoint costs |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Level 0 sprawl eats time meant for Levels 1 and 2 | Time-boxed Level 0, model freeze, week 14 checkpoint, cut order in Section 9 |
| Synthetic categories make results circular | Cluster on the training window only, documented leaks (D4, D5), classifier and judge on independent constructs, honest write-up |
| The judge measures little because items are IDs | Calibrate against your labels, state the limit (D19), treat as a coherence check |
| CPU-only Level 1 training is too slow | Measure early, keep models small, fall back to spot training (D21) |
| arm64 wheels missing (FAISS, ONNX Runtime, LightGBM) | Check in week 1 with a hello-world image, use conda-forge or a source build |
| Lambda cold start or image size hurts live mode | Baked bundle, ONNX, measure, fall back to Fargate (D26) |
| Cost runaway from an exposed endpoint or idle resources | Budget alarms, API key, reserved concurrency cap, kill switch (D32, D34) |
| Too few cart or order events in the sample | Stratify the sample and record the reweighting (D3) |
| Scope creep from extras (LLM ranker, SASRec) | Parked list, revisit only after Level 2 |
| OTTO license or terms limit publishing derived data | Check in week 1, keep samples out of public repos unless allowed |
| Bedrock model availability or pricing changes | Configuration-driven model IDs, verify early |
| Frontend polish consumes the schedule | Static site, fixed 3-week window, live toggle is first to cut |

## 12. Open questions and checkpoints

| Question | Decide by |
|---|---|
| Region and Bedrock model availability in it | Week 1 |
| OTTO license and what may be published | Week 1 |
| Number of clusters K and the naming alignment method | Week 2 |
| Graded gains for NDCG | Week 3 |
| Source for the published OTTO benchmark comparing GBDT and deep rankers, cited in the original brief (find the citation before quoting it) | Week 5 |
| Composition of the shared candidate pool | Week 9 |
| CPU training time and whether a spot-training task is needed | Weeks 10 to 12 |
| Whether the main pipeline should train on simulated exposure (D15) | Week 14 |
| Attribution method for the DCN-V2 explainer | Week 15 |
| Terraform or CDK, and Lambda or Fargate | Weeks 22 and 23 |

## 13. Definition of done

- A reviewer can follow documented steps to reproduce the sample-based metrics table from the frozen bundle.
- The three levels differ by measured numbers (manual steps, retrain time, deploy time, rollback time, reproducibility, test count).
- All five methods have NDCG@10 with confidence intervals and judge scores with a calibration report.
- The position-bias study is reproducible and plotted.
- A rejected promotion and a canary rollback have each been demonstrated and recorded.
- The static frontend is deployed with four tabs.
- The architecture tab lists omissions with reasons and prices.
- Spend stayed inside your budget and teardown works.
- The decision log is current.

## 14. References

These are from memory. Check titles and details before citing them.

- Google Cloud Architecture Center, "MLOps: Continuous delivery and automation pipelines in machine learning" (the source of maturity levels 0, 1 and 2).
- OTTO recommender dataset and Kaggle competition (otto-de recsys-dataset).
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" (2010).
- Wang et al., "DCN V2: Improved Deep and Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (2021).
- Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018).
- Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019).
- Joachims, Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback" (WSDM 2017).
- Zalando's published LLM-as-judge design for search quality assurance, and the Netflix GenRec work, both named in the original brief (the latter is no longer built).
