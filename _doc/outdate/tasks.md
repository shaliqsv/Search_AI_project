# Backlog: Search and Ranking on OTTO

The work runs in two halves. Phases A to F are notebooks only: as a full-stack data scientist you build and understand every algorithm using local tools, with no AWS, no MLflow and no pipelines. Phase F ends with a decision on which models to keep. Only after that do the MLOps levels begin (Phases G to I), where AWS, MLflow and the rest first appear. Phase J is the frontend and wrap-up.

The levels follow these definitions:
- Level 0, manual: notebook-driven training, artifacts handed off by hand, manual and infrequent deployment, no monitoring, no separation between experiment and production code.
- Level 1, pipeline automation: an automated training pipeline, a feature store shared by training and serving, experiment tracking, a model registry, continuous training on new data, and a human approval before production.
- Level 2, CI/CD: pipeline code under CI/CD, automated model validation, automated canary deployment, monitoring with alerting and retraining triggers, comparison of models on live traffic, and full reproducibility from data version to deployed model.

If time runs short, cut in this order: BM25 on real text (task 20), explainer polish, the frontend live toggle, then shrink the position-bias study to one clean simulation. Protect the evaluation tasks, the promotion gate and the failure-injection demo, because they carry the most signal.

Each task is written to stand alone, so the descriptions name their own inputs. Decision IDs (D#) in plan.md are background only.

## Phase A: Setup

## 1. Set up an empty project with a passing test
Goal: A fresh Python project where one trivial test runs and passes.
Description: Create the `src/ranking/` package layout, a `pyproject.toml`, a `tests/` folder, and a linter and test runner configuration. Add one placeholder test that passes with a single command, and document that command in the README.

## 2. Set up the notebook environment
Goal: Jupyter runs against the project environment, with a notebook folder and conventions in place.
Description: Add a `notebooks/` folder with a numbered naming scheme, a kernel that uses the project's environment, and a tool that strips outputs before commit. Include a short template notebook (title, question, data used, result, what I learned) that every later notebook starts from.

## 3. Check the OTTO license and terms
Goal: A written note on what OTTO data may be used and published.
Description: Read the OTTO dataset license and terms and state whether derived samples can go in a public repo. Record the answer in the README so later tasks know what may be committed.

## Phase B: Data notebooks

## 4. Notebook: sample OTTO sessions
Goal: A notebook that samples 1 to 2 million whole sessions into parquet, reproducibly.
Description: Select sessions by hash of session ID so the sample is stable, keeping all events of each chosen session. Run it in a Kaggle notebook with the dataset attached, using only the four labeled training weeks, and download the parquet. Explain in the notebook why whole sessions are sampled and not random events.

## 5. Notebook: explore the sample and check event balance
Goal: A notebook showing event counts by type, session length distribution and item frequency distribution.
Description: Use DuckDB or Polars to load the parquet without exhausting 8GB of memory. Check whether carts and orders are frequent enough to train on, and if not, prototype oversampling sessions that contain them and record how it changes the base rates.

## 6. Notebook: temporal split
Goal: A notebook and function splitting the sample by time into training and evaluation windows.
Description: Use weeks 1 and 2 as the initial training window, weeks 3 and 4 as later arrivals, and evaluation days directly after each training window. Show with assertions that no evaluation event precedes the end of its training window, and explain why random splits leak.

## 7. Notebook: build the item co-visitation matrix
Goal: A notebook building a sparse item-to-item co-visitation matrix from a training window.
Description: Count co-occurrence within sessions, weighted by event type and time decay, using only the given training window. Save the matrix and the item ID mapping. Check a few pairs by hand on a tiny example.

## 8. Notebook: cluster items into synthetic categories
Goal: A notebook assigning items above a minimum frequency to one of about 50 to 100 categories.
Description: Reduce the co-visitation matrix with truncated SVD and cluster with k-means. Choose K by stability across several seeds and cluster size balance, and plot both. Compare against one alternative such as Leiden community detection.

## 9. Notebook: name the categories from a fictional taxonomy
Goal: A category table mapping each cluster ID to a readable synthetic name.
Description: Generate a fictional fashion-and-home taxonomy once with an LLM (Claude) and save it as a fixed file. Order clusters by hierarchical clustering of their centroids, order taxonomy leaves by tree position, and pair them so related names land on related clusters. Label every output as synthetic.

## 10. Notebook: build ranking-evaluation queries
Goal: A dataset of (category query, session prefix, held-out labels) records from the evaluation window.
Description: For each evaluation session, take the category of the held-out next click as the query and the earlier events as the prefix, and record the held-out clicks, carts and orders as labels. State clearly in the notebook that this is a documented leak used as a proxy for intent.

## 11. Notebook: generate classifier evaluation queries
Goal: A held-out set of synthetic search terms per category, including hard cases.
Description: Use an LLM (Claude) to generate paraphrases, misspellings, multi-intent terms and out-of-scope terms for each category name, and keep the set separate from the ranking queries. Save it with the true labels and the out-of-scope flag as a versioned file.

## 12. Notebook: leakage audit
Goal: A checklist and runnable checks showing no future information reaches training.
Description: List every place leakage could occur (categories, co-visitation, features, splits, query construction) and write an assertion for each one that can be automated. Run them against the sample and record which leaks are known and accepted.

## Phase C: Evaluation notebooks

## 13. Notebook: implement ranking metrics from scratch
Goal: NDCG@K, recall@K and OTTO weighted recall@20 written by hand and checked against a library.
Description: Implement graded gains (click 1, cart 2, order 3, adjustable) and OTTO's weights of 0.10, 0.30 and 0.60. Confirm NDCG matches scikit-learn or ranx on toy cases, and explain what each metric rewards.

## 14. Notebook: bootstrap confidence intervals
Goal: A function returning 95 percent bootstrap intervals over sessions, plus a paired version for comparing two models.
Description: Resample sessions to get the interval for one model, and for two models resample the per-session differences. Test on synthetic data with a known true gap, and show how interval width shrinks with more sessions.

## 15. Notebook: sliced evaluation
Goal: A function reporting metrics by category, session length and item popularity bucket.
Description: Take per-session results with metadata and produce a table of metric and interval per slice. Flag slices with too few sessions instead of hiding them, and explain what popularity slices reveal.

## 16. Notebook: freeze the benchmark set
Goal: A versioned file of about 300 (category query, session prefix) pairs from the evaluation window.
Description: Sample pairs stratified by category and session length with a fixed seed. Save them with a version number and content hash. This set is used later by the judge and the methods comparison.

## Phase D: Model notebooks

## 17. Notebook: popularity baseline
Goal: A popularity-within-category ranker scored on the evaluation queries.
Description: Compute popularity from the training window, rank candidates by it, and report NDCG@10 with an interval. It sets the lowest bar.

## 18. Notebook: co-visitation baseline
Goal: A co-visitation ranker scored on the evaluation queries.
Description: Aggregate co-visitation scores from the session's items, filter to the query category, and return the top results. Report recall@K, weighted recall@20 and NDCG@10, and note why this is the strong non-learned bar.

## 19. Notebook: BM25 from scratch
Goal: A BM25 ranker over synthetic item documents, checked against a library.
Description: Build documents from each item's category name tokens plus the names of its most co-visited categories. Implement BM25 by hand and confirm scores match a library on a small fixture. Score it as the floor method and explain why its results look like set membership with ties.

## 20. Notebook: BM25 on real text
Goal: A side notebook that applies BM25 to a public text dataset where it works.
Description: Use any small public corpus with queries and relevance judgements, and evaluate your BM25 with the metrics from earlier. This keeps the lexical retrieval lesson intact even though OTTO has no text. Lowest priority; skip if time is short.

## 21. Notebook: shared candidate pool
Goal: A function merging candidate lists from several retrievers with deduplication and a size cap, with recall@K measured.
Description: Start with co-visitation candidates and leave a slot for Two-Tower candidates to be added later. Report recall@20, 50, 100 and 200 of the union. Explain why the rerankers must share a pool to be compared fairly.

## 22. Notebook: point-in-time features
Goal: Functions computing item, session and category features as of a given time.
Description: Compute popularity, click, cart and order rates, recency, session length and co-visitation aggregates from data before the cutoff only. Keep them as plain importable functions, so there is one definition of each feature.

## 23. Notebook: LightGBM LambdaMART
Goal: A trained LightGBM lambdarank model over the candidate pool, with a scored result.
Description: Build a training table of candidates with graded labels and features, train with the lambdarank objective, and report NDCG@10 with an interval. Add a binary-logloss ablation and feature importances, and write up how lambda gradients work in your own words.

## 24. Notebook: session encoder and Two-Tower training
Goal: A Two-Tower model trained with in-batch negatives and logQ correction.
Description: The query tower takes a category distribution and a recency-and-event-weighted pooled session embedding. The item tower takes an item embedding, category and popularity bucket. Train in PyTorch on the sample, save the checkpoint and item embeddings, and explain why sampling bias needs correcting.

## 25. Notebook: Two-Tower hard negatives ablation
Goal: A recall comparison between in-batch negatives and same-category hard negatives.
Description: Train a second version with hard negatives from the same category and compare recall@K. Record training time for both.

## 26. Notebook: FAISS index and ANN recall
Goal: Exact, IVF-PQ and HNSW indexes over item embeddings, with recall against exact search.
Description: For each index, measure recall against exact search, query latency and index size. Pick one and write down why. Then add Two-Tower candidates to the shared pool and report the recall gain.

## 27. Notebook: DCN-V2 with MMoE
Goal: A trained multi-task ranker that predicts click, cart and order probabilities.
Description: Implement cross layers, a mixture-of-experts layer and three task towers in PyTorch, using session, item, category, Two-Tower and co-visitation features. Train with a weighted sum of per-task binary cross-entropy and blend outputs with OTTO's weights for the score.

## 28. Notebook: multi-task ablations
Goal: A table comparing MMoE with single-task and shared-bottom models per task.
Description: Train single-task versions and a shared-bottom multi-task version on the same data and features. Compare per-task metrics with intervals, and say whether multi-task helps, especially for orders.

## 29. Notebook: compare all five methods
Goal: One table of NDCG@10 with intervals and recall@K for BM25, co-visitation, LightGBM, Two-Tower and DCN-V2 with MMoE.
Description: Score all five on the full evaluation window and on the frozen benchmark set, with sliced results for at least category and popularity. Write down what surprised you.

## 30. Notebook: measure CPU training time
Goal: A recorded training time and memory use for each neural model on CPU.
Description: Run Two-Tower and DCN-V2 training on CPU at a fixed sample size and record wall-clock time and memory. State whether each fits in about an hour on CPU, because later phases may need training that runs without a GPU.

## Phase E: Position bias notebooks

## 31. Notebook: click simulator
Goal: A simulator that produces ranked impressions and clicks from known position propensities.
Description: Implement a logging policy (popularity plus noise) and a position-based examination model with known propensities, using held-out behavior as true relevance. Save the impressions, clicks and the propensities used.

## 32. Notebook: position-bias IPW study
Goal: A comparison of naive and IPW-trained rankers against known ground truth, with plots.
Description: Train one model naively and one with inverse propensity weights on the simulated data, then evaluate both against true relevance. Add clipping and self-normalization and plot the bias and variance tradeoff.

## 33. Add a propensity-weight column to the training code
Goal: The training code accepts a per-example weight column that defaults to 1.
Description: Pass a weight column to LightGBM and to the neural loss. Check that weights of 1 reproduce previous results and that non-uniform weights change the loss.

## Phase F: Claude roles notebooks and the model decision

## 34. Notebook: Claude query classifier
Goal: A function that takes a search term and returns the top 3 categories with confidences and an out-of-scope flag.
Description: Call a Claude model through the Anthropic API with a structured-output prompt that lists the category names. Put the client behind a small interface so the provider can change later, keep the model ID in configuration, cache by normalized query in a local file, and save recorded responses as fixtures.

## 35. Notebook: evaluate the classifier against baselines
Goal: A report of top-1 and top-3 accuracy, calibration error and out-of-scope precision and recall for the classifier and two baselines.
Description: Run the classifier, a sentence-embedding nearest-category baseline and a TF-IDF baseline on the held-out synthetic query set. Say whether the LLM justifies its cost and latency.

## 36. Notebook: end-to-end run with the classifier in the loop
Goal: A measurement of how classifier errors change ranking quality.
Description: Feed the classifier's predicted category distribution into the Two-Tower query tower on a subset of the evaluation queries and compare NDCG@10 with the true-category version. Report how the loss varies with classifier confidence.

## 37. Notebook: explainer and faithfulness test
Goal: A Claude explainer that cites only numbers in its input, with an automated check that verifies this.
Description: Define a numeric payload per result (category confidence, co-visitation score, task probabilities, popularity rank) and prompt the model to explain using only that. Parse numbers and feature names from the output and compare them with the payload, using recorded fixtures.

## 38. Notebook: attribution for the explainer
Goal: A per-result attribution method for the DCN-V2 model that fits in the explainer payload.
Description: Implement leave-one-feature-group-out deltas and compare with integrated gradients on a few results. Pick one by faithfulness and speed, computed only for the top few results.

## 39. Notebook: LLM judge batch run
Goal: A batch run that scores each method's top 10 items from 0 to 4 for each benchmark pair.
Description: For each (pair, method), send the query, the session's recent items with category names and the candidates' category names, hiding the method and randomizing order. Cache results in a local file by pair, method, model version and prompt version so reruns cost nothing.

## 40. Notebook: calibrate the judge against your labels
Goal: Agreement figures between the judge and hand labels on 150 to 200 pairs.
Description: Label a sample of judged pairs by hand, then compute weighted kappa and Spearman correlation against the judge. Test for position and verbosity effects by re-running with shuffled order and padded descriptions.

## 41. Notebook: NDCG-versus-judge disagreement analysis
Goal: A written analysis of cases where NDCG and the judge disagree, grouped by cause.
Description: Find benchmark pairs where a method ranks high on one measure and low on the other, and sort them by cause. State plainly that the judge mostly measures category coherence because items are IDs.

## 42. Decide and freeze the model set and results
Goal: The final choice of models, frozen with configuration and a results table, for the ops phases to deploy.
Description: Review the five-method comparison and decide which models go forward and which are dropped. Fix the seeds, save the final model files, the FAISS index, the category table and the benchmark set, and record their hashes and the final metrics table. After this task no modeling changes, only engineering.

## Phase G: Level 0 (manual process)

## 43. Set AWS budget alarms and cost tags
Goal: Budget alarms exist at low dollar thresholds and a cost allocation tag is defined.
Description: In one chosen AWS region, create monthly budget alerts (for example $5, $20 and $50) that email you. Define a project tag to apply to all later resources. This is the first AWS step in the project, so do it before anything else in this phase.

## 44. Create a manual-step log
Goal: A log file and template for recording every hand-run step and its duration.
Description: Create `docs/manual-steps.md` with a table of step, date, time taken and what went wrong. Start using it from this task on. These numbers feed the level comparison later and cannot be reconstructed after the fact.

## 45. Check Bedrock model availability and move the LLM client to it
Goal: The Claude calls from the notebooks run through Amazon Bedrock in the chosen region.
Description: Check which Claude models Bedrock offers in the region, including whether inference profiles are needed, and record model IDs and prices. Swap the notebook LLM client for a Bedrock one behind the same interface, keeping model IDs in configuration, and re-run the recorded fixtures to check nothing changed.

## 46. Hand off the model artifacts by hand
Goal: The frozen models and data files exported from the notebooks and uploaded to S3 manually.
Description: Save the LightGBM model, the Two-Tower and DCN-V2 with MMoE weights, the FAISS index, the category table and item features from the notebooks, export the neural models to ONNX, and check ONNX outputs match PyTorch on a few inputs. Upload the files to an S3 bucket by hand and log each step and its time.

## 47. Write a serving script by copying notebook code
Goal: A Lambda-style handler that takes a query and session and returns the top 10 results with explanations.
Description: Copy the needed code from the notebooks into a single handler file, loading the artifacts at start and running the classifier, query tower, FAISS search, reranker and score blend, then the explainer. Level 0 deliberately has no shared package and no tests, and you should log how much copy-and-adapt work this took.

## 48. Build the serving container image
Goal: An arm64 serving image that runs the handler locally through the Lambda runtime interface emulator.
Description: Bake the artifacts into a Lambda base image and confirm a local request returns results. Record image size and cold-start time, and check arm64 wheels for faiss, onnxruntime and lightgbm.

## 49. Deploy the container to Lambda by hand
Goal: A Lambda function serving live requests, deployed manually once.
Description: Push the image to ECR, create the function with 2 to 4GB of memory and a Function URL, and send a test request. Record every manual step and its time in the manual-step log.

## 50. Add live endpoint protection
Goal: The Function URL rejects requests without an API key and has a spending cap and a kill switch.
Description: Check an API key in the handler, set a reserved concurrency limit, cache responses, and write a script that sets reserved concurrency to zero. Test that unauthenticated requests are refused.

## 51. Record the Level 0 baseline measurements
Goal: A table of manual steps, retrain time, deploy time and rollback time for Level 0, with a note on what is missing.
Description: From the manual-step log, count the hand-run steps and total the time for retraining (re-running notebooks), deployment and rollback. Note that there is no monitoring, no automated tests and no separation between experiment and production code. These become the baseline for Levels 1 and 2.

## Phase H: Level 1 (pipeline automation)

## 52. Move notebook code into the package with tests
Goal: The data, evaluation, feature and model code from the notebooks lives in `src/ranking/` with unit tests.
Description: Extract the sampling, split, co-visitation, category, metric, bootstrap, feature and model functions into modules with tests on tiny fixtures. This is the separation of experiment and production code that Level 0 lacked. Notebooks should import from the package afterwards, and results should match the frozen table.

## 53. Turn model training into scripts with configs
Goal: Each model trains from a command line script and a config file, and reproduces the frozen results.
Description: Turn the LightGBM, Two-Tower and DCN-V2 with MMoE training code into scripts that read a config, set seeds and save checkpoints. Confirm the metrics match the frozen table within tolerance.

## 54. Build a training container image
Goal: An arm64 training image that runs the training scripts on the sample.
Description: Write a multi-stage Dockerfile for training, run one training script in the container and compare its metrics with the frozen table. Confirm the run fits the CPU time measured earlier or note that a spot training task will be needed.

## 55. Run Airflow and MLflow in Docker Compose
Goal: A Compose file that starts Airflow and an MLflow server with a healthy scheduler and UI.
Description: Configure Airflow with a local executor and MLflow with a SQLite or Postgres backend and an S3 or local artifact store. Include a trivial DAG that logs one run to MLflow to prove the setup works. MLflow provides the experiment tracking for this level.

## 56. Add experiment tracking to training
Goal: Every training run logs its parameters, metrics and artifacts to MLflow.
Description: Log the config, seed, code commit SHA, data snapshot ID, metrics with intervals and model files from each training script. Check a run can be found and compared with another in the MLflow UI.

## 57. Write pandera schemas and validation checks
Goal: Schemas for raw events, built features and the assembled training table, plus a model output check.
Description: Define pandera schemas with types, ranges and null rules for each stage. Add a check that model scores are in range, contain no NaN and return the expected number of results. Cover each with passing and failing tests.

## 58. Build the shared feature store
Goal: Item features stored point-in-time in Parquet for training and in DynamoDB for serving, from one shared implementation.
Description: Write the weekly item features to Parquet computed as of the day before each session, and load the same values into a DynamoDB on-demand table keyed by item ID. Have the training code and the serving handler call the same feature functions, and add a batched read for serving. If DynamoDB adds no value over baking features into the bundle, record that and use the bundle instead.

## 59. Add a training-serving feature parity test
Goal: A test proving training and serving compute identical features on a fixture.
Description: Compute features for a fixture session through the training path and the serving path and assert they match within a tolerance. Run it as part of the normal test suite.

## 60. Define the model bundle and manifest
Goal: A bundle format holding all model artifacts and a manifest with lineage.
Description: Specify a folder layout with the model files, category table, FAISS index and item features, plus a manifest with model versions, feature schema hash, data snapshot ID, code commit SHA and metrics. Write a script that builds and validates a bundle.

## 61. Implement the promotion gate
Goal: A function deciding whether a candidate model beats the champion, with a logged reason.
Description: Promote only if validation passes, the lower bound of the paired-bootstrap interval on the NDCG@10 difference is above zero, guardrails (recall@20, small-category NDCG, latency, bundle size) hold and a smoke test passes. Return the decision and reason, and test each rejection path.

## 62. Wire the MLflow model registry
Goal: Functions that register a model bundle and move the champion alias.
Description: Register bundles with the manifest as tags and the metrics attached, set the challenger alias on registration and move champion on promotion. Test that all manifest fields are present.

## 63. Simulate weekly data arrival
Goal: A script that lands the next weekly partition into an S3 or local folder on demand.
Description: Copy the next week of the sample into an arrival location and write a marker file when it is complete. This stands in for real data arriving, and the pipeline will watch it.

## 64. Build the Airflow training DAG
Goal: A DAG that runs from partition arrival through validate, build features, train, evaluate, gate and register.
Description: Use a sensor on the arrival marker and let each step call the existing package code. End at registering the challenger, and add a DAG import test and an end-to-end run on tiny fixture data.

## 65. Add the manual deployment approval step
Goal: A human approval that must happen before a promoted model reaches production.
Description: Add an approval step, such as a manually triggered Airflow task or DAG, that deploys the champion bundle to Lambda only after you confirm. Log who approved and when, and time the deployment for the level comparison.

## 66. Show one promotion and one rejection over two weeks
Goal: Two consecutive weekly runs where one model is promoted and one is rejected, with logged reasons.
Description: Run the DAG on the week 3 and week 4 arrivals, arranging (for example with a deliberately degraded candidate) for one to fail the gate. Save the logged metrics and reasons, and check that rerunning the same commit gives the same metrics.

## 67. Record the Level 1 measurements
Goal: The same measurements as Level 0, recorded for Level 1.
Description: Count manual steps and time retraining, deployment and rollback, and record reproducibility (same commit, same metrics) and the automated test count. Add the row to the comparison table.

## Phase I: Level 2 (CI/CD)

## 68. Set up GitHub Actions CI
Goal: A workflow that runs lint, type checks, unit tests, DAG import tests and schema tests on every push.
Description: Add a workflow file with the checks and dependency caching. Include the parity test and the faithfulness tests on recorded fixtures, and make it fail on any error.

## 69. Add a container build and smoke test to CI
Goal: CI builds the arm64 serving image and runs a smoke request against a tiny fixture bundle.
Description: Extend the workflow to build the image (using an ARM runner or emulation), start it locally and send one request, failing if the response is malformed.

## 70. Write Terraform for base infrastructure
Goal: Terraform that creates the S3 buckets, ECR repository, DynamoDB table, IAM roles and log groups.
Description: Configure remote state in S3, apply short log retention and S3 and ECR lifecycle rules, and tag everything with the cost tag. Run `terraform plan` on a clean account to check it works.

## 71. Set up GitHub OIDC to AWS
Goal: GitHub Actions authenticates to AWS with a least-privilege role and no stored keys.
Description: Use Terraform to create the OIDC identity provider and an IAM role trusted only for this repository and the main branch. Add a workflow step that assumes the role and runs a harmless AWS command.

## 72. Deploy Lambda with Terraform and a CodeDeploy canary
Goal: Terraform for the Lambda function, alias and CodeDeploy deployment shifting 10 percent of traffic before the rest.
Description: Define the function from an ECR image, an alias and a CodeDeploy application with a canary configuration. Add a BeforeAllowTraffic hook that runs a smoke test.

## 73. Add automated model validation before traffic
Goal: A pre-traffic check that validates a new model against a golden set, with no human involved.
Description: In the BeforeAllowTraffic hook, run the golden benchmark queries and check top-k overlap with the champion, score ranges, latency and a small-category regression test. Fail the deployment if any check fails.

## 74. Add monitoring, alarms and automatic rollback
Goal: CloudWatch metrics and alarms on error rate, p95 latency, empty results and score-distribution shift, wired to CodeDeploy rollback.
Description: Have the handler publish custom metrics, define the alarms and attach them to the deployment. Verify each alarm exists and that a triggered alarm rolls the deployment back.

## 75. Compute drift metrics
Goal: A function computing PSI and KS statistics for input features and score distributions between two windows.
Description: Compare each feature's distribution in a new week with the training window and report PSI and KS per feature, plus a summary of score distribution shift. Add a threshold configuration, tests with synthetic shifted data, and an alert when a threshold is breached.

## 76. Add drift-triggered retraining
Goal: A trigger that starts the training DAG when drift exceeds a threshold, without promoting automatically.
Description: Run the drift function on each new week, and if a threshold is breached start a retraining run that still goes through the promotion gate. Test with a shifted week and an unshifted one.

## 77. Build the traffic replay job
Goal: A job that sends OTTO-like requests to the deployed alias during a canary window.
Description: Read sessions from the sample and send them to the function URL at a controlled rate. Record latency and error counts for later display. This stands in for real traffic, which the project does not have.

## 78. Build the model comparison on live traffic
Goal: Infrastructure that sends the same requests to a champion and a challenger and compares their outputs.
Description: Split or shadow replayed traffic between two Lambda aliases and log each version's top-k lists, latency and scores. Report overlap and metric differences. State that this is a simulated A/B setup because there are no real users, so it shows the infrastructure and not real conversion lift.

## 79. Complete the CD workflow
Goal: A GitHub Actions workflow that, on merge to main, pushes the image, applies Terraform and starts the deployment with no approval step.
Description: Chain the build, ECR push, `terraform apply` and CodeDeploy start, gated on a passing CI run. Confirm a feature-definition change reaches production with no manual step, and that any bundle can be traced back to its data version and code commit.

## 80. Demonstrate failure injection both ways
Goal: A recorded run where a bad model is rejected by the gate and a degraded deployment is rolled back.
Description: Feed the pipeline a deliberately worse model and show it is not promoted, then deploy a deliberately broken image and show the alarms roll it back. Save the logs and timings.

## 81. Record the Level 2 measurements and the comparison
Goal: Level 2 measurements added, and one comparison table across all three levels saved as JSON.
Description: Count manual steps and time retraining, deployment and rollback at Level 2, then combine with Levels 0 and 1. Save the table as a JSON file the frontend will read.

## Phase J: Frontend and wrap-up

## 82. Define the frontend data exports
Goal: A versioned set of JSON files with a documented schema for everything the frontend displays.
Description: Specify and generate JSON for metrics with intervals, per-query results, judge scores, model lineage, canary events and per-level measurements. Add a schema validation test.

## 83. Build the frontend shell and Search tab
Goal: A static app with a tab layout and a Search tab that replays exported results.
Description: Create the app (Vite with React or plain HTML and JavaScript) with tabs for Search, Methods, MLOps and Architecture. The Search tab shows a query, the classifier output, the top 10 with scores and the explanation, all read from the exports.

## 84. Add the live-mode toggle
Goal: A toggle that sends the Search tab's requests to the Lambda instead of using the exports.
Description: Add a settings field for the API key and endpoint, call the function, and handle errors and cold starts with a clear loading state. Fall back to replay if the call fails.

## 85. Build the Methods tab
Goal: A tab comparing the five methods with NDCG@10, intervals and judge scores.
Description: Show the same benchmark query run through all five methods, and highlight cases where NDCG and the judge disagree. Label categories as synthetic.

## 86. Build the MLOps tab
Goal: A tab showing the three maturity levels and their measured differences.
Description: Display the comparison table, the promotion and rejection runs with reasons, the canary and rollback events, and model lineage from the manifest. Read everything from the exports.

## 87. Build the Architecture tab
Goal: A tab explaining each platform choice and what was left out, with reasons and prices.
Description: Render the architecture diagram, the reasoning for each major choice and the table of omissions from the plan. Note that the prices are rough estimates that should be checked.

## 88. Deploy the frontend
Goal: The static site is live on S3 with CloudFront, or on GitHub Pages.
Description: Add the hosting resources to Terraform or configure GitHub Pages and publish through a workflow. Check the site works with the backend torn down.

## 89. Write the teardown script
Goal: A single command that removes running infrastructure but keeps the state and artifact buckets.
Description: Wrap `terraform destroy` so the state bucket and artifact bucket stay, and check afterwards that no billable resources remain. Test by tearing down and re-applying.

## 90. Write the reproduction guide
Goal: A document a reviewer can follow to reproduce the sample-based metrics table from the frozen bundle.
Description: List the exact commands, inputs and expected outputs, and try it from a clean checkout. Record anything that failed or was unclear and fix it.
