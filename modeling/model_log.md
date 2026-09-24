# Modelling log

One entry per step of `_doc/modeling_guide.md`.

### Step 0. Setup and checks
Kind: Diagnostic
Candidates run: Provisional decisions listed (D3, D4, D8, D15, D18, D21, D25); sample check on weeks 1-2 only (counts, event mix, sessions with carts and orders, length and item-frequency distributions, week-over-week shift, duplicates); positives per head against stated minimums; licence rows; environment checks (Docker, AWS, credentials, Bedrock configuration, libraries); experiment table, Bedrock cost log and manual-steps log created. NOT done: the arm64 hello-world image (Docker daemon not running) and the Bedrock availability check (no AWS access).
Observed: Weeks 1-2 hold 13,064,647 events in 1,037,192 sessions; the event-type mix is the same in both weeks (week 2 versus week 1: events +4.0%, click share -0.0%, cart share -0.4%, order share +2.8%). Raw session length looks much shorter in week 2 (median -33.3%, share with a cart -26.6%) but that is censoring: sessions can last for days and week-2 sessions are cut at day 14; with every session limited to its first two days the weeks agree (median length 3 against 3, share with a cart 20.6% against 20.4%, with an order 8.0% against 8.1%). Event rows that have an exact copy (first occurrence included) 39,403. Under the Level 0 split (Step 1) every head has enough positive sessions: train order 83,207; validation order 8,036; test order 9,526 (orders; the minimums are 5,000 training and 2,000 evaluation sessions). Licence: 2 rows still Unclear (publishing derived samples and trained models). Environment: Docker daemon not running, AWS CLI absent, no AWS or Anthropic credentials.
Decision: No oversampling or reweighting is needed (D3). Steps 1 to 8 can proceed locally. Blocked on the human: Docker Desktop (arm64 image), AWS region and Bedrock or API access (Step 9), and the Unclear licence rows (publishing derived data).
Why: Guide step 0: list the decisions, check the sample and the positives, confirm the licence and Bedrock, build the image, create the logs. The positives minimums are stated here because the guide asks for a number. Session-level statistics are compared over a common two-day window because raw session length depends on how long a session can be observed.
Rejected: Oversampling carts and orders: not needed. Publishing any derived data: held back because the licence rows are Unclear. Checking week 3 or 4 for shifts: rejected, they are Level 1 arrivals.
Confidence: High for the sample checks; the environment items are open until the human acts.
Affects later steps: Step 1 fixes the split and the harness; Step 9 cannot start without Bedrock or API access and hand labels; the arm64 image must be built before Step 11.
Needs human input: yes: start Docker Desktop; choose the AWS region and give Bedrock or Anthropic API access; settle the two Unclear licence rows (stop-and-ask trigger in the guide)













### Step 1. Splits and evaluation harness
Kind: Selection
Candidates run: Split rule inside weeks 1-2 (training days 0-9, validation days 10-11, test days 12-13, sessions assigned by their first event, crossing sessions cut at the boundary; the same function for a Level 1 arrival); harness tests against scikit-learn (NDCG@10 on eight toy cases, recall, OTTO weighted recall, bootstrap coverage, paired bootstrap); gain sensitivity with five baselines under four gain sets; harness output with intervals and slices on one method.
Observed: Slices: train 9,060,280 events, 802,267 sessions, 83,207 with an order; validation 661,000 events, 107,467 sessions, 8,036 with an order; test 833,544 events, 127,458 sessions, 9,526 with an order. Tests: 15 passed in 2.25s. Gain sets give the same method ranking (Kendall tau 1.00, 1.00, 1.00); best baseline co-visitation, type weights 1/3/6, decay 0.8 with NDCG@10 0.0934 under the provisional gains. Harness output per method on 20,000 validation queries in the notebook table.
Decision: Split rule recorded as D3 (Provisional) in docs/decisions/D3-split-rule.md; gains frozen at click 1, cart 2, order 3 (docs/decisions/D7-gains.md). Baseline queries here have no category restriction; category-based ranking queries come with Step 2.
Why: Guide step 1: propose day boundaries and a crossing rule, build and verify the harness before any model, freeze the gains after checking that the method ranking is stable.
Rejected: Training on days 0-11 with the test right after: no validation slice would remain for tuning. Days 14-15 as evaluation (first phase): touches week 3, reserved for Level 1. Alternative gain sets: ranking identical, so the simplest set stays.
Confidence: High for the split and the harness (tested); Medium for the gains, because the check used simple baselines and is repeated in Step 3.
Affects later steps: Every later step uses these slices and this harness. Steps 2 and 3 rebuild categories and the co-visitation graph from the training window only.
Needs human input: no












### Step 2. Categories, queries and the benchmark set
Kind: Selection
Candidates run: k-means on the SVD embedding of the training-window co-visitation graph for K = [50, 60, 70, 80, 90, 100] (five seeds each: stability by adjusted Rand index, size balance); Leiden at four resolutions; names by hierarchical alignment with the existing taxonomy; ranking queries for the validation and test slices; the classifier evaluation set remapped to the new clusters; the benchmark set.
Observed: K sweep: K=50 ARI 0.677, largest 6.4%, small 0; K=60 ARI 0.743, largest 3.9%, small 0; K=70 ARI 0.783, largest 4.3%, small 0; K=80 ARI 0.763, largest 3.9%, small 0; K=90 ARI 0.794, largest 3.0%, small 0; K=100 ARI 0.773, largest 2.9%, small 0. Best by the rule K = 90 (ARI 0.794; noise 0.036, the larger pairwise spread of the two); K = 80 is a tie within the noise. Leiden: res 0.02 -> 26 communities, ARI 0.726, largest 100.0%; res 0.1 -> 65 communities, ARI 0.765, largest 72.2%; res 0.5 -> 69 communities, ARI 0.828, largest 25.6%; res 2.0 -> 128 communities, ARI 0.719, largest 7.6%. Categories: 80, sizes 898-9,859, nearest cluster shares the taxonomy group for 52% of clusters (chance 6%). Ranking queries kept: validation 55,721; test 69,545. Classifier set: 2614 queries (dev and held-out parts), all names mapped. Benchmark v2: 300 pairs, sha256 d42ee478220b.
Decision: K = 80 with the existing taxonomy (K is an open item if it is not a tie); categories labelled synthetic; ranking queries built for validation and test (the test queries are built, not scored); classifier set v2 and benchmark v2 versioned. The D5 leak is written down (docs/decisions/D4-categories.md).
Why: Guide step 2: graph from the training window only, SVD plus k-means, K by stability and size balance, quick Leiden comparison, taxonomy alignment by hierarchical order, queries with the leak documented, classifier set with dev and held-out parts, benchmark from the test slice.
Rejected: Other K values (including the nominally most stable K = 90, a tie within the noise): they need a new taxonomy (Claude access). Leiden: the number of communities is not controlled and it does not pair with a fixed taxonomy. Opaque labels: a classifier cannot use them.
Confidence: High for the stability numbers and for K = 80 (a tie within the noise); the human hand-check of the classifier set is still open.
Affects later steps: Step 3 uses the categories for BM25 documents and the category-restricted co-visitation; Steps 4-7 use the ranking queries; Step 9 uses the classifier set; Step 11 scores the benchmark.
Needs human input: no











### Step 3. Baselines
Kind: Selection
Candidates run: Co-visitation (category-restricted, popularity fill; type weights, decay and history length tuned) and BM25 (hand-written, checked against rank_bm25 on 7 toy cases in tests/test_bm25.py; documents = category name tokens plus the names of the 3 most co-visited categories; k1 and b tuned); 12 random configurations plus the default each, on 5,000 tuning queries; scored on 50,721 other validation queries; floors: popularity and random inside the category.
Observed: Scoring part of validation (NDCG@10 [95% interval], recall@20/100/200): co-visitation (tuned) 0.2032 [0.2005, 0.2062], 0.372/0.543/0.606; BM25 (tuned) 0.0003 [0.0002, 0.0004], 0.001/0.005/0.009; popularity inside the category 0.0448 [0.0435, 0.0461], 0.122/0.293/0.397; random inside the category 0.0014 [0.0012, 0.0016], 0.006/0.029/0.057. Best co-visitation parameters {'w_cart': 1.9731935392598379, 'w_order': 6.133931040997376, 'decay': 0.5219018828936144, 'last_n': 50}; best BM25 parameters {'k1': 1.5, 'b': 0.75}. BM25 ties: category 0: 9 distinct score values among 303,159 items; category 20: 9 distinct score values among 303,159 items; category 40: 9 distinct score values among 303,159 items; category 60: 9 distinct score values among 303,159 items. Gain re-check with these baselines: Kendall tau 1.00, 1.00, 1.00 (stable).
Decision: Co-visitation (tuned) is the bar for the learned models: NDCG@10 0.2032, recall@100 0.543. BM25 stays the floor (0.0003). Gains stay frozen.
Why: Guide step 3: co-visitation with event-type weights and decay tuned on validation within the standard budget; BM25 from scratch, checked against a library, with category-name documents; record recall@K and NDCG@10 for both.
Rejected: Popularity fill only (no co-visitation signal): kept as a floor. BM25 on real text: optional side lab, not built (no item text). Wider tuning: the budget is the same modest 12 trials for both, as the guide asks.
Confidence: High: both are scored on the same 50,721 queries with intervals and the difference is far outside noise; Medium for BM25 tuning (its scores tie heavily, so parameters barely matter).
Affects later steps: Steps 4-7 must beat the co-visitation numbers here on the same validation queries; the candidate pool (Step 5) uses co-visitation candidates; the judge and the comparison tab use both baselines.
Needs human input: no










### Step 4. Two-Tower with FAISS
Kind: Selection
Candidates run: Minimum item frequency [1, 2, 5, 10, 20, 50] (catalog size against coverage of relevant validation items); Two-Tower (true category one-hot, recency- and type-weighted session pooling, in-batch sampled softmax with logQ, 302,670 training queries) tuned with the same budget as the other models (the default plus 12 random configurations of learning rate, temperature and batch size, scored on the tuning part), then with and without same-category hard negatives at the best configuration; exact search versus IVF-PQ (nprobe 4, 16, 64) and HNSW (efSearch 32, 64, 128, 256). Training as a script with configs/two_tower.json in its own process.
Observed: Minimum frequency 2: catalog 295,657 items. Tuning: default 0.0483 against best 0.0558 NDCG@10 on the tuning part (all trials 0.0303 to 0.0558), best parameters {'lr': 0.0027950876331863635, 'batch_size': 1024, 'temperature': 0.09943861677685649, 'epochs': 3}. On the same validation sessions as the baselines (co-visitation NDCG@10 0.2032): in-batch negatives + logQ (tuned) NDCG@10 0.0537 [0.0523, 0.0552], recall@20/100/200 0.142/0.307/0.395, difference to co-visitation -0.1495 [-0.1524, -0.1466]; + same-category hard negatives NDCG@10 0.0539 [0.0525, 0.0555], recall@20/100/200 0.143/0.308/0.398, difference to co-visitation -0.1493 [-0.1521, -0.1464]. Hard negatives minus base: NDCG@10 +0.0003 [-0.0003, +0.0008], recall@100 +0.0007 [-0.0005, +0.0020]. ANN: Flat (exact) - recall 1.000, p95 2.95 ms, 76 MB; IVF-PQ nprobe=4 recall 0.876, p95 0.07 ms, 7 MB; IVF-PQ nprobe=16 recall 0.880, p95 0.11 ms, 7 MB; IVF-PQ nprobe=64 recall 0.880, p95 0.42 ms, 7 MB; HNSW efSearch=32 recall 0.819, p95 0.06 ms, 156 MB; HNSW efSearch=64 recall 0.970, p95 0.09 ms, 156 MB; HNSW efSearch=128 recall 0.996, p95 0.16 ms, 156 MB; HNSW efSearch=256 recall 0.999, p95 0.37 ms, 156 MB. Training: 45 s per epoch, a full run about 2.3 minutes.
Decision: Two-Tower variant: in-batch negatives + logQ (tuned). Minimum item frequency 2. Item index: Flat (exact) - (recall 1.000 against exact, p95 2.95 ms, 76 MB); approximate search is not needed at this catalog size. Training time is within the one-hour target, so no spot training is needed for Level 0 (D21 evidence).
Why: Guide step 4: hard negatives only as an ablation (kept if the paired interval excludes zero, else the simpler variant); index chosen by measured recall against exact search, latency and size (exact search included as a candidate); time a full CPU run (D21).
Rejected: Feeding a simulated-confusion category distribution: not done, the true category is used and the real classifier enters in Step 10 (log of the choice). Larger catalogs: coverage gain below 1%. IVF-PQ: 10 times smaller but recall stalls at 0.88; HNSW: recall 0.986 at efSearch 128 but twice the memory of exact search.
Confidence: Medium: one seed and one training run per variant; the paired bootstrap covers the ranking sample, not training variance.
Affects later steps: The candidate pool (Step 5) uses these Two-Tower candidates and the co-visitation candidates; the LightGBM and DCN-V2 features include Two-Tower similarity; Step 11 exports the towers to ONNX and ships the index.
Needs human input: yes: STOP-AND-ASK (guide): the Two-Tower, a live-path component, scores below co-visitation beyond noise as a stand-alone ranker (difference [-0.1524, -0.1466] NDCG@10). Reported, not swapped: Step 5 tests whether it still adds recall in the union pool.









### Step 5. Candidate pool
Kind: Selection
Candidates run: Two-Tower only, co-visitation only, the union by reciprocal-rank fusion (k = 60) and the union by quota (two thirds co-visitation, one third Two-Tower items not already present), each at [50, 100, 200, 300] candidates, on 55,721 validation queries; recall overall and by event type (click, cart, order), build time per query, pool size.
Observed: Recall overall (click / cart / order): Two-Tower only 50: 0.227 (0.228 / 0.222 / 0.203); Two-Tower only 100: 0.308 (0.308 / 0.301 / 0.289); Two-Tower only 200: 0.396 (0.397 / 0.395 / 0.382); Two-Tower only 300: 0.453 (0.454 / 0.447 / 0.432); co-visitation only 50: 0.474 (0.474 / 0.482 / 0.483); co-visitation only 100: 0.544 (0.544 / 0.556 / 0.555); co-visitation only 200: 0.608 (0.608 / 0.623 / 0.601); co-visitation only 300: 0.647 (0.647 / 0.660 / 0.641); union (RRF) 50: 0.381 (0.382 / 0.381 / 0.388); union (RRF) 100: 0.517 (0.518 / 0.519 / 0.509); union (RRF) 200: 0.601 (0.601 / 0.606 / 0.591); union (RRF) 300: 0.644 (0.645 / 0.653 / 0.637); union (quota: 2/3 co-visitation, 1/3 Two-Tower) 50: 0.457 (0.458 / 0.461 / 0.458); union (quota: 2/3 co-visitation, 1/3 Two-Tower) 100: 0.534 (0.535 / 0.540 / 0.538); union (quota: 2/3 co-visitation, 1/3 Two-Tower) 200: 0.604 (0.605 / 0.613 / 0.606); union (quota: 2/3 co-visitation, 1/3 Two-Tower) 300: 0.645 (0.645 / 0.653 / 0.634). Best: co-visitation only at 300 (0.6470). Per-query build time: Two-Tower scoring 0.15 ms, co-visitation 0.16 ms, RRF fusion 1.00 ms. At 300 candidates the quota union has recall 0.6447 against 0.6470 for co-visitation alone (-0.0023).
Decision: Frozen pool (D8): co-visitation only, 300 candidates, recall 0.6470 (+0.0000 from the best), written to artifacts/candidate_pool.json. Both rerankers use it. The frozen pool contains NO Two-Tower candidates: STOP-AND-ASK (guide): a live-path component adds no recall beyond noise; reported, not swapped.
Why: Rule fixed before looking: the best option by overall recall; a pool is within noise if its paired recall difference to the best is at least -0.005; take the smallest such pool.
Rejected: Not chosen (recall, difference to the best): union (quota: 2/3 co-visitation, 1/3 Two-Tower) 300 (0.645, -0.0023); union (RRF) 300 (0.644, -0.0028); co-visitation only 200 (0.608, -0.0395); union (quota: 2/3 co-visitation, 1/3 Two-Tower) 200 (0.604, -0.0429), and the rest of the grid (table).
Confidence: High: recall differences are tested with a paired bootstrap on the same validation sessions.
Affects later steps: Steps 6 and 7 build features and rerank exactly these candidates; the training-query pools (days 8-9) use the days 0-7 Two-Tower and co-visitation; the test slice pools are built in Step 11 with the same rule.
Needs human input: yes: STOP-AND-ASK (guide): the Two-Tower, a live-path component, adds no recall to the pool beyond noise; the pool is frozen without it (co-visitation only) and reported, not swapped silently.








### Step 6. LightGBM LambdaMART
Kind: Selection
Candidates run: LightGBM on the frozen pool (co-visitation only, 300 candidates): lambdarank on graded labels, binary logloss, and lambdarank without the Two-Tower features; each with the default plus 12 random configurations tuned on the 5,000-query tuning part and scored on the other 50,721 validation queries; propensity-weight column defaulting to 1. Features (point in time, D25): rank_covis, rank_tt, log_clicks, log_clicks_3d, cart_rate, order_rate, pop_cat_pct, item_cat_size, hours_since_last_seen, covis_max, covis_mean, covis_wsum, prefix_len, prefix_n_cart, prefix_n_order, prefix_share_same_cat, tt_sim. Trained on 29,681 queries from days 8-9.
Observed: On the scoring part (NDCG@10 [95% interval], difference to co-visitation [interval]): lambdarank, without Two-Tower features 0.2054 [0.2028, 0.2083], +0.0022 [+0.0014, +0.0031]; lambdarank, all features 0.2054 [0.2027, 0.2084], +0.0022 [+0.0012, +0.0032]; binary logloss, all features 0.2052 [0.2027, 0.2081], +0.0020 [+0.0011, +0.0029]. Lambdarank minus binary +0.0001 [-0.0006, +0.0009]; with minus without Two-Tower features -0.0001 [-0.0009, +0.0007]. All-ones propensity weights reproduce the unweighted scores exactly (max difference 0.0). Top features by gain: rank_covis 36%, covis_wsum 20%, covis_mean 18%, prefix_share_same_cat 6%.
Decision: Best LightGBM variant: lambdarank, without Two-Tower features (NDCG@10 0.2054, +0.0022 against co-visitation [+0.0014, +0.0031]). Simplicity rule: lambdarank over binary unless the interval of the difference is entirely below zero; the Two-Tower features are kept only if the interval of their gain excludes zero.
Why: Guide step 6: D11 features point in time; lambdarank on graded labels; ablations (binary logloss, with and without Two-Tower similarity); propensity column defaulting to 1 with an exactness check; tuning within the standard budget.
Rejected: Variants whose paired interval against the winner includes zero are ties and go to the simpler one; the rest are reported in the table.
Confidence: Medium: one training seed; intervals cover the sampling of validation queries, not training variance.
Affects later steps: Step 7 uses the same features and the same pool; Step 10 compares this model with co-visitation, Two-Tower and DCN-V2 on the same queries; the propensity column is used in Step 8.
Needs human input: no







### Step 7. DCN-V2 with MMoE
Kind: Selection
Candidates run: DCN-V2 cross layers plus MMoE with click, cart and order towers on the frozen pool and the Step 6 features; families: MMoE, shared-bottom multi-task, single-task (one model per head), each with 6 random configurations (cross layers 1-3, experts 2-8, expert width, learning rate, task loss weights 1/1/1 or 1/2/4) tuned on the tuning part; blend weights: OTTO 0.10/0.30/0.60 against a simplex grid tuned on the tuning part; scored on the same 50,721 validation queries.
Observed: Scoring part (NDCG@10 [95% interval]; difference to co-visitation; difference to the best LightGBM): MMoE / OTTO weights 0.2012 [0.1984, 0.2040], -0.0020, -0.0042 [-0.0054, -0.0031]; MMoE / tuned weights 0.2031 [0.2002, 0.2059], -0.0001, -0.0024 [-0.0035, -0.0014]; shared-bottom multi-task / OTTO weights 0.2022 [0.1993, 0.2051], -0.0010, -0.0033 [-0.0044, -0.0023]; shared-bottom multi-task / tuned weights 0.2021 [0.1993, 0.2050], -0.0011, -0.0033 [-0.0044, -0.0023]; single-task (one model per head) / OTTO weights 0.2033 [0.2003, 0.2062], +0.0001, -0.0022 [-0.0033, -0.0011]; single-task (one model per head) / tuned weights 0.2040 [0.2011, 0.2069], +0.0008, -0.0014 [-0.0025, -0.0005]. MMoE minus shared-bottom +0.0010 [+0.0001, +0.0018]; MMoE minus single-task -0.0010 [-0.0018, -0.0001]; tuned minus OTTO blend +0.0018 [+0.0013, +0.0023]. Order-head AUC MMoE minus single-task +0.0553. Training: 17 s per epoch, 1.1 minutes per run.
Decision: Best MMoE: tuned weights (NDCG@10 0.2031; against LightGBM -0.0024 [-0.0035, -0.0014]). STOP-AND-ASK: the live-path component DCN-V2 with MMoE loses to LightGBM beyond noise; reported, not swapped. Blend weights: keep OTTO's unless the tuned weights beat them beyond noise. CPU training is far below the one-hour target (D21, docs/decisions/D21-training-compute.md).
Why: Guide step 7: cross layers, experts, task weights and blend weights compared with the same budget; ablations (single-task, shared-bottom); watch for seesaw effects on orders (PLE is the follow-up, not built); measure CPU training time.
Rejected: Any variant whose paired interval against the MMoE includes zero is a tie and the simpler one wins on the ladder; PLE is not built.
Confidence: Medium: one training seed per configuration and six configurations per family.
Affects later steps: Step 10 compares DCN-V2 with the other four methods on the same queries; Step 11 exports it to ONNX with a parity check; Step 8 reuses the propensity-weight argument of `train_model`.
Needs human input: yes: STOP-AND-ASK (guide): DCN-V2 with MMoE, a live-path component, loses to LightGBM beyond noise (difference [-0.0035, -0.0014] NDCG@10). Reported, not swapped.






### Step 8. Position-bias study
Kind: Selection
Candidates run: Semi-synthetic simulator (logging policy: popularity plus Gaussian noise, top 20 shown; click probability = position examination probability (1/k)^eta times true relevance 0.02/0.5/0.7/0.9 for labels 0-3 from held-out behaviour); naive click model against IPW with clipping (M = 1 naive, 2, 5, 10, 20, none) and self-normalisation; bias strength eta in [0.5, 1.0, 2.0]; 5 seeds each; LightGBM binary click model on 17 features; true NDCG@10 on 10,000 validation queries against the known relevance; simulator check against the known propensities; weight-column exactness checks (LightGBM, PyTorch).
Observed: Simulator: implied propensity equals (1/k)^eta within 12.0% for positions 1-10. References (true NDCG@10): random order 0.3960, logging policy 0.4294, model trained on real labels of all candidates 0.5646. Naive click model by eta: 0.5 0.5133, 1.0 0.4692, 2.0 0.4520. IPW against naive (mean difference, paired 95% interval over seeds), best setting per eta: eta 0.5: clip 2 -0.0065 [-0.0114, -0.0015]; eta 1.0: clip 2 +0.0030 [-0.0042, +0.0101]; eta 2.0: clip 2 +0.0011 [-0.0071, +0.0093]. Settings whose interval excludes zero: 0 of 15. Figure: modeling/figures/step8_bias_variance.png.
Decision: Real OTTO labels stay unweighted in the main models (D15). The propensity column defaults to 1 and is verified exact in LightGBM (Step 6) and in the PyTorch loss and training path. IPW is kept as supported machinery, not used in the main pipeline.
Why: Guide step 8: simulate with known propensities, compare naive with IPW (clipping, self-normalisation), sweep bias strength, clipping and seeds, plot bias against variance, keep real labels unweighted, then ask the human.
Rejected: Training the main pipeline on simulated exposure: not done (asked below). Estimating propensities from data: out of scope (no positions exist in OTTO).
Confidence: Medium: five seeds per cell; intervals are t-intervals over seeds and small samples widen them.
Affects later steps: Nothing in Steps 9-11 changes; the propensity column stays in the feature tables and the training scripts.
Needs human input: yes: the guide asks after this step whether the main pipeline should train on simulated exposure. Recommendation: no, keep real OTTO labels unweighted. In this simulation IPW never beat the naive click model beyond noise, so there is nothing to gain even where the bias is known, and OTTO has no real exposure data to weight with.





### Step 9. Claude roles
Kind: Selection
Candidates run: Done locally (no model call): pass criteria fixed before evaluation (docs/decisions/D17-D19-role-criteria.md); TF-IDF prototype classifier baseline on the held-out classifier set; explainer numeric payload, templated fallback and faithfulness test with tests; DCN-V2 attribution comparison (leave-one-feature-group-out against integrated gradients on 1,000 top results); judge statistics (weighted kappa, Spearman, position and verbosity checks) with tests; a 180-row hand-labelling sheet. NOT done (blocked): every Claude call (classifier, explainer, judge), calibration on the dev part, the sentence-embedding baseline, kappa against hand labels, Bedrock availability and cost.
Observed: Classifier baseline on 2077 held-out queries: top-1 0.764, top-3 0.843, expected calibration error 0.038, out-of-scope precision 0.618, recall 0.635, F1 0.626 (temperature 0.05, threshold 0.20); so the Claude classifier must reach top-3 of at least 0.893 and top-1 of at least 0.814. Attribution: removal-test lift LOGO 1.81, integrated gradients 1.79; IG completeness error 0.045; agreement (Spearman) 0.81; stability under noise LOGO 0.99, IG 0.99; latency per result LOGO 0.03 ms, IG 0.85 ms. Faithfulness and judge-statistics tests pass.
Decision: Attribution method for the explainer: leave-one-group-out (rule fixed in advance). Pass criteria for the three roles are fixed. The Claude evaluations cannot run here: they stay open (issue #131) and the classifier, explainer and judge are unresolved.
Why: Guide step 9 order: criteria first, baselines, faithfulness test with fixtures and templated fallback, attribution comparison, judge preparation. Everything that needs no model call was done so that the evaluations can run in one pass once access exists.
Rejected: Running the LLM roles with a simulated client: rejected, it would produce numbers that mean nothing. Building the judge calls before the labelling sheet is labelled: rejected, calibration against the human is the point.
Confidence: High for what was done; the LLM decisions are open, not low-confidence.
Affects later steps: Step 10 uses the TF-IDF classifier as a stand-in to measure how category errors propagate through the live path (labelled as a stand-in); judge scores and the NDCG-versus-judge disagreement analysis wait for API access and the human labels.
Needs human input: yes: (1) AWS region and Bedrock or Anthropic API access, (2) the human hand-labels the sheet modeling/artifacts/judge_labeling_sheet.csv (180 rows, score 0-4), (3) confirm the classifier pass criteria in docs/decisions/D17-D19-role-criteria.md




### Step 10. Comparison and analysis
Kind: Diagnostic
Candidates run: Five methods scored on the same 50,721 validation sessions with the same metric code (NDCG@10 with 95% bootstrap intervals, recall@20/50/100/200, OTTO weighted recall@20); paired bootstrap between methods and the ladder tie rule; slices by prefix length, popularity of the held-out click and category; popularity bias of the top 10; the ablations of Steps 4-8 in one table; the live path with a stand-in classifier (TF-IDF, not Claude) on 2,000 queries; latency and sizes. NOT done (blocked): judge scores on the benchmark (the benchmark lives in the test slice, so it waits for Step 11 anyway, and the judge needs API access), the NDCG-versus-judge disagreement analysis, the live path with the real Claude classifier, Bedrock cost per query.
Observed: NDCG@10 [95% interval]: BM25 0.0003 [0.0002, 0.0004]; co-visitation 0.2032 [0.2003, 0.2060]; Two-Tower 0.0537 [0.0523, 0.0552]; LightGBM LambdaMART 0.2054 [0.2025, 0.2083]; DCN-V2 + MMoE 0.2031 [0.2002, 0.2059]. Paired differences: LightGBM minus co-visitation +0.0022 [+0.0014, +0.0031]; DCN-V2 minus LightGBM -0.0024 [-0.0035, -0.0014]; Two-Tower minus co-visitation -0.1495 [-0.1524, -0.1466]. Best LightGBM LambdaMART; tied with it: LightGBM LambdaMART; winner by the ladder: LightGBM LambdaMART. Slices (NDCG@10, co-visitation / LightGBM): popularity of the held-out click head (top 1,000): 0.363 / 0.354; popularity of the held-out click tail or unseen: 0.157 / 0.160; popularity of the held-out click torso (1,000-20,000): 0.220 / 0.224; prefix length 1: 0.220 / 0.221; prefix length 11+: 0.152 / 0.156; prefix length 2-3: 0.201 / 0.203; prefix length 4-10: 0.187 / 0.191. Popularity bias: the share of top-1,000 popular items is 0.12 among relevant items and BM25 0.00, co-visitation 0.38, Two-Tower 0.48, LightGBM LambdaMART 0.36, DCN-V2 + MMoE 0.39 in the top 10. Live path with the TF-IDF stand-in classifier (78.5% correct): LightGBM path NDCG@10 0.2163 with the true category, 0.1654 with the predicted one; queries with a wrong category score 0.0000. Latency per query: candidates 0.16 ms, features 1.9 ms, LightGBM 0.5 ms, DCN-V2 0.5 ms (p95, one thread).
Decision: Report: LightGBM LambdaMART is the method to beat on the ladder. Two-Tower loses to co-visitation beyond noise; DCN-V2 with MMoE loses to LightGBM beyond noise. No component is swapped silently; stop-and-ask items for live-path components are listed under 'needs human input'.
Why: Guide step 10: score all five methods with intervals, recall@K for the retrieval-style ones, slices, popularity bias, the ablation tables, the live path with a classifier, the cost measures, an honest report.
Rejected: Judge-based comparison and the disagreement analysis: blocked, not simulated. Scoring the test slice here: rejected, it is used once in Step 11.
Confidence: High for the ranking of methods (paired intervals on 50,000+ sessions); Medium for the stand-in live-path result (a TF-IDF classifier is not the Claude classifier).
Affects later steps: Step 11 freezes the models and scores the test slice once with the same code; the disagreement analysis and the judge columns of the frontend exports stay empty until API access exists.
Needs human input: yes: STOP-AND-ASK (guide): Two-Tower loses to co-visitation beyond noise; DCN-V2 with MMoE loses to LightGBM beyond noise. Reported, not swapped; a decision is needed on whether the live path keeps these components.



### Step 11. Freeze and Level 0 exit
Kind: Diagnostic
Candidates run: Expected test range written from validation (mean +/- 2 bootstrap standard deviations) before scoring; pipeline frozen (docs and artifacts/frozen_config.json); test slice scored once for all five methods (guard file); bundle assembled (LightGBM, DCN-V2 and Two-Tower weights, ONNX exports of DCN-V2 and the Two-Tower query tower with parity check, exact item index, co-visitation matrix, item feature tables as of day 14, category map, pool spec, prompts and configuration, manifest with hashes); feature parity test between the training path and the serving path rebuilt from the bundle; aggregate frontend exports. NOT done (blocked): the manual upload of the bundle to S3 (no AWS access) and the judge scores for the frontend.
Observed: Test slice (69,545 queries, scored once) NDCG@10 [95% interval] against the expected range: BM25 0.0003 [0.0002, 0.0004] (in the range 0.0002-0.0004); co-visitation 0.1968 [0.1946, 0.1991] (OUTSIDE the range 0.2003-0.2061); Two-Tower 0.0507 [0.0495, 0.0520] (OUTSIDE the range 0.0522-0.0552); LightGBM LambdaMART 0.2003 [0.1981, 0.2026] (OUTSIDE the range 0.2025-0.2084); DCN-V2 + MMoE 0.1980 [0.1958, 0.2004] (OUTSIDE the range 0.2001-0.2060). Paired differences on test: LightGBM LambdaMART minus co-visitation +0.0035 [+0.0028, +0.0043]; DCN-V2 + MMoE minus LightGBM LambdaMART -0.0023 [-0.0032, -0.0014]; Two-Tower minus co-visitation -0.1461 [-0.1484, -0.1437]; DCN-V2 + MMoE minus co-visitation +0.0012 [+0.0002, +0.0022]. ONNX parity: DCN-V2 3.81e-06, Two-Tower query tower 1.19e-07 (limit 1e-4). Feature parity: 5,200 feature columns on 200 fixture queries, 0 differ. Bundle: 20 files, 501 MB. Benchmark mean NDCG@10: BM25 0.0000, co-visitation 0.2127, Two-Tower 0.0587, LightGBM LambdaMART 0.2120, DCN-V2 + MMoE 0.2024.
Decision: Frozen and scored once; the test slice is now spent for these models: any change from here on needs a new slice. The bundle is ready for the manual S3 upload; the manifest records hashes, model versions, feature schema hash, snapshot id and commit SHA. Methods outside the expected range: co-visitation, Two-Tower, LightGBM LambdaMART, DCN-V2 + MMoE: nothing was tuned on the test slice; Investigation (observational, nothing re-scored): every method with a signal is 0.0030 to 0.0064 below its validation value, the range was built from session sampling error alone, and the daily co-visitation NDCG@10 moves between 0.1958 and 0.2054 across the four days (spread 0.0095, larger than the sampling interval); the ordering of the methods and the paired gaps are unchanged (LightGBM minus co-visitation +0.0035 on test against +0.0022 on validation). The shift is a day-level level effect, possibly with some staleness of the fixed day 0-9 matrix; the data cannot separate the two. It is not a defect of the pipeline, and the range should be widened by day-to-day spread in Level 1.
Why: Guide step 11: expected range first, then one scoring; bundle with ONNX parity; feature parity test; upload by hand and time it; frontend exports.
Rejected: Re-scoring or tuning after the test result: rejected, it would spend the slice. Committing per-query exports: rejected, the licence for derived data is Unclear.
Confidence: High for the numbers (paired intervals on the full test slice) and for the parity checks; the S3 upload is open.
Affects later steps: Step 12 summarises; Level 1 starts from this bundle, the split rule (arrival_slice), the frozen gains and the needs-review list.
Needs human input: yes: AWS access to upload the bundle to S3 by hand and time it (manual-steps log), plus the open stop-and-ask items listed in the summary

### Step 12. Summary
Kind: Diagnostic
Candidates run: modeling/model_summary.md written from the log, the tables and the test result: approach, decision table, five-method table, ablations, position-bias findings, Claude role evaluations (blocked), limits, cost, settled Provisional decisions, needs-review list, Level 1 notes; final checks.
Observed: Summary written (14,866 characters). Test slice used once (69,545 sessions). Experiments table: 163 rows. Needs-review items: 8 steps with an open human decision. Best on validation and on test: LightGBM LambdaMART / LightGBM LambdaMART.
Decision: Level 0 modelling is complete except the items that need the human (AWS access, Docker, hand labels, stop-and-ask decisions on Two-Tower and DCN-V2, S3 upload).
Why: Guide step 12: one document a reader can use without opening the notebook, with the limits stated plainly.
Rejected: A separate results table per step in the summary: rejected, the decision table and the five-method table carry it and the log keeps the detail.
Confidence: High for the numbers (read from files); the open items are decisions, not doubts about the results.
Affects later steps: Level 1 starts from the bundle, the split rule and the needs-review list.
Needs human input: yes: the needs-review list in modeling/model_summary.md (stop-and-ask decisions, AWS access, Docker, hand labels, licence rows)
