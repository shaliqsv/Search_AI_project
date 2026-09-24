# Level 0 modelling summary

Written from `modeling/model_log.md` (one entry per step), `modeling/experiments.csv` (163 rows), the tables in `modeling/tables/` and `notebooks/MODELING.ipynb`. GitHub issues #121 (umbrella) and #122-#134 carry the same trail. Every result is on the synthetic-category ranking task described under Limits.

## 1. Approach

Level 0 builds and compares five ways of ranking items for a session prefix and a category query: BM25 (floor), item co-visitation (the bar), a Two-Tower retrieval model with FAISS, a LightGBM LambdaMART reranker and a DCN-V2 with multi-gate mixture-of-experts reranker. Everything lives inside weeks 1-2 (train days 0-9, validation 10-11, test 12-13, D3); weeks 3 and 4 are untouched and reserved for Level 1. The metric is NDCG@10 with graded gains click 1, cart 2, order 3 (D7), 95% bootstrap intervals over sessions and paired bootstraps for every comparison; ties go to the simpler method on the ladder co-visitation, LightGBM, Two-Tower, DCN-V2. Rerankers share one frozen candidate pool (D8). Each method had the same bounded tuning budget (12 trials; 6 for the neural models). The test slice was scored once, in Step 11, after the expected range had been written from validation and the pipeline had been frozen. The test slice is now spent for these models; any change needs a new slice.

## 2. Decision table

| step | what was compared | winner | confidence |
|---|---|---|---|
| 0 | Sample, licence, environment | sample is healthy; two licence rows Unclear; Docker and AWS pending | High for the sample checks |
| 1 | Split rules (train 0-9, validation 10-11, test 12-13), gain sets | time split D3; gains 1/2/3 (D7); same method order under 4 gain sets | High for the split and the harness (tested) |
| 2 | K = 50-100, k-means vs Leiden | K = 80 k-means with taxonomy (tie with K = 90 within noise) | High for the stability numbers and for K = 80 (a tie within the noise) |
| 3 | Co-visitation, BM25, popularity, random | co-visitation is the bar (validation NDCG@10 0.2032); BM25 is the floor | High |
| 4 | Two-Tower variants, catalog size, ANN (exact, IVF-PQ, HNSW) | in-batch + logQ; exact search; loses to co-visitation | Medium |
| 5 | Pool composition: Two-Tower, co-visitation, RRF union, quota union | co-visitation only, 300 candidates (Two-Tower adds no recall) | High |
| 6 | LambdaRank vs binary; with and without Two-Tower features; 12 trials | LambdaRank without Two-Tower features; +0.0022 over co-visitation on validation, +0.0035 on test | Medium |
| 7 | MMoE vs shared-bottom vs single-task; blend weights; 6 trials | MMoE with tuned weights; loses to LightGBM beyond noise | Medium |
| 8 | Naive vs IPW (clip, self-normalisation), 3 bias strengths, 5 seeds | real labels unweighted; IPW gave no gain beyond noise | Medium |
| 9 | Classifier baseline, attribution methods, judge statistics | attribution: leave-one-group-out; Claude evaluations blocked | High for what was done |
| 10 | Five methods, slices, ablations, live path with stand-in classifier | best on validation: LightGBM LambdaMART; ladder winner LightGBM | High for the ranking of methods (paired intervals on 50,000+ sessions) |
| 11 | Freeze, one test scoring, bundle, parity | test scored once; best on test: LightGBM LambdaMART | High for the numbers (paired intervals on the full test slice) and for the parity checks |

## 3. Five-method table (same sessions, same metric code)

Validation: 5 methods scored on the same validation sessions; test: 69,545 sessions, scored once.

| method | kind | validation NDCG@10 | validation 95% interval | recall20 | recall100 | recall200 | test NDCG@10 | test 95% interval | in expected range |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | retrieval-style | 0.0003 | [0.0002, 0.0004] | 0.0014 | 0.0051 | 0.0088 | 0.0003 | [0.0002, 0.0004] | True |
| co-visitation | retrieval-style | 0.2032 | [0.2003, 0.2060] | 0.3720 | 0.5430 | 0.6064 | 0.1968 | [0.1946, 0.1991] | False |
| Two-Tower | retrieval-style | 0.0537 | [0.0523, 0.0552] | 0.1424 | 0.3071 | 0.3954 | 0.0507 | [0.0495, 0.0520] | False |
| LightGBM LambdaMART | reranker (frozen pool) | 0.2054 | [0.2025, 0.2083] | 0.3769 | 0.5463 | 0.6100 | 0.2003 | [0.1981, 0.2026] | False |
| DCN-V2 + MMoE | reranker (frozen pool) | 0.2031 | [0.2002, 0.2059] | 0.3777 | 0.5467 | 0.6090 | 0.1980 | [0.1958, 0.2004] | False |

Paired differences on the test slice (NDCG@10, 95% interval): LightGBM minus co-visitation +0.0035 [+0.0028, +0.0043]; DCN-V2 minus LightGBM -0.0023 [-0.0032, -0.0014]; Two-Tower minus co-visitation -0.1461 [-0.1484, -0.1437]. The ordering seen on validation held on the test slice. All four methods with a signal scored below the range predicted from validation; the investigation in Step 11 traces this to a day-level shift (the daily co-visitation NDCG@10 varies by more than the sampling interval), not to a defect, and the paired gaps are unchanged.

In one sentence: nothing beats co-visitation by much. LightGBM adds about one percent relative; the two neural models do not pay for their complexity in this setting, and the Two-Tower adds no recall to the pool.

## 4. Ablations

| step | ablation | ndcg10 | difference |
|---|---|---|---|
| 4 | in-batch negatives + logQ (tuned) | 0.0537 | -0.1495 vs co-visitation [-0.1524, -0.1466] |
| 4 | + same-category hard negatives | 0.0539 | -0.1493 vs co-visitation [-0.1521, -0.1464] |
| 5 | pool: Two-Tower only, 300 candidates |  | recall 0.4527 (-0.1943 vs best) |
| 5 | pool: co-visitation only, 300 candidates |  | recall 0.6470 (+0.0000 vs best) |
| 5 | pool: union (RRF), 300 candidates |  | recall 0.6442 (-0.0028 vs best) |
| 5 | pool: union (quota: 2/3 co-visitation, 1/3 Two-Tower), 300 candidates |  | recall 0.6447 (-0.0023 vs best) |
| 6 | lambdarank, without Two-Tower features | 0.2054 | +0.0022 vs co-visitation [+0.0014, +0.0031] |
| 6 | lambdarank, all features | 0.2054 | +0.0022 vs co-visitation [+0.0012, +0.0032] |
| 6 | binary logloss, all features | 0.2052 | +0.0020 vs co-visitation [+0.0011, +0.0029] |
| 7 | MMoE / OTTO weights | 0.2012 | -0.0042 vs LightGBM [-0.0054, -0.0031] |
| 7 | MMoE / tuned weights | 0.2031 | -0.0024 vs LightGBM [-0.0035, -0.0014] |
| 7 | shared-bottom multi-task / OTTO weights | 0.2022 | -0.0033 vs LightGBM [-0.0044, -0.0023] |
| 7 | shared-bottom multi-task / tuned weights | 0.2021 | -0.0033 vs LightGBM [-0.0044, -0.0023] |
| 7 | single-task (one model per head) / OTTO weights | 0.2033 | -0.0022 vs LightGBM [-0.0033, -0.0011] |
| 7 | single-task (one model per head) / tuned weights | 0.2040 | -0.0014 vs LightGBM [-0.0025, -0.0005] |
| 8 | click model, eta 0.5, clip 1 (naive) | 0.5133 | +0.0000 vs naive |
| 8 | click model, eta 0.5, clip 2 | 0.5069 | -0.0065 vs naive |
| 8 | click model, eta 0.5, clip 5 | 0.4911 | -0.0222 vs naive |
| 8 | click model, eta 0.5, clip 10 | 0.4911 | -0.0222 vs naive |
| 8 | click model, eta 0.5, clip 20 | 0.4911 | -0.0222 vs naive |
| 8 | click model, eta 0.5, clip none | 0.4911 | -0.0222 vs naive |
| 8 | click model, eta 1.0, clip 1 (naive) | 0.4692 | +0.0000 vs naive |
| 8 | click model, eta 1.0, clip 2 | 0.4722 | +0.0030 vs naive |
| 8 | click model, eta 1.0, clip 5 | 0.4655 | -0.0037 vs naive |
| 8 | click model, eta 1.0, clip 10 | 0.4590 | -0.0102 vs naive |
| 8 | click model, eta 1.0, clip 20 | 0.4520 | -0.0172 vs naive |
| 8 | click model, eta 1.0, clip none | 0.4520 | -0.0172 vs naive |
| 8 | click model, eta 2.0, clip 1 (naive) | 0.4520 | +0.0000 vs naive |
| 8 | click model, eta 2.0, clip 2 | 0.4531 | +0.0011 vs naive |
| 8 | click model, eta 2.0, clip 5 | 0.4492 | -0.0028 vs naive |
| 8 | click model, eta 2.0, clip 10 | 0.4444 | -0.0076 vs naive |
| 8 | click model, eta 2.0, clip 20 | 0.4453 | -0.0067 vs naive |
| 8 | click model, eta 2.0, clip none | 0.4314 | -0.0206 vs naive |

## 5. Position-bias findings (Step 8)

A simulation with known propensities (examination probability (1/k)^eta): the naive click model loses true NDCG@10 as the bias strength rises, and inverse-propensity weighting did not repair it beyond noise (paired intervals over five seeds include zero at eta 1.0 and 2.0; at eta 0.5 every clipping level did worse than naive). The data have no real exposure, so the main pipeline keeps real labels unweighted; IPW stays as supported machinery (propensity column, verified exact in LightGBM and PyTorch). Human question (train on simulated exposure?): recommendation no.

| eta | clip | mean_ndcg_true | diff_to_naive | diff_low | diff_high |
|---|---|---|---|---|---|
| 0.5000 | 1 (naive) | 0.5133 | 0.0000 | 0.0000 | 0.0000 |
| 0.5000 | 2 | 0.5069 | -0.0065 | -0.0114 | -0.0015 |
| 0.5000 | 5 | 0.4911 | -0.0222 | -0.0306 | -0.0138 |
| 1.0000 | 1 (naive) | 0.4692 | 0.0000 | 0.0000 | 0.0000 |
| 1.0000 | 2 | 0.4722 | 0.0030 | -0.0042 | 0.0101 |
| 1.0000 | 5 | 0.4655 | -0.0037 | -0.0149 | 0.0075 |
| 2.0000 | 1 (naive) | 0.4520 | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 2 | 0.4531 | 0.0011 | -0.0071 | 0.0093 |
| 2.0000 | 5 | 0.4492 | -0.0028 | -0.0142 | 0.0085 |

## 6. Claude role evaluations (Step 9)

Blocked: no AWS or Anthropic access on this machine, so every Claude call (classifier, explainer, judge) is unrun and unscored, and Bedrock cost is unmeasured (`modeling/bedrock_costs.csv` is empty). Done without a model: pass criteria fixed in `docs/decisions/D17-D19-role-criteria.md`; a TF-IDF prototype classifier baseline (top-1 0.764, top-3 0.843, calibration error 0.038 on 2,077 held-out queries; the Claude classifier must reach top-3 0.893 and top-1 0.814); leave-one-group-out attribution chosen over integrated gradients by the pre-stated rule; the explainer's numeric payload, templated fallback and faithfulness test; judge statistics with tests; a 180-row hand-labelling sheet. The live path with the TF-IDF stand-in classifier shows that a wrong category costs a lot (Step 10).

## 7. Limits, stated plainly

- **Categories are synthetic.** OTTO has no text, queries or categories: 80 clusters from co-visitation, named from a fictional taxonomy. Every number is on that constructed task.
- **The query category is a documented leak (D5).** The ranking query's category is derived from the held-out items, so the numbers with the true category are an upper bound; the stand-in classifier run measures the drop when the category is predicted.
- **The judge measures coherence, not relevance to a real need.** It scores whether results fit the category and prefix, which is what a synthetic category can support; it is not a proxy for buyer satisfaction, and it is unrun.
- BM25 has no text to work with, so it is a floor, not a competitor.
- One training seed per neural configuration; intervals cover the sampling of sessions, not training variance or day-to-day shift.
- Licence for OTTO-derived data is Unclear: no per-session or derived data is committed; only aggregates.

## 8. Cost spent

Cloud spend: none (everything trained on the Mac CPU, D21: Two-Tower about 2.3 minutes, DCN-V2 about 1.1 minutes). Bedrock: not used ($0, calls blocked). Latency and size per component: see below.

| component | ms_per_query | size_mb |
|---|---|---|
| co-visitation candidates (pool) | 0.1593 | 105.2486 |
| point-in-time features (shared implementation) | 1.8508 |  |
| LightGBM scoring of the pool | 0.5222 | 0.5514 |
| DCN-V2 + MMoE scoring of the pool (1 thread, p95) | 0.5186 | 0.1011 |
| Two-Tower query tower and exact item search | 3.0973 | 227.2641 |
| Bedrock classifier and explainer (cost per query) |  |  |

## 9. Provisional decisions settled

D3 split rule (`docs/decisions/D3-split-rule.md`, still Provisional until Level 1 reuses it), D4 categories (`docs/decisions/D4-categories.md`, accepted for Level 0), D7 gains (`docs/decisions/D7-gains.md`, accepted), D8 pool (frozen in `artifacts/candidate_pool.json`, Step 5), D15 position bias (Step 8, unweighted), D17-D19 role criteria (`docs/decisions/D17-D19-role-criteria.md`, criteria fixed, evaluations open), D21 training compute (`docs/decisions/D21-training-compute.md`, accepted).

## 10. Needs human review

- **Step 0 (Setup and checks):** yes: start Docker Desktop; choose the AWS region and give Bedrock or Anthropic API access; settle the two Unclear licence rows (stop-and-ask trigger in the guide)
- **Step 4 (Two-Tower with FAISS):** yes: STOP-AND-ASK (guide): the Two-Tower, a live-path component, scores below co-visitation beyond noise as a stand-alone ranker (difference [-0.1524, -0.1466] NDCG@10). Reported, not swapped: Step 5 tests whether it still adds recall in the union pool.
- **Step 5 (Candidate pool):** yes: STOP-AND-ASK (guide): the Two-Tower, a live-path component, adds no recall to the pool beyond noise; the pool is frozen without it (co-visitation only) and reported, not swapped silently.
- **Step 7 (DCN-V2 with MMoE):** yes: STOP-AND-ASK (guide): DCN-V2 with MMoE, a live-path component, loses to LightGBM beyond noise (difference [-0.0035, -0.0014] NDCG@10). Reported, not swapped.
- **Step 8 (Position-bias study):** yes: the guide asks after this step whether the main pipeline should train on simulated exposure. Recommendation: no, keep real OTTO labels unweighted. In this simulation IPW never beat the naive click model beyond noise, so there is nothing to gain even where the bias is known, and OTTO has no real exposure data to weight with.
- **Step 9 (Claude roles):** yes: (1) AWS region and Bedrock or Anthropic API access, (2) the human hand-labels the sheet modeling/artifacts/judge_labeling_sheet.csv (180 rows, score 0-4), (3) confirm the classifier pass criteria in docs/decisions/D17-D19-role-criteria.md
- **Step 10 (Comparison and analysis):** yes: STOP-AND-ASK (guide): Two-Tower loses to co-visitation beyond noise; DCN-V2 with MMoE loses to LightGBM beyond noise. Reported, not swapped; a decision is needed on whether the live path keeps these components.
- **Step 11 (Freeze and Level 0 exit):** yes: AWS access to upload the bundle to S3 by hand and time it (manual-steps log), plus the open stop-and-ask items listed in the summary

## 11. What Level 1 needs to know

- Start from `data/modeling/bundle_v1` (manifest `modeling/artifacts/bundle_manifest.json`: hashes, feature schema hash, snapshot id, commit SHA) and the arrival rule `arrival_slice` in `src/ranking/data/split.py` (weeks 3-4 arrive in slices; nothing in Level 0 read them).
- Point-in-time features come from one shared implementation (`AsOf`), and the parity test passed (0 differences on 5,200 feature columns), so serving can reuse it.
- The expected test range should include day-to-day spread, not only sampling error (Step 11 investigation).
- The live path today is co-visitation candidates plus LightGBM; Two-Tower and DCN-V2 are decisions for the human (stop-and-ask items above).
- Blocked items: AWS region and access, Docker Desktop, S3 upload of the bundle, Claude evaluations, hand labels.
