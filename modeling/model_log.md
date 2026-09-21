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
Observed: Slices: train 9,060,280 events, 802,267 sessions, 83,207 with an order; validation 661,000 events, 107,467 sessions, 8,036 with an order; test 833,544 events, 127,458 sessions, 9,526 with an order. Tests: 15 passed in 2.48s. Gain sets give the same method ranking (Kendall tau 1.00, 1.00, 1.00); best baseline co-visitation, type weights 1/3/6, decay 0.8 with NDCG@10 0.0934 under the provisional gains. Harness output per method on 20,000 validation queries in the notebook table.
Decision: Split rule recorded as D3 (Provisional) in docs/decisions/D3-split-rule.md; gains frozen at click 1, cart 2, order 3 (docs/decisions/D7-gains.md). Baseline queries here have no category restriction; category-based ranking queries come with Step 2.
Why: Guide step 1: propose day boundaries and a crossing rule, build and verify the harness before any model, freeze the gains after checking that the method ranking is stable.
Rejected: Training on days 0-11 with the test right after: no validation slice would remain for tuning. Days 14-15 as evaluation (first phase): touches week 3, reserved for Level 1. Alternative gain sets: ranking identical, so the simplest set stays.
Confidence: High for the split and the harness (tested); Medium for the gains, because the check used simple baselines and is repeated in Step 3.
Affects later steps: Every later step uses these slices and this harness. Steps 2 and 3 rebuild categories and the co-visitation graph from the training window only.
Needs human input: no
