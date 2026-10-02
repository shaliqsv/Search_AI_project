# D3: Reopen D2 — build all five ranking methods, pivot to recsys framing

**Phase**: 6 (Modeling), but also reopens Phase 0-1 framing. Supersedes `D2-phase6-scope.md`'s trim.

**Decision**: Build all five methods from the archived plan's D8 (BM25, co-visitation, LightGBM, Two-Tower, DCN-V2+MMoE), not just co-visitation + LightGBM. This is bundled with two other changes the human has been working toward in conversation, not yet formally recorded:

1. **Recsys framing, not search.** Drop the synthetic "query" construct (`query_category`-as-intent-proxy). Categories remain as item features, not as the thing that defines a correct answer.
2. **Real competition format, aimed at an actual Kaggle submission.** Labels change from leave-last-out (one hidden item) to the real OTTO competition's format: given a session prefix, predict up to 20 items separately for each of clicks/carts/orders. Metric changes from NDCG@10 to the competition's weighted Recall@20 (0.10/0.30/0.60). Target: a real submission to `otto-recommender-system` (closed 2023-01-31, but still likely scorable as a late submission).

**Why**: the human wants to compare all five methods properly (D2's trim was accepted at the time to keep Level 0 small, but is now explicitly being revisited), and the real competition task is a better, more standard fit for a recsys problem than the search-flavored framing D2 operated under.

**What carries over unchanged**: co-visitation computation, category clustering, candidate-pool generation, and their three bug fixes (leakage, clustering collapse, OOM/batching) — all of that logic is reusable, just retargeted at multi-label outputs instead of a single leave-last-out item.

**What needs rebuilding**: Phase 1's label construction (multi-event-type labels, not single-item), the eval harness (weighted Recall@20 alongside NDCG), and three new methods (BM25, Two-Tower, DCN-V2+MMoE).

**Status**: Accepted (human decision). Sequencing to be decided step by step per the human's stated preference.
