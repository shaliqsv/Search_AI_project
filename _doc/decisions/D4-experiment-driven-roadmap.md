# D4: Experiment-driven roadmap for the five methods, BM25 dropped

**Phase**: 6 (Modeling). Refines D3 (which reopened all five methods).

**Decision**: Replace "build all five methods and compare them" with five specific, falsifiable
experiments, each answering a real question rather than just producing another number:

1. **Recall comparison**: co-visitation (view-to-view, view-to-cart, cart-to-order matrices) vs.
   Two-Tower neural embeddings, scored on the same metric (Recall@20 per type). Which retrieves the
   correct next item more often, per event type?
2. **Blended retrieval**: feed LightGBM candidates from co-visitation and Two-Tower together. Does
   combining sources beat either alone? (This mirrors the actual 1st-place architecture - see the
   mrkmakr slides reviewed in conversation.)
3. **Trees vs. deep learning**: LightGBM (LambdaMART) vs. DCN-V2, same candidates, same features. Which
   wins on this data?
4. **Shared vs. separate**: MMoE (one model, three task heads) vs. three separate DCN-V2 models (one
   per event type). The actual OTTO winner explicitly chose *not* to share representations across
   click/cart/order - this tests whether that was the right call, or just one valid choice among others.
5. **Ensemble**: blend LambdaMART + DCN-V2 + MMoE scores. Does the blend beat any single model? (The
   winner's own account: "no single model was magic, the combination was.")

**BM25 is dropped**, not deferred. Neither of the two real reference solutions reviewed (a strong public
notebook at LB 0.575, and the actual 1st-place solution at 0.605) used text-based retrieval at all -
unsurprising, since OTTO has no real item text for BM25 to work on. The three real recall sources,
per the 1st-place team's own account, are co-visitation, revisitation (an item already in the user's
own session history), and neural embeddings.

**Why**: addresses the concern raised in conversation that running models just to get a number,
on a dataset whose absolute ceiling is constrained by catalog sparsity, doesn't teach much. Each
experiment above has a real, falsifiable answer regardless of how high or low the absolute scores end
up being - the value is in the comparison, not the number.

**What this means for the already-built code**: `src/ranking/data/covisitation.py`'s existing
function (event-count sliding window, used for D4-style synthetic category clustering) stays as-is for
that purpose. A new, purpose-built co-visitation function is needed for retrieval: time-windowed
(not event-count-windowed, matching both reference solutions), with source/target event-type
restriction (to build view-to-view / view-to-cart / cart-to-order variants) and session history capped
to the most recent ~30 events before pairing (bounds the join size, same fix as the Phase 4 OOM incident).

**Status**: Accepted (human decision).
