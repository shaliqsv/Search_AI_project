# D2: Phase 6 (Modeling) scope — trimmed to data_scientist.md's default

**Phase**: 6, Modeling. Also affects how `_doc/modeling_guide.md` should be read from here on.

**Decision**: `modeling_guide.md`'s steps are not followed as written. It treats the archived `_doc/outdate/plan.md` as its source of truth, references ~30 decision IDs (D4–D33) that don't exist in the fresh `_doc/decisions/`, and its scope is the full old plan: five methods (co-visitation, BM25, LightGBM LambdaMART, Two-Tower + FAISS, DCN-V2 + MMoE), three Bedrock/Claude roles (classifier, explainer, judge) with cost tracking, a position-bias IPW simulation study, and an ONNX/FAISS/S3 deployment bundle that it calls "the next piece of Level 0 work."

Phase 6 instead follows `data_scientist.md`'s default steps: list candidate algorithms, train with fixed seeds and logged configs, tune on validation within a small budget, handle class imbalance, and select one model by comparing against the Phase 5 baseline with a paired bootstrap over sessions. Concretely: co-visitation (the Phase 5 baseline) vs. one learned reranker (LightGBM LambdaMART on graded gains).

**Why**: `modeling_guide.md`'s scope directly contradicts the Level/Phase mapping already agreed (Level 0 = Phases 0-7, notebooks only; Level 2 = Phase 8-9, deployment) — it puts deployment inside Level 0. It also reintroduces AWS/Bedrock cost and complexity into a phase that's supposed to be notebook-only. Rebuilding all ~30 referenced decisions from scratch to make the guide followable as written would be a large, mostly-redone effort disconnected from this restart's simpler methodology.

**Alternatives considered**: following `modeling_guide.md` in full (rejected — see contradiction above; this is the "keep the full scope" option the human explicitly declined); a mix of the two (rejected for Phase 6 itself — the human chose the clean trim, not a partial one).

**What this drops for now (Level 0)**: BM25, Two-Tower, DCN-V2+MMoE, the Bedrock classifier/explainer/judge roles, the position-bias IPW study, and the ONNX/FAISS/S3 deployment bundle. None of this is ruled out permanently — if wanted later (e.g. as Level 1/2 work, or a deliberate scope expansion), it should be re-scoped against the current phase structure rather than resurrected from `modeling_guide.md` as-is.

**Status**: Accepted (human decision, not provisional).
