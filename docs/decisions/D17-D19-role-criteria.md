# D17-D19. Pass criteria for the Claude roles (written before evaluation; Level 0)

Status: Accepted (criteria only; the evaluations are blocked until Bedrock or API access exists).

## Rule for every role
Pick the smallest Bedrock model that passes the role's criteria (order: claude-haiku-4-5, then claude-sonnet-5), keep the LLM only if it beats the non-LLM baseline by the stated margin, cache every call by the D19 keys, and log calls, tokens and dollars in modeling/bedrock_costs.csv.

## Classifier (D17), on the held-out part of the classifier evaluation set (artifacts/classifier_queries_v2.jsonl)
- Top-3 accuracy at least 0.05 above the TF-IDF prototype baseline (below), and top-1 accuracy at least 0.05 above it.
- Expected calibration error at most 0.10 after calibration on the dev part.
- Out-of-scope F1 at least the baseline's.
- Below the confidence threshold: blend the top categories (decided from the calibrated confidence histogram in Step 10).

## Explainer (D18)
- The faithfulness test (numbers and feature-group names cited must appear in the payload) passes on 100 percent of recorded fixtures; any unfaithful output falls back to the templated explanation.
- Attribution: the method with the higher removal-test lift; if within 10 percent, the lower latency.

## Judge (D19)
- Weighted kappa and Spearman against the human labels both at least 0.60 on the 150-200 labelled rows; position effect (range of mean score across presentation slots) at most 0.30; verbosity correlation (absolute Spearman) at most 0.30.
- The judge mainly measures category coherence with the session, not true product relevance; that limit is stated wherever judge scores appear.
