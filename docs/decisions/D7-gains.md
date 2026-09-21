# D7 (gains part). Graded gains for NDCG (frozen in modelling Step 1)

Status: Accepted for Level 0.

## Decision
Gains: click 1, cart 2, order 3 (linear, so NDCG matches scikit-learn). Frozen before any method comparison; changing them means rerunning every method.

## Why
Five baselines (random, two popularity lists, two co-visitation variants) scored on 20,000 validation queries under four gain sets give the same method ranking (Kendall tau against the provisional gains: flat (1, 1, 1) 1.00, steep (1, 3, 9) 1.00, OTTO weights (0.1, 0.3, 0.6) 1.00). The provisional set is the simplest and is the one the plan proposed.

## Alternatives considered
Flat (1, 1, 1), steep (1, 3, 9), OTTO weights (0.1, 0.3, 0.6) as gains. They change the level of NDCG but not the order of the methods. To be re-checked in Step 3 with the real baselines (co-visitation and BM25 with categories).
