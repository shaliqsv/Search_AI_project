"""Ranking metrics: NDCG@k and OTTO's weighted recall@k.

Every function takes graded relevance per candidate, aligned to a ranking order,
one query/session at a time - callers aggregate across queries themselves
(see bootstrap.py for confidence intervals over sessions).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of the first k gains, in the given order."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(gains: Sequence[float], k: int = 10) -> float:
    """NDCG@k for one query: gains are relevance scores in *predicted rank order*.

    Returns 0.0 when no gain is positive (nothing relevant exists to rank,
    so there is nothing to normalise against).
    """
    ideal = sorted(gains, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def recall_at_20_micro(
    predictions: Sequence[Sequence[int]], ground_truths: Sequence[Sequence[int]], k: int = 20
) -> float:
    """The real OTTO competition's Recall@20, for one event type, across many sessions.

    Micro-averaged, not a mean of per-session ratios: sum(|pred ∩ truth|) over sum(min(k,
    |truth|)). A session with empty ground truth contributes 0 to both the numerator and
    the denominator (per the official spec - verified against otto-de/recsys-dataset/KAGGLE.md).
    Sessions are implicitly paired by list position between `predictions` and `ground_truths`.
    """
    numerator = 0
    denominator = 0
    for pred, truth in zip(predictions, ground_truths, strict=True):
        truth_set = set(truth)
        numerator += len(truth_set & set(pred[:k]))
        denominator += min(k, len(truth_set))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def otto_weighted_score(click_recall: float, cart_recall: float, order_recall: float) -> float:
    """The competition's official blend: 0.10 clicks + 0.30 carts + 0.60 orders."""
    return 0.10 * click_recall + 0.30 * cart_recall + 0.60 * order_recall
