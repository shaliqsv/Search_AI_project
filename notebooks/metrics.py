"""Ranking metrics written by hand (issue #13).

Import from a notebook in this folder with `from metrics import ndcg_at_k, ...`.
They stay here, not in `src/ranking/`, until issue #52 moves notebook code into the package.

Conventions
- `ranking`: list of item ids, best first. Repeated ids count only at their first position.
- `labels`: dict with keys "click", "cart", "order", each a set of item ids that happened later.
- Gains are linear (not 2**g - 1), so results match scikit-learn's `ndcg_score`.
"""

import math

DEFAULT_GAINS = {"click": 1, "cart": 2, "order": 3}
OTTO_WEIGHTS = {"click": 0.10, "cart": 0.30, "order": 0.60}


def _dedupe(ranking):
    seen = set()
    out = []
    for item in ranking:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def item_gains(labels, gains=DEFAULT_GAINS):
    """Gain per item. An item with several event types takes the highest gain."""
    out = {}
    for event_type, items in labels.items():
        for item in items:
            out[item] = max(out.get(item, 0), gains[event_type])
    return out


def _dcg(gain_list):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gain_list))


def ndcg_at_k(ranking, labels, k=10, gains=DEFAULT_GAINS):
    """NDCG@k. Returns 0.0 when there are no labels (nothing could be found)."""
    item_gain = item_gains(labels, gains)
    if not item_gain:
        return 0.0
    top = _dedupe(ranking)[:k]
    dcg = _dcg([item_gain.get(item, 0) for item in top])
    ideal = _dcg(sorted(item_gain.values(), reverse=True)[:k])
    return dcg / ideal


def recall_at_k(ranking, relevant, k=20):
    """Share of relevant items found in the top k. Returns 0.0 when nothing is relevant."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = set(_dedupe(ranking)[:k])
    return len(top & relevant) / len(relevant)


def otto_recall_by_type(ranking, labels, k=20):
    """OTTO recall per event type, hits / min(k, number of items). Only types that have labels."""
    top = set(_dedupe(ranking)[:k])
    return {
        event_type: len(top & set(items)) / min(k, len(items))
        for event_type, items in labels.items()
        if len(items) > 0
    }


def otto_weighted_recall(per_session_recalls, weights=OTTO_WEIGHTS):
    """OTTO's score: mean recall per type over the sessions that have that type, then weighted."""
    total = 0.0
    for event_type, weight in weights.items():
        values = [r[event_type] for r in per_session_recalls if event_type in r]
        if values:
            total += weight * sum(values) / len(values)
    return total


def otto_weighted_recall_session(ranking, labels, k=20, weights=OTTO_WEIGHTS):
    """Single-session version, for bootstrap. Weights are renormalised over the types present."""
    recalls = otto_recall_by_type(ranking, labels, k)
    if not recalls:
        return 0.0
    used = sum(weights[t] for t in recalls)
    return sum(weights[t] * r for t, r in recalls.items()) / used
