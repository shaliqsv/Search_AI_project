"""Judge support (plan D19): agreement with hand labels and bias checks. No model calls happen here.

- weighted_kappa: quadratic-weighted Cohen's kappa between two integer ratings (0-4).
- spearman: rank correlation.
- position_effect: does the score depend on where a method's list appeared (order randomised)? Returns the mean score by slot and the spread.
- verbosity_effect: correlation between the length of a rendered result list and its score.
"""

import numpy as np
from scipy.stats import spearmanr


def weighted_kappa(a, b, levels=5):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    obs = np.zeros((levels, levels))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0)); w = ((np.arange(levels)[:, None] - np.arange(levels)[None, :]) ** 2) / (levels - 1) ** 2
    return float(1 - (w * obs).sum() / (w * exp).sum())


def spearman(a, b):
    return float(spearmanr(a, b).statistic)


def position_effect(scores, slots):
    """Mean judge score per presentation slot and the range of those means (a large range would signal position bias)."""
    scores, slots = np.asarray(scores, dtype=float), np.asarray(slots)
    means = {int(s): float(scores[slots == s].mean()) for s in np.unique(slots)}
    return means, max(means.values()) - min(means.values())


def verbosity_effect(scores, lengths):
    return spearman(scores, lengths)
