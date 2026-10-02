"""Bootstrap confidence intervals over sessions, and paired bootstrap for model comparison.

Resampling is always over sessions/queries, never over individual candidates -
candidates within one session are not independent observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    per_session_scores: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the given per-session metric values."""
    values = np.asarray(per_session_scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(resampled_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def paired_bootstrap_pvalue(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap test for whether model A beats model B on the same sessions.

    Returns (mean_diff, p_value), where p_value is the two-sided fraction of
    resampled mean differences that cross zero relative to the observed direction.
    Scores must be aligned session-for-session (same order, same sessions).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must be paired, same-length arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = diff[idx].mean()
    observed = diff.mean()
    if observed >= 0:
        p_value = float((resampled_means <= 0).mean())
    else:
        p_value = float((resampled_means >= 0).mean())
    return float(observed), 2 * min(p_value, 0.5)
