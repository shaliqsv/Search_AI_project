"""Bootstrap confidence intervals over sessions (issue #14).

Import from a notebook in this folder with `from bootstrap import bootstrap_ci, paired_bootstrap_diff`.
They stay here, not in `src/ranking/`, until issue #52.

Always pass one value per *session*. Resampling events or candidate rows would give intervals
that are too narrow, because rows from one session are not independent.
"""

import numpy as np


def _check(values, name="values"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.isnan(values).any():
        raise ValueError(f"{name} contains NaN")
    return values


def _boot_means(values, n_boot, seed, batch_bytes=200_000_000):
    """Means of n_boot resamples. Done in batches so 200,000 sessions x 1,000 resamples fits in RAM."""
    rng = np.random.default_rng(seed)
    n = values.size
    batch = max(1, min(n_boot, batch_bytes // (8 * n)))
    out = np.empty(n_boot)
    for start in range(0, n_boot, batch):
        stop = min(start + batch, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = values[idx].mean(axis=1)
    return out


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Mean of per-session values and its (1 - alpha) percentile interval: (mean, low, high)."""
    values = _check(values)
    means = _boot_means(values, n_boot, seed)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap_diff(a, b, n_boot=1000, alpha=0.05, seed=0):
    """Mean of (a - b) over the same sessions and its interval: (mean_diff, low, high).

    Resamples the per-session differences, so both models see the same resampled sessions.
    """
    a = _check(a, "a")
    b = _check(b, "b")
    if a.size != b.size:
        raise ValueError(f"a and b must have the same length, got {a.size} and {b.size}")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)
