"""Sliced evaluation (issue #15).

Import from a notebook in this folder with `from slicing import ...`.
Stays here, not in `src/ranking/`, until issue #52.
"""

import numpy as np
import polars as pl
from bootstrap import bootstrap_ci

LENGTH_LABELS = ["1", "2-3", "4-10", "11+"]
POPULARITY_LABELS = ["head", "torso", "tail"]


def length_bucket(n_events):
    """Session length bucket: 1, 2-3, 4-10, 11+ events."""
    if n_events <= 1:
        return "1"
    if n_events <= 3:
        return "2-3"
    if n_events <= 10:
        return "4-10"
    return "11+"


def popularity_buckets(train_counts, head_share=0.5, torso_share=0.3):
    """Bucket items by popularity, using TRAINING-window counts only.

    Items are sorted by count. Head = the most popular items that together make up `head_share`
    of all events, torso = the next `torso_share`, tail = the rest. Returns {item: bucket}.
    """
    items = sorted(train_counts, key=lambda i: (-train_counts[i], i))
    total = sum(train_counts.values())
    out = {}
    running = 0
    for item in items:
        share_before = running / total
        if share_before < head_share:
            out[item] = "head"
        elif share_before < head_share + torso_share:
            out[item] = "torso"
        else:
            out[item] = "tail"
        running += train_counts[item]
    return out


def sliced_metrics(df, slice_col, value_col, min_n=100, n_boot=1000, seed=0):
    """Mean of `value_col` with a bootstrap interval for each value of `slice_col`.

    `df` has one row per session. Slices with fewer than `min_n` sessions are flagged "low n",
    never dropped. Columns: slice, n, metric, low, high, flag.
    """
    if df[slice_col].null_count() > 0:
        raise ValueError(f"{slice_col} has missing values, every session needs exactly one slice")
    rows = []
    for key, part in df.group_by(slice_col, maintain_order=True):
        values = part[value_col].to_numpy()
        mean, low, high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "slice": key[0] if isinstance(key, tuple) else key,
                "n": len(values),
                "metric": mean,
                "low": low,
                "high": high,
                "flag": "low n" if len(values) < min_n else "",
            }
        )
    out = pl.DataFrame(rows, schema_overrides={"slice": pl.Utf8}) if rows else pl.DataFrame()
    return out.sort("slice")


def check_partition(table, df):
    """Each session belongs to exactly one slice: counts add up to the number of sessions."""
    assert table["n"].sum() == df.height, (table["n"].sum(), df.height)
    return np.True_
