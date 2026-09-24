"""Frozen benchmark set of (category query, session prefix) pairs (issue #16).

Import from a notebook in this folder with `from benchmark import build_benchmark, content_hash`.
Stays here, not in `src/ranking/`, until issue #52.

Allocation rule (fixed before looking at results)
- Categories get an equal share: n // n_categories each, and the remainder goes one each to
  categories picked by the seeded shuffle.
- Inside the picks, prefix-length buckets (1, 2-3, 4-10, 11+) are kept as equal as possible.
  Equal shares over-represent long sessions compared with the data (44% have a one-event
  prefix), on purpose, so that longer sessions are tested too.
- Fallback: if a (category, bucket) cell has no queries left, the next least-used bucket that
  has some is used. Every session appears at most once.
"""

import hashlib
import json

import numpy as np
import polars as pl
from queries import prefix_bucket

BUCKETS = ["1", "2-3", "4-10", "11+"]


def build_benchmark(queries, n=300, seed=0):
    """Return a DataFrame with `benchmark_id`, `prefix_bucket` and the query columns."""
    rng = np.random.default_rng(seed)
    q = queries.sort("session").with_columns(
        pl.Series("prefix_bucket", [prefix_bucket(k) for k in queries.sort("session")["prefix_aid"].list.len().to_list()])
    )
    cats = sorted(q["query_category"].unique().to_list())
    base, extra = divmod(n, len(cats))
    order = list(rng.permutation(cats))
    quota = {c: base + (1 if i < extra else 0) for i, c in enumerate(order)}

    # group_by order is not guaranteed, so walk the cells in sorted order and sort sessions
    # inside each cell. Otherwise the same seed could give a different benchmark.
    cell = {}
    groups = {key: part for key, part in q.group_by(["query_category", "prefix_bucket"])}
    for key in sorted(groups):
        cell[key] = list(rng.permutation(np.sort(groups[key]["session"].to_numpy())))
    used_bucket = {b: 0 for b in BUCKETS}
    chosen = []
    for c in order:
        for _ in range(quota[c]):
            options = [b for b in BUCKETS if cell.get((c, b))]
            if not options:
                break
            least = min(used_bucket[b] for b in options)
            b = str(rng.choice([b for b in options if used_bucket[b] == least]))
            chosen.append(cell[(c, b)].pop())
            used_bucket[b] += 1
    out = q.filter(pl.col("session").is_in(chosen)).sort("session")
    return out.with_row_index("benchmark_id")


def content_hash(benchmark):
    """SHA-256 of a canonical JSON form, so it does not depend on the parquet writer version."""
    rows = benchmark.sort("session").select(
        "session", "prefix_aid", "prefix_type", "query_category", "label_aid", "label_type"
    ).to_dicts()
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
