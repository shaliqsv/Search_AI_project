"""Ranking-evaluation queries from the held-out part of sessions (issue #10).

Import from a notebook in this folder with `from queries import build_ranking_queries`.
Stays here, not in `src/ranking/`, until issue #52.

KNOWN LEAK. The query category is the category of the held-out next click, so the "query"
already contains part of the answer. Ranking inside that category is easier than real search.
This is a stand-in for intent, because OTTO has no query text.
"""

import numpy as np
import polars as pl

HORIZON = 10  # labels = the next 10 events, starting at the held-out click
LENGTH_EDGES = [(1, "1"), (3, "2-3"), (10, "4-10")]


def prefix_bucket(n):
    for edge, name in LENGTH_EDGES:
        if n <= edge:
            return name
    return "11+"


def build_ranking_queries(eval_events, item_cluster, cap=200_000, seed=0, horizon=HORIZON):
    """One query per usable evaluation session.

    eval_events: DataFrame with session, aid, ts, type sorted by session, ts (see split.py).
    item_cluster: dict aid -> cluster (synthetic category).
    Returns (queries DataFrame, stats dict).

    For each session, the cut position c (1 <= c) is chosen at random among positions whose
    event is a click on an item that has a category. Prefix = events before c. The query category
    is that click's category. Labels = events c .. c + horizon - 1.
    """
    rng = np.random.default_rng(seed)
    sessions = eval_events.group_by("session", maintain_order=True).agg(
        pl.col("aid"), pl.col("ts"), pl.col("type")
    )
    stats = {"eval_sessions": sessions.height, "too_short": 0, "no_categorised_click": 0}
    rows = []
    for session, aids, tss, types in sessions.iter_rows():
        if len(aids) < 2:
            stats["too_short"] += 1
            continue
        clusters = [item_cluster.get(a, -1) for a in aids]
        candidates = [c for c in range(1, len(aids)) if types[c] == 0 and clusters[c] >= 0]
        if not candidates:
            stats["no_categorised_click"] += 1
            continue
        c = int(rng.choice(candidates))
        end = min(len(aids), c + horizon)
        rows.append(
            {
                "session": session,
                "prefix_aid": aids[:c],
                "prefix_type": types[:c],
                "prefix_ts_last": tss[c - 1],
                "query_category": clusters[c],
                "label_aid": aids[c:end],
                "label_type": types[c:end],
                "label_cluster": clusters[c:end],
                "label_ts_first": tss[c],
            }
        )
    queries = pl.DataFrame(
        rows,
        schema={
            "session": pl.Int64,
            "prefix_aid": pl.List(pl.Int32),
            "prefix_type": pl.List(pl.Int8),
            "prefix_ts_last": pl.Int64,
            "query_category": pl.Int32,
            "label_aid": pl.List(pl.Int32),
            "label_type": pl.List(pl.Int8),
            "label_cluster": pl.List(pl.Int32),
            "label_ts_first": pl.Int64,
        },
    )
    stats["usable"] = queries.height
    if queries.height > cap:  # stratify by prefix length bucket
        bucket = np.array([prefix_bucket(n) for n in queries["prefix_aid"].list.len().to_list()])
        keep = np.zeros(queries.height, dtype=bool)
        for b in np.unique(bucket):
            idx = np.flatnonzero(bucket == b)
            take = rng.choice(idx, size=round(cap * len(idx) / queries.height), replace=False)
            keep[take] = True
        queries = queries.filter(pl.Series(keep))
    stats["kept"] = queries.height
    return queries, stats
