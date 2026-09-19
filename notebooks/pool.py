"""Shared candidate pool (issue #21).

Import from a notebook in this folder with `from pool import build_pool`.
Stays here, not in `src/ranking/`, until issue #52.

Several retrievers each return a ranked list of items. The pool merges them using ranks only
(reciprocal rank fusion), because raw scores from different retrievers are not comparable:
    rrf(item) = sum over retrievers that returned the item of 1 / (rrf_k + rank)   (rank starts at 1)
Prefix items and duplicates are removed and the pool is cut at `cap`.

To add a retriever later (Two-Tower in #26), pass one more entry in `retrievers`.
"""


def build_pool(prefix, retrievers, cap=200, rrf_k=60):
    """Return a list of candidate dicts, best first.

    retrievers: {name: ranked list of item ids}. Each candidate has
      aid, rrf, source (retriever where it ranked best), source_rank (that rank, 1-based),
      ranks ({retriever: rank}).
    """
    ranks = {}
    for name, items in retrievers.items():
        seen = set()
        r = 0
        for aid in items:
            aid = int(aid)
            if aid in prefix or aid in seen:
                continue
            seen.add(aid)
            r += 1
            ranks.setdefault(aid, {})[name] = r
    rows = []
    for aid, per in ranks.items():
        best_name, best_rank = min(per.items(), key=lambda kv: (kv[1], kv[0]))
        rows.append(
            {
                "aid": aid,
                "rrf": sum(1.0 / (rrf_k + r) for r in per.values()),
                "source": best_name,
                "source_rank": best_rank,
                "ranks": per,
            }
        )
    rows.sort(key=lambda d: (-d["rrf"], d["source_rank"], d["aid"]))
    return rows[:cap]
