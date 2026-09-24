"""Shared evaluation rules for every ranking method (issue #17, reused by #18 onward).

Import from a notebook in this folder with `from evalset import prepare_eval, evaluate`.
Stays here, not in `src/ranking/`, until issue #52.

Rules, stated once and used by all methods
1. A method returns a ranked list of items for the query category. Items already in the session
   prefix are NOT allowed in the list (a search result should show something new).
2. Relevant labels = held-out events on items that are in the query category AND not in the
   prefix. Labels outside the category cannot be found by a method that ranks inside it.
3. Queries left with no relevant label are dropped for ALL methods (they cannot score).
4. Per session: NDCG@10, recall@20, OTTO weighted recall@20 (session version), and per-type
   recall@20 (for OTTO's official aggregate).
"""

import polars as pl
from bootstrap import bootstrap_ci
from metrics import (
    ndcg_at_k,
    otto_recall_by_type,
    otto_weighted_recall,
    otto_weighted_recall_session,
    recall_at_k,
)

TYPE_NAMES = {0: "click", 1: "cart", 2: "order"}
TOP = 200  # length of the list a method must return (recall@200 needs 200)


def prepare_eval(queries):
    """Return (list of query dicts, stats). Each dict: session, prefix (set and list), category, labels."""
    out = []
    stats = {"queries": queries.height, "no_relevant_label": 0}
    for row in queries.iter_rows(named=True):
        prefix = set(row["prefix_aid"])
        labels = {"click": set(), "cart": set(), "order": set()}
        for aid, typ, cluster in zip(row["label_aid"], row["label_type"], row["label_cluster"], strict=True):
            if cluster == row["query_category"] and aid not in prefix:
                labels[TYPE_NAMES[typ]].add(aid)
        if not any(labels.values()):
            stats["no_relevant_label"] += 1
            continue
        out.append(
            {
                "session": row["session"],
                "prefix": prefix,
                "prefix_list": row["prefix_aid"],
                "prefix_types": row["prefix_type"],
                "prefix_ts_last": row["prefix_ts_last"],
                "category": row["query_category"],
                "labels": labels,
            }
        )
    stats["kept"] = len(out)
    return out, stats


def evaluate(prepared, rank_fn):
    """Score a method. rank_fn(query dict) -> list of item ids (best first, no prefix items)."""
    rows = []
    for q in prepared:
        ranking = rank_fn(q)
        assert not (set(ranking) & q["prefix"]), "a prefix item is in the result list"
        relevant = set().union(*q["labels"].values())
        by_type = otto_recall_by_type(ranking, q["labels"], k=20)
        rows.append(
            {
                "session": q["session"],
                "category": q["category"],
                "prefix_len": len(q["prefix_list"]),
                "n_returned": len(ranking),
                "ndcg10": ndcg_at_k(ranking, q["labels"], k=10),
                "recall20": recall_at_k(ranking, relevant, k=20),
                "recall50": recall_at_k(ranking, relevant, k=50),
                "recall100": recall_at_k(ranking, relevant, k=100),
                "recall200": recall_at_k(ranking, relevant, k=200),
                "wrecall20": otto_weighted_recall_session(ranking, q["labels"], k=20),
                "click_recall20": by_type.get("click"),
                "cart_recall20": by_type.get("cart"),
                "order_recall20": by_type.get("order"),
            }
        )
    return pl.DataFrame(rows, schema_overrides={"click_recall20": pl.Float64, "cart_recall20": pl.Float64, "order_recall20": pl.Float64})


def summarize(results, n_boot=1000, seed=0):
    """Mean and 95 percent bootstrap interval of each metric, plus OTTO's official weighted recall."""
    out = {}
    for col in ("ndcg10", "recall20", "recall50", "recall100", "recall200", "wrecall20"):
        out[col] = bootstrap_ci(results[col].to_numpy(), n_boot=n_boot, seed=seed)
    per_session = [
        {t: r for t, r in (("click", a), ("cart", b), ("order", c)) if r is not None}
        for a, b, c in zip(results["click_recall20"], results["cart_recall20"], results["order_recall20"], strict=True)
    ]
    out["otto_weighted_recall20"] = otto_weighted_recall(per_session)
    return out
