"""Point-in-time features for (session, candidate item) rows (issue #22).

Import from a notebook in this folder with `from features import AsOf, FEATURE_DICTIONARY`.
Stays here, not in `src/ranking/`, until issue #52. Written as plain importable functions with
no notebook state, because serving will reuse them (issue #58).

Point in time: every item-level number comes from events with ts < cutoff and nothing else.
An `AsOf` object is built for one cutoff. Use a cutoff at or before the start of every session
you build features for.
"""

import numpy as np
import polars as pl
from covis_rank import TYPE_WEIGHTS, CovisRanker
from metrics import item_gains
from pool import build_pool

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
RANK_FILL = 201  # rank given when a retriever did not return the item (pool cap is 200)
SMOOTH = 10.0    # pseudo-clicks used to smooth cart and order rates

FEATURE_DICTIONARY = {
    "session": "Session ID (not a feature).",
    "category": "Query category (synthetic cluster ID), the same for every candidate of a query. Categorical.",
    "aid": "Candidate item ID (not a feature).",
    "label": "Graded label: 0 not found later, 1 clicked, 2 carted, 3 ordered (highest wins). Target for ranking.",
    "y_click": "1 if the item was clicked in the held-out events, else 0.",
    "y_cart": "1 if the item was carted in the held-out events, else 0.",
    "y_order": "1 if the item was ordered in the held-out events, else 0.",
    "rank_covis": "Rank (1-200) of the item in the co-visitation list, 201 if that retriever did not return it.",
    "rank_pop": "Rank (1-200) of the item in the category popularity list, 201 if not returned.",
    "rrf": "Reciprocal rank fusion score of the item in the candidate pool.",
    "log_clicks": "log(1 + clicks on the item before the cutoff).",
    "log_clicks_3d": "log(1 + clicks on the item in the 3 days before the cutoff).",
    "cart_rate": "Smoothed carts per click of the item before the cutoff: (carts + 10 * global rate) / (clicks + 10).",
    "order_rate": "Smoothed orders per click of the item before the cutoff, same smoothing.",
    "pop_cat_pct": "Popularity rank of the item inside its category divided by the category size (small = popular).",
    "item_cat_size": "Number of categorised items in the item's category.",
    "item_seen": "1 if the item had any event before the cutoff, else 0.",
    "hours_since_last_seen": "Hours between the item's last event before the cutoff and the end of the session prefix. Filled with 24 * (days before cutoff) for unseen items.",
    "covis_max": "Largest co-visitation weight between the item and any of the last 30 prefix items.",
    "covis_mean": "Mean co-visitation weight between the item and the last 30 prefix events (zeros count).",
    "covis_wsum": "Sum of co-visitation weights, weighted by event type (click 1, cart 3, order 6) and recency (0.8 per step back).",
    "prefix_len": "Number of events in the session prefix.",
    "prefix_n_cart": "Number of cart events in the prefix.",
    "prefix_n_order": "Number of order events in the prefix.",
    "prefix_share_same_cat": "Share of categorised prefix items that are in the query category.",
    "category_conf": "Confidence of the query category. 1.0 for now (true category); a classifier value from #34 later.",
}
FEATURE_COLUMNS = [c for c in FEATURE_DICTIONARY if c not in ("session", "aid", "label", "y_click", "y_cart", "y_order")]


class AsOf:
    """Item-level tables and rankers built only from events before `cutoff_ms`."""

    def __init__(self, events, cutoff_ms, categories, matrix, item_ids, cap=200, last_n=30, decay=0.8):
        self.cutoff = int(cutoff_ms)
        self.cap, self.last_n, self.decay = cap, last_n, decay
        st = (
            events.filter(pl.col("ts") < self.cutoff)
            .group_by("aid")
            .agg(
                (pl.col("type") == 0).sum().alias("clicks"),
                (pl.col("type") == 1).sum().alias("carts"),
                (pl.col("type") == 2).sum().alias("orders"),
                ((pl.col("type") == 0) & (pl.col("ts") >= self.cutoff - 3 * DAY_MS)).sum().alias("clicks_3d"),
                pl.col("ts").max().alias("last_ts"),
            )
            .collect()
            .sort("aid")
        )
        self.stat_aids = st["aid"].to_numpy()
        self.clicks = st["clicks"].to_numpy().astype(np.float64)
        self.carts = st["carts"].to_numpy().astype(np.float64)
        self.orders = st["orders"].to_numpy().astype(np.float64)
        self.clicks_3d = st["clicks_3d"].to_numpy().astype(np.float64)
        self.last_ts = st["last_ts"].to_numpy()
        total_clicks = max(self.clicks.sum(), 1.0)
        self.cart_prior = self.carts.sum() / total_clicks
        self.order_prior = self.orders.sum() / total_clicks

        joined = (
            categories.join(st.select("aid", "clicks"), on="aid", how="left")
            .with_columns(pl.col("clicks").fill_null(0))
            .sort(["cluster", "clicks", "aid"], descending=[False, True, False])
            .with_columns(
                (pl.int_range(pl.len()).over("cluster") + 1).alias("rank_in_cat"),
                pl.len().over("cluster").alias("cat_size"),
            )
        )
        joined = joined.with_columns((pl.col("rank_in_cat") / pl.col("cat_size")).alias("pct"))
        by_aid = joined.sort("aid")
        self.cat_aids = by_aid["aid"].to_numpy()
        self.cat_pct = by_aid["pct"].to_numpy()
        self.cat_size = by_aid["cat_size"].to_numpy()
        self.item_cluster = dict(zip(categories["aid"].to_list(), categories["cluster"].to_list(), strict=True))
        self.pop_by_cat = {
            c: part["aid"].to_numpy()[: 2 * cap] for (c,), part in joined.group_by("cluster", maintain_order=True)
        }
        self.covis = CovisRanker(matrix, item_ids, self.item_cluster, fill_by_cat=None, last_n=last_n, decay=decay, top=cap)
        self.matrix = self.covis.m
        self.max_hours = 24.0 * (self.cutoff - int(self.last_ts.min())) / DAY_MS if len(self.last_ts) else 24.0

    # -- export and reload (the same implementation serves training and serving) ---------------
    STATE_FIELDS = ("stat_aids", "clicks", "carts", "orders", "clicks_3d", "last_ts", "cat_aids", "cat_pct", "cat_size")

    def export_state(self, path):
        """Write everything `query_features` needs except the co-visitation matrix (stored separately) so serving can rebuild the object without events."""
        np.savez_compressed(path, cutoff=self.cutoff, cap=self.cap, last_n=self.last_n, decay=self.decay, cart_prior=self.cart_prior, order_prior=self.order_prior,
                            max_hours=self.max_hours, covis_ids=self.covis.item_ids, **{k: getattr(self, k) for k in self.STATE_FIELDS})

    @classmethod
    def from_state(cls, path, matrix, item_cluster):
        z = np.load(path)
        obj = object.__new__(cls)
        obj.cutoff, obj.cap, obj.last_n, obj.decay = int(z["cutoff"]), int(z["cap"]), int(z["last_n"]), float(z["decay"])
        for k in cls.STATE_FIELDS:
            setattr(obj, k, z[k])
        obj.cart_prior, obj.order_prior, obj.max_hours = float(z["cart_prior"]), float(z["order_prior"]), float(z["max_hours"])
        obj.item_cluster = item_cluster
        obj.pop_by_cat = {}
        obj.covis = CovisRanker(matrix, z["covis_ids"], item_cluster, fill_by_cat=None, last_n=obj.last_n, decay=obj.decay, top=obj.cap)
        obj.matrix = obj.covis.m
        return obj

    # -- helpers -----------------------------------------------------------------
    def _lookup(self, table_aids, aids):
        pos = np.searchsorted(table_aids, aids)
        pos = np.clip(pos, 0, len(table_aids) - 1)
        return pos, table_aids[pos] == aids

    def candidates(self, q):
        """The shared candidate pool of a prepared query (see pool.py)."""
        cov = self.covis.rank(q, pure=True)
        pop = []
        for a in self.pop_by_cat[q["category"]]:
            if int(a) not in q["prefix"]:
                pop.append(int(a))
                if len(pop) == self.cap:
                    break
        return build_pool(q["prefix"], {"covis": cov, "pop": pop}, cap=self.cap)

    def query_features(self, q, pool=None):
        """Feature columns (dict of arrays, one value per candidate) for one prepared query."""
        pool = pool if pool is not None else self.candidates(q)
        aids = np.array([c["aid"] for c in pool], dtype=np.int64)
        n = len(aids)
        gains = item_gains(q["labels"])
        types = {t: q["labels"][t] for t in ("click", "cart", "order")}

        pos, seen = self._lookup(self.stat_aids, aids)
        clicks = np.where(seen, self.clicks[pos], 0.0)
        carts = np.where(seen, self.carts[pos], 0.0)
        orders = np.where(seen, self.orders[pos], 0.0)
        clicks_3d = np.where(seen, self.clicks_3d[pos], 0.0)
        hours = np.where(
            seen, (q["prefix_ts_last"] - self.last_ts[pos]) / HOUR_MS, self.max_hours
        )
        cpos, cseen = self._lookup(self.cat_aids, aids)

        # co-visitation of each candidate with the last prefix events
        idx = self.covis.index
        n_pre = len(q["prefix_list"])
        start = max(0, n_pre - self.last_n)
        rows = [(p, idx.get(int(q["prefix_list"][p]))) for p in range(start, n_pre)]
        w = np.array(
            [TYPE_WEIGHTS[q["prefix_types"][p]] * self.decay ** (n_pre - 1 - p) for p, r in rows if r is not None]
        )
        prow = [r for _, r in rows if r is not None]
        cols = np.array([idx.get(int(a), -1) for a in aids])
        sub = np.zeros((len(prow), n))
        ok = cols >= 0
        if prow and ok.any():
            sub[:, ok] = self.matrix[prow][:, cols[ok]].toarray()
        c_max = sub.max(axis=0) if prow else np.zeros(n)
        c_wsum = w @ sub if prow else np.zeros(n)
        c_mean = sub.sum(axis=0) / max(len(rows), 1)   # events whose item is unknown to the matrix count as zeros

        categorised = [self.item_cluster.get(int(a), -1) for a in q["prefix_list"]]
        categorised = [c for c in categorised if c >= 0]
        share = (sum(c == q["category"] for c in categorised) / len(categorised)) if categorised else 0.0
        p_types = np.asarray(q["prefix_types"])

        return {
            "session": np.full(n, q["session"], dtype=np.int64),
            "category": np.full(n, q["category"], dtype=np.int32),
            "aid": aids.astype(np.int32),
            "label": np.array([gains.get(int(a), 0) for a in aids], dtype=np.int8),
            "y_click": np.array([int(a) in types["click"] for a in aids], dtype=np.int8),
            "y_cart": np.array([int(a) in types["cart"] for a in aids], dtype=np.int8),
            "y_order": np.array([int(a) in types["order"] for a in aids], dtype=np.int8),
            "rank_covis": np.array([c["ranks"].get("covis", RANK_FILL) for c in pool], dtype=np.int16),
            "rank_pop": np.array([c["ranks"].get("pop", RANK_FILL) for c in pool], dtype=np.int16),
            "rrf": np.array([c["rrf"] for c in pool], dtype=np.float32),
            "log_clicks": np.log1p(clicks).astype(np.float32),
            "log_clicks_3d": np.log1p(clicks_3d).astype(np.float32),
            "cart_rate": ((carts + SMOOTH * self.cart_prior) / (clicks + SMOOTH)).astype(np.float32),
            "order_rate": ((orders + SMOOTH * self.order_prior) / (clicks + SMOOTH)).astype(np.float32),
            "pop_cat_pct": np.where(cseen, self.cat_pct[cpos], 1.0).astype(np.float32),
            "item_cat_size": np.where(cseen, self.cat_size[cpos], 0).astype(np.int32),
            "item_seen": seen.astype(np.int8),
            "hours_since_last_seen": hours.astype(np.float32),
            "covis_max": c_max.astype(np.float32),
            "covis_mean": c_mean.astype(np.float32),
            "covis_wsum": c_wsum.astype(np.float32),
            "prefix_len": np.full(n, n_pre, dtype=np.int16),
            "prefix_n_cart": np.full(n, int((p_types == 1).sum()), dtype=np.int16),
            "prefix_n_order": np.full(n, int((p_types == 2).sum()), dtype=np.int16),
            "prefix_share_same_cat": np.full(n, share, dtype=np.float32),
            "category_conf": np.ones(n, dtype=np.float32),
        }
