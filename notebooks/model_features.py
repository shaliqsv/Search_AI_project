"""Candidate feature families of Step 1 (`_doc/modeling_guide.md`). Every family is an `Extra`: `fit(q)` learns on the training
queries of a fold only (bins, PCA, clusters, encodings), `__call__(q)` returns (names, array with one row per candidate).
All of them are computable at prediction time from the candidate rows of the query and, for time features, the request time."""

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer

POOL = 200


def _col(q, name):
    return q.df[name].to_numpy().astype(np.float64)


def _per_query(x, q):
    return x.reshape(q.nq, POOL)


class Extra:
    name = "extra"

    def fit(self, q):
        return self


class Ratios(Extra):
    """Hypothesis: co-visitation strength only means 'associated' when it is high relative to how popular the item is, and recent
    popularity relative to all-time popularity signals a trend."""
    name = "ratios, differences, interactions"

    def __call__(self, q):
        wsum, cmax, cmean = np.log1p(_col(q, "covis_wsum")), np.log1p(_col(q, "covis_max")), np.log1p(_col(q, "covis_mean"))
        lc, lc3 = _col(q, "log_clicks"), _col(q, "log_clicks_3d")
        rc = _col(q, "rank_covis"); pl_ = _col(q, "prefix_len")
        a = np.stack([wsum - lc, cmax - lc, lc3 - lc, cmean - cmax, np.log1p(pl_) * (rc < 201), cmax * _col(q, "prefix_share_same_cat"),
                      _col(q, "pop_cat_pct") * _col(q, "prefix_share_same_cat")], axis=1)
        return ["covis_wsum_minus_popularity", "covis_max_minus_popularity", "recent_minus_all_popularity", "covis_mean_minus_max",
                "prefix_len_x_has_covis", "covis_max_x_same_category_share", "category_popularity_x_same_category_share"], a


class Bins(Extra):
    """Hypothesis: the EDA decile plots showed non-linear effects for the linear model; quantile bins let it fit them."""
    name = "non-linear forms (bins)"
    cols = ("covis_max", "covis_wsum", "log_clicks", "log_clicks_3d", "pop_cat_pct", "prefix_len", "item_cat_size")

    def __init__(self, n_bins=8):
        self.n_bins = n_bins

    def fit(self, q):
        x = np.stack([_col(q, c) for c in self.cols], axis=1)
        idx = np.random.default_rng(42).choice(len(x), min(len(x), 400_000), replace=False)
        self.kb = KBinsDiscretizer(self.n_bins, encode="onehot-dense", strategy="quantile", subsample=None).fit(x[idx] + 1e-9 * np.random.default_rng(1).random((len(idx), x.shape[1])))
        return self

    def __call__(self, q):
        x = np.stack([_col(q, c) for c in self.cols], axis=1)
        a = self.kb.transform(x)
        return [f"bin{i}" for i in range(a.shape[1])], a


class QueryContext(Extra):
    """Hypothesis: relevance is relative inside a pool: a candidate's rank among the 200 (and how many candidates have a
    co-visitation list at all) says more than its raw value. Aggregates use the query's own pool only, so they exist at prediction time."""
    name = "query-context aggregates"
    cols = ("covis_max", "covis_wsum", "log_clicks", "log_clicks_3d", "rank_pop", "prefix_share_same_cat")

    def __call__(self, q):
        parts, names = [], []
        for c in self.cols:
            x = _per_query(_col(q, c), q)
            parts.append((rankdata(x, axis=1) / POOL).ravel()); names.append(f"{c}_pool_percentile")
            z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9); parts.append(z.ravel()); names.append(f"{c}_pool_z")
        has = (_per_query(_col(q, "rank_covis"), q) < 201)
        parts.append(np.repeat(has.mean(1), POOL)); names.append("pool_share_with_covis_list")
        parts.append(np.repeat(np.log1p(_per_query(_col(q, "covis_wsum"), q).max(1)), POOL)); names.append("pool_max_covis_wsum")
        return names, np.stack(parts, axis=1)


class TimeParts(Extra):
    """Hypothesis: browsing behaviour differs by hour and weekday. Uses the session start time (the request time at serving)."""
    name = "time features (hour, weekday)"

    def __init__(self, first_ts_df, start_ms):
        self.first, self.start = first_ts_df, start_ms

    def __call__(self, q):
        ts = pl.DataFrame({"session": q.sessions}).join(self.first, on="session", how="left")["first_ts"].to_numpy().astype(np.float64)
        hours = (ts - self.start) / 3.6e6 + 22.0                     # day 0 starts at 22:00 UTC
        hod, dow = hours % 24, (hours // 24) % 7
        a = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=1)
        return ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"], np.repeat(a, POOL, axis=0)


class CategoryFrequency(Extra):
    """Hypothesis: how many candidates/queries a synthetic category has says something about its intent (EDA: no gain expected)."""
    name = "categorical: category frequency"

    def fit(self, q):
        cnt = q.df["category"].value_counts(); self.freq = dict(zip(cnt["category"].to_list(), (cnt["count"] / q.df.height).to_list(), strict=True)); return self

    def __call__(self, q):
        return ["category_frequency"], np.array([self.freq.get(c, 0.0) for c in q.df["category"].to_list()])[:, None]


class CategoryTargetRate(Extra):
    """Cross-fitted target rate of the category (5 folds over training queries; unseen levels get the prior)."""
    name = "categorical: cross-fitted category target rate"

    def __init__(self, smooth=50):
        self.smooth = smooth

    def _rates(self, cats, y):
        prior = y.mean(); d = pl.DataFrame({"c": cats, "y": y}).group_by("c").agg(pl.col("y").sum().alias("s"), pl.len().alias("n"))
        return {r["c"]: (r["s"] + self.smooth * prior) / (r["n"] + self.smooth) for r in d.iter_rows(named=True)}, prior

    def fit(self, q):
        cats, y = q.df["category"].to_numpy(), q.y.astype(float); self.map, self.prior = self._rates(cats, y)
        self.oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=42).split(np.arange(q.nq)):
            ra, rb = (a[:, None] * POOL + np.arange(POOL)).ravel(), (b[:, None] * POOL + np.arange(POOL)).ravel()
            m, p = self._rates(cats[ra], y[ra]); self.oof[rb] = [m.get(c, p) for c in cats[rb]]
        self.train_key = q.sessions[:3].tolist() + [q.nq]
        return self

    def __call__(self, q):
        if q.nq == self.train_key[-1] and q.sessions[:3].tolist() == self.train_key[:3]:        # the training queries themselves: cross-fitted values
            return ["category_target_rate"], self.oof[:, None]
        return ["category_target_rate"], np.array([self.map.get(c, self.prior) for c in q.df["category"].to_numpy()])[:, None]


class Structure(Extra):
    """Hypothesis: the EDA found one small cluster with 2.4x the target rate; distances to centroids and leading principal components
    may expose such structure to the model."""
    name = "structure (cluster distances, principal components)"

    def fit(self, q):
        from model_cv import Prep
        self.prep = Prep("linear").fit(q); x = self.prep.transform(q); idx = np.random.default_rng(42).choice(len(x), min(len(x), 300_000), replace=False)
        self.pca = PCA(3, random_state=42).fit(x[idx]); self.km = KMeans(2, n_init=3, random_state=42).fit(x[idx]); return self

    def __call__(self, q):
        x = self.prep.transform(q)
        return ["pc1", "pc2", "pc3", "dist_cluster0", "dist_cluster1"], np.hstack([self.pca.transform(x), self.km.transform(x)])


class Missingness(Extra):
    """Hypothesis: a candidate can come from the co-visitation list, the popularity list or both; the joint pattern and count of
    missing ranks (code 201) is a compact description of where it came from."""
    name = "missingness (count, joint pattern)"

    def __call__(self, q):
        a, b = _col(q, "rank_covis") == 201, _col(q, "rank_pop") == 201
        return ["missing_count", "missing_both", "covis_only", "pop_only"], np.stack([a * 1.0 + b, a & b, (~a) & b, a & (~b)], axis=1).astype(float)
