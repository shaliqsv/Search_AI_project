"""Shared harness for the modelling notebook (`_doc/modeling_guide.md`, rules 3, 7 and 8).

Unit of analysis = one query = 200 candidate rows (contiguous). Folds are made over queries, so a query is never split.
Primary metric: NDCG@10 of the ranking a model produces inside the 200-candidate pool, with the ideal DCG taken from the
FULL label set (same scale as the earlier method comparison, `notebooks/evalset.py`). Ties in a score keep the pool order.
Secondary: PR-AUC and ROC-AUC on y = label > 0.
Preprocessing is the EDA specification (`eda/preprocessing_spec.md`) as a fit/transform object: anything learned (median,
IQR) is fit on the training queries of a fold only.
"""

import pickle
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker
from model_data import DEV_TABLE, dev_idcg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

POOL = 200
SEED = 42
F15 = ["rank_covis", "rank_pop", "log_clicks", "log_clicks_3d", "cart_rate", "order_rate", "pop_cat_pct", "item_cat_size",
       "covis_max", "covis_mean", "covis_wsum", "prefix_len", "prefix_n_cart", "prefix_n_order", "prefix_share_same_cat"]
SKEWED = ["rank_covis", "cart_rate", "order_rate", "pop_cat_pct", "covis_max", "covis_mean", "covis_wsum", "prefix_len",
          "prefix_n_cart", "prefix_n_order"]
MISS = ["rank_covis", "rank_pop"]
DISCOUNT = 1.0 / np.log2(np.arange(10) + 2)


class Queries:
    """Whole queries of a candidate table: `df` (rows of one session contiguous, 200 each), grades, ideal DCG."""

    def __init__(self, df, idcg):
        assert df.height % POOL == 0 and (df["session"].to_numpy()[::POOL] == df["session"].to_numpy()[POOL - 1::POOL]).all()
        self.df, self.nq = df, df.height // POOL
        self.sessions = df["session"].to_numpy()[::POOL]
        self.grade = df["label"].to_numpy().reshape(self.nq, POOL).astype(np.float32)
        self.idcg = np.asarray(idcg, dtype=np.float64)
        self.y = (df["label"].to_numpy() > 0).astype(np.int8)

    def take(self, q_idx):
        rows = (np.asarray(q_idx)[:, None] * POOL + np.arange(POOL)).ravel()
        return Queries(self.df[rows], self.idcg[q_idx])


def load_queries(path, idcg_df, n_queries=None, seed=SEED, columns=None):
    """Load (a random sample of whole) queries of a candidate table together with their ideal DCG."""
    lf = pl.scan_parquet(path)
    sessions = lf.select("session").unique().collect()["session"].sort().to_numpy()
    if n_queries and n_queries < len(sessions):
        sessions = np.sort(np.random.default_rng(seed).choice(sessions, n_queries, replace=False))
    cols = columns or lf.collect_schema().names()
    df = lf.select(cols).filter(pl.col("session").is_in(sessions.tolist())).collect().sort("session", maintain_order=True)
    ideal = idcg_df.filter(pl.col("session").is_in(sessions.tolist())).sort("session")
    assert ideal["session"].to_list() == sessions.tolist()
    return Queries(df, ideal["idcg"].to_numpy())


def load_dev(n_queries=None, seed=SEED, columns=None):
    return load_queries(DEV_TABLE, dev_idcg(), n_queries, seed, columns)


def ndcg10(scores, grade, idcg):
    """Per-query NDCG@10 of the ranking induced by `scores` (nq x 200); ties keep the pool order."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    return (np.take_along_axis(grade, order, 1) * DISCOUNT).sum(1) / idcg


def pool_ceiling(grade, idcg):
    """NDCG@10 of a perfect ordering of the pool (what a perfect reranker could reach)."""
    return ndcg10(grade + np.linspace(0, 1e-3, POOL)[::-1], grade, idcg)


class Prep:
    """EDA preprocessing per model family, fit on training queries only. `extras`: functions Queries -> (names, array)."""

    def __init__(self, family, cols=None, extras=()):
        self.family, self.cols, self.extras = family, list(cols or F15), list(extras)
        self.names = []

    def _raw(self, q):
        x = q.df.select(self.cols).to_numpy().astype(np.float32)
        names = list(self.cols)
        mi = [self.cols.index(c) for c in MISS if c in self.cols]
        if self.family == "linear":
            x = np.hstack([x, (x[:, mi] == 201).astype(np.float32)])
            names += [f"{self.cols[i]}_missing" for i in mi]
            for c in SKEWED:
                if c in self.cols:
                    j = self.cols.index(c)
                    x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
        else:
            for i in mi:
                x[x[:, i] == 201, i] = np.nan
        for fn in self.extras:
            n, a = fn(q)
            x = np.hstack([x, a.astype(np.float32)])
            names += list(n)
        self.names = names
        return x

    def fit(self, q):
        for fn in self.extras:                       # stateful features (bins, PCA, clusters, encodings) learn on the training queries only
            if hasattr(fn, "fit"):
                fn.fit(q)
        x = self._raw(q)
        self.scaler = RobustScaler().fit(x[np.random.default_rng(SEED).choice(len(x), min(len(x), 600_000), replace=False)]) if self.family == "linear" else None
        return self

    def transform(self, q):
        x = self._raw(q)
        return self.scaler.transform(x).astype(np.float32) if self.scaler is not None else x


@dataclass
class Spec:
    """A model recipe: family decides the preprocessing; ladder = position on the complexity ladder (1 = simplest)."""

    name: str
    family: str
    factory: object
    ladder: int
    rank: bool = False
    cols: tuple = ()
    extras: tuple = ()
    selector: object = None          # fit(x_train, y_train, names, family) -> list of column indices to keep (learned inside the fold)

    def make(self):
        return self.factory()


def linear(C=1.0, name="regularised linear"):
    return Spec(name, "linear", lambda: LogisticRegression(C=C, max_iter=300), 1)


def lgbm(name="LightGBM defaults (class weights)", ladder=3, rank=False, **kw):
    params = {"n_estimators": 100, "n_jobs": 4, "random_state": SEED, "verbose": -1}
    if rank:
        params.update(objective="lambdarank", label_gain=[0, 1, 2, 3])
        fac = lambda: LGBMRanker(**{**params, **kw})
    else:
        params.update(class_weight="balanced")
        fac = lambda: LGBMClassifier(**{**params, **kw})
    return Spec(name, "gbm", fac, ladder, rank)


def fit_score(spec, q_tr, q_va, prep=None):
    """Fit on training queries, return (scores nq x 200 for q_va, fit seconds, predict seconds, size in MB, n features)."""
    prep = prep or Prep(spec.family, spec.cols or None, spec.extras).fit(q_tr)
    x_tr, x_va = prep.transform(q_tr), prep.transform(q_va)
    if spec.selector is not None:
        keep = spec.selector(x_tr, q_tr, prep.names, spec.family)
        x_tr, x_va = x_tr[:, keep], x_va[:, keep]
    m = spec.make()
    t0 = time.time()
    m.fit(x_tr, q_tr.y, group=[POOL] * q_tr.nq) if spec.rank else m.fit(x_tr, q_tr.y)
    fit_s = time.time() - t0
    t0 = time.time()
    s = m.predict(x_va) if spec.rank else m.predict_proba(x_va)[:, 1]
    return s.reshape(q_va.nq, POOL), fit_s, time.time() - t0, len(pickle.dumps(m)) / 1e6, x_tr.shape[1]


def folds(nq, seed, k=3):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(nq)))


def cv(spec, data, seeds=(SEED,), k=3, keep_oof=False):
    """Cross-validate one recipe on the same folds for every recipe. Returns (rows DataFrame, oof dict seed -> nq x 200 scores)."""
    rows, oof = [], {}
    for seed in seeds:
        scores = np.zeros((data.nq, POOL), dtype=np.float32)
        for fold, (tr, va) in enumerate(folds(data.nq, seed, k)):
            q_tr, q_va = data.take(tr), data.take(va)
            s, fit_s, pred_s, size, nf = fit_score(spec, q_tr, q_va)
            nd = ndcg10(s, q_va.grade, q_va.idcg)
            rows.append({"option": spec.name, "seed": seed, "fold": fold, "ndcg10": float(nd.mean()), "pr_auc": average_precision_score(q_va.y, s.ravel()),
                         "roc_auc": roc_auc_score(q_va.y, s.ravel()), "fit_seconds": fit_s, "predict_seconds": pred_s, "size_mb": size, "n_features": nf, "queries": q_va.nq})
            scores[va] = s
        oof[seed] = scores
    return pl.DataFrame(rows), (oof if keep_oof else None)


def summarise(rows):
    """Mean and standard deviation over folds per option, plus the spread of the seed means (rule 4)."""
    per_seed = rows.group_by("option", "seed").agg(pl.col("ndcg10").mean().alias("m"))
    spread = per_seed.group_by("option").agg(pl.col("m").std().alias("seed_spread"))
    return (rows.group_by("option", maintain_order=True)
            .agg(pl.col("ndcg10").mean().alias("ndcg10"), pl.col("ndcg10").std().alias("ndcg10_sd"), pl.col("pr_auc").mean().alias("pr_auc"),
                 pl.col("pr_auc").std().alias("pr_auc_sd"), pl.col("roc_auc").mean().alias("roc_auc"), pl.col("fit_seconds").mean().alias("fit_seconds"),
                 pl.col("predict_seconds").mean().alias("predict_seconds"), pl.col("size_mb").mean().alias("size_mb"), pl.col("n_features").first().alias("n_features"),
                 pl.len().alias("folds")).join(spread, on="option").with_columns(pl.col("seed_spread").fill_null(0.0)))


def decide(summary, ladder, noise, metric="ndcg10"):
    """Rule 4: options within `noise` of the best are tied; the simplest (lowest ladder number) wins ties.
    Returns (winner, tied options, best option, gap of winner to best)."""
    s = summary.sort(metric, descending=True)
    best = s.row(0, named=True)
    tied = [r["option"] for r in s.iter_rows(named=True) if best[metric] - r[metric] <= noise]
    winner = min(tied, key=lambda o: (ladder.get(o, 9), -float(s.filter(pl.col("option") == o)[metric][0])))
    return winner, tied, best["option"], best[metric] - float(s.filter(pl.col("option") == winner)[metric][0])


def experiment_rows(step, rows, summary, descriptions, features, parameters, seeds):
    """Convert a summary table into rows of modeling/experiments.csv."""
    out = []
    for r in summary.iter_rows(named=True):
        eid = f"S{step}-" + "".join(c if c.isalnum() else "_" for c in r["option"])[:60]
        out.append({"id": eid, "step": str(step), "description": descriptions.get(r["option"], r["option"]), "features": features, "model": r["option"],
                    "parameters": parameters.get(r["option"], ""), "seeds": str(list(seeds)), "ndcg10_mean": r["ndcg10"], "ndcg10_sd": r["ndcg10_sd"],
                    "pr_auc_mean": r["pr_auc"], "pr_auc_sd": r["pr_auc_sd"], "fit_seconds": r["fit_seconds"], "predict_seconds": r["predict_seconds"],
                    "size_mb": r["size_mb"], "notes": f"{r['folds']} folds; seed spread {r['seed_spread']:.4f}"})
    return out
