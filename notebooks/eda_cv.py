"""Shared cross-validation for the model-comparison steps of the EDA (`_doc/eda_guide.md`, section 3).

Same rows, same folds, same seed, same metric for every option of a step; two model families at
default settings (a regularised linear model and gradient boosting); anything that learns from data
is fit on the training part of each fold only (the `prep` function receives train and validation
separately). Folds are grouped by session, because the 200 candidate rows of one session are not
independent (Step 15 revisits the scheme).
"""

import warnings

import numpy as np
import polars as pl
from features import FEATURE_COLUMNS
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
NUM = [c for c in FEATURE_COLUMNS if c not in ("category", "category_conf")]    # 18 numeric features
FAMILIES = ("linear", "gbm")


def load_sample(root, n_queries=5000, seed=SEED, columns=None):
    """A sample of whole queries (200 rows each) of the candidates table. Returns a polars DataFrame with y = label > 0."""
    cols = list(dict.fromkeys(["session", "aid", "label"] + NUM + ["category"] + (columns or [])))
    df = pl.read_parquet(root / "data" / "features" / "train_asof11.parquet", columns=cols)
    sessions = df["session"].unique().sort().to_numpy()
    keep = np.random.default_rng(seed).choice(sessions, size=min(n_queries, len(sessions)), replace=False)
    return df.filter(pl.col("session").is_in(keep.tolist())).with_columns((pl.col("label") > 0).cast(pl.Int8).alias("y"))


def make_folds(df, k=3, seed=SEED):
    groups = df["session"].to_numpy()
    return list(GroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(groups)), groups=groups))


def make_model(family, scale=True):
    if family == "linear":
        lr = LogisticRegression(C=1.0, max_iter=300)
        return make_pipeline(StandardScaler(), lr) if scale else lr
    return LGBMClassifier(n_estimators=100, n_jobs=4, random_state=SEED, verbose=-1)


def evaluate_option(name, prep, df, folds, families=FAMILIES, scale=True):
    """Score one option. prep(train_df, val_df) -> (Xtr, ytr, Xva, yva) or a dict family -> that tuple.

    Returns a list of result rows (one per family and fold).
    """
    rows = []
    for fold, (tr, va) in enumerate(folds):
        out = prep(df[tr], df[va])
        for fam in families:
            if isinstance(out, dict) and fam not in out:
                continue                                    # the option does not apply to this family
            xtr, ytr, xva, yva = out[fam] if isinstance(out, dict) else out
            if fam == "linear" and np.isnan(xtr).any():
                continue                                    # a linear model cannot use NaN: option not applicable
            m = make_model(fam, scale=scale).fit(xtr, ytr)
            p = m.predict_proba(xva)[:, 1]
            rows.append({"option": name, "family": fam, "fold": fold, "pr_auc": average_precision_score(yva, p),
                         "roc_auc": roc_auc_score(yva, p), "rows_train": len(ytr), "rows_val": len(yva), "n_features": xtr.shape[1]})
    return rows


def summarise(rows):
    """Mean and standard deviation over folds per option and family."""
    return (pl.DataFrame(rows).group_by("option", "family", maintain_order=True)
            .agg(pl.col("pr_auc").mean().alias("pr_auc"), pl.col("pr_auc").std().alias("pr_auc_sd"),
                 pl.col("roc_auc").mean().alias("roc_auc"), pl.col("n_features").first().alias("n_features"),
                 pl.col("rows_train").mean().alias("rows_train"), pl.len().alias("folds")))


def decide(summary, family, complexity, exclude=()):
    """Guide rules 6 and 7: options within the larger of the two fold standard deviations of the best are tied;
    among tied options take the simplest (lowest complexity number). Returns (winner, tied list, noise)."""
    s = summary.filter((pl.col("family") == family) & ~pl.col("option").is_in(list(exclude))).sort("pr_auc", descending=True)
    best = s.row(0, named=True)
    tied = [r["option"] for r in s.iter_rows(named=True) if best["pr_auc"] - r["pr_auc"] < max(best["pr_auc_sd"], r["pr_auc_sd"])]
    winner = min(tied, key=lambda o: (complexity.get(o, 9), -float(s.filter(pl.col("option") == o)["pr_auc"][0])))
    gap = best["pr_auc"] - (s.filter(pl.col("option") == winner)["pr_auc"][0])
    return winner, tied, {"best": best["option"], "best_pr_auc": best["pr_auc"], "noise": best["pr_auc_sd"], "gap_to_best": gap}
