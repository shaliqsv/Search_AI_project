"""Data builders for the modelling phase (`_doc/modeling_guide.md`).

- `dev_idcg`: the ideal DCG@10 of every development query, from its FULL label set (the held-out events on items of the
  query category that are not in the prefix), so that NDCG@10 is on the same scale as the earlier method comparison.
  The 40,000 queries are re-created exactly as in issue #22 (same events, same seed) and checked against the feature table.
- `build_holdout`: a FRESH holdout window (sessions starting on days 16-17, features as of day 16), built with the same code
  as the evaluation window of issue #22. Days 14-15 were already used to compare methods in issues #17-#29, so they cannot
  serve as an untouched holdout. Building the table does not score anything.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from covis import build_covisitation
from evalset import prepare_eval
from features import AsOf
from metrics import _dcg, item_gains
from queries import build_ranking_queries
from split import day_to_ms, split_by_time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "modeling" / "artifacts"
HOLDOUT_START_DAY, HOLDOUT_END_DAY = 16, 18          # sessions starting on days 16-17, features as of day 16
DEV_TABLE = DATA / "features" / "train_asof11.parquet"
HOLDOUT_TABLE = DATA / "features" / "holdout_h0.parquet"


def _events():
    return pl.scan_parquet(DATA / "sample" / "events.parquet")


def _item_cluster():
    cat = pl.read_parquet(DATA / "categories" / "item_category.parquet")
    return cat, dict(zip(cat["aid"].to_list(), cat["cluster"].to_list(), strict=True))


def _idcg(prepared, k=10):
    return pl.DataFrame({
        "session": [q["session"] for q in prepared],
        "idcg": [_dcg(sorted(item_gains(q["labels"]).values(), reverse=True)[:k]) for q in prepared],
        "n_relevant": [len(item_gains(q["labels"])) for q in prepared],
    })


def dev_idcg():
    """Per-session ideal DCG@10 for the 40,000 development queries (cached in modeling/artifacts)."""
    path = ARTIFACTS / "dev_idcg.parquet"
    if path.exists():
        return pl.read_parquet(path)
    _, item_cluster = _item_cluster()
    d11, d14 = day_to_ms(11), day_to_ms(14)
    _, train_events = split_by_time(_events(), d11, d11, d14)
    train_events = train_events.select("session", "aid", "ts", "type").sort("session", "ts").collect()
    queries, _ = build_ranking_queries(train_events, item_cluster, cap=10**9, seed=0)
    prepared_all, _ = prepare_eval(queries)
    keep = np.sort(np.random.default_rng(0).choice(len(prepared_all), size=min(40_000, len(prepared_all)), replace=False))
    out = _idcg([prepared_all[i] for i in keep]).sort("session")
    sessions = pl.scan_parquet(DEV_TABLE).select("session").unique().collect()["session"].sort()
    assert out["session"].to_list() == sessions.to_list(), "re-created development queries differ from the feature table"
    out.write_parquet(path)
    return out


def _build_table(prepared, asof, path, chunk=5000):
    writer, rows, start = None, 0, time.time()
    for i in range(0, len(prepared), chunk):
        table = pl.concat([pl.DataFrame(asof.query_features(q)) for q in prepared[i:i + chunk]]).to_arrow()
        writer = writer or pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        rows += table.num_rows
    writer.close()
    print(f"{path.name}: {len(prepared):,} queries, {rows:,} rows in {time.time() - start:.0f} s")


def holdout_fingerprint():
    q = pl.scan_parquet(HOLDOUT_TABLE).select("session").unique().collect()["session"].sort()
    return {"rows": int(pl.scan_parquet(HOLDOUT_TABLE).select(pl.len()).collect().item()), "queries": q.len(),
            "sha256_of_sorted_session_ids": hashlib.sha256(q.to_numpy().tobytes()).hexdigest(),
            "window": f"sessions starting on days {HOLDOUT_START_DAY}-{HOLDOUT_END_DAY - 1}, features as of day {HOLDOUT_START_DAY}"}


def build_holdout():
    """Build the fresh holdout table and its ideal DCGs once; return its fingerprint (nothing is scored)."""
    fp_path = ARTIFACTS / "holdout_fingerprint.json"
    if HOLDOUT_TABLE.exists() and fp_path.exists():
        return json.loads(fp_path.read_text())
    events = _events()
    categories, item_cluster = _item_cluster()
    d0, d1 = day_to_ms(HOLDOUT_START_DAY), day_to_ms(HOLDOUT_END_DAY)
    hist = events.filter(pl.col("ts") < d0).select("session", "aid", "ts", "type").collect()
    assert hist["ts"].max() < d0
    matrix, ids = build_covisitation(*(hist[c].to_numpy() for c in ("session", "aid", "ts", "type")), min_freq=5, window=10, top_k=100)
    del hist
    _, evaluation = split_by_time(events, d0, d0, d1)
    eval_events = evaluation.select("session", "aid", "ts", "type").sort("session", "ts").collect()
    queries, stats = build_ranking_queries(eval_events, item_cluster, cap=200_000, seed=0)
    prepared, s2 = prepare_eval(queries)
    assert all(q["prefix_ts_last"] >= d0 for q in prepared), "a holdout prefix starts before the cutoff"
    print("holdout queries:", stats, s2)
    _build_table(prepared, AsOf(events, d0, categories, matrix, ids), HOLDOUT_TABLE)
    _idcg(prepared).sort("session").write_parquet(ARTIFACTS / "holdout_idcg.parquet")
    fp = holdout_fingerprint()
    fp_path.write_text(json.dumps(fp, indent=1))
    return fp
