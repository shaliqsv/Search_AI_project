"""Export Two-Tower query vectors for the evaluation queries (used by notebook 26).

Run as a separate process: `python notebooks/tt_export_queries.py`.
PyTorch and FAISS cannot be imported in one process on macOS (two OpenMP runtimes, error 15;
the workaround KMP_DUPLICATE_LIB_OK is documented as unsafe), so the FAISS notebook only
reads the .npy files written here and never imports torch.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import polars as pl
import torch
from evalset import prepare_eval
from twotower import TwoTower, encode_queries

TT = ROOT / "data" / "models" / "two_tower"
cfg = json.loads((TT / "config.json").read_text())
catalog_aids = np.load(TT / "catalog_aids.npy")
index_of = {int(a): i for i, a in enumerate(catalog_aids)}
model = TwoTower(cfg["n_items"], cfg["n_cats"], cfg["n_pop"])
model.load_state_dict(torch.load(TT / "model.pt"))
model.eval()
prepared, _ = prepare_eval(pl.read_parquet(ROOT / "data" / "eval" / "ranking_queries.parquet"))
hist, w, cat, _ = encode_queries(prepared, index_of, cfg["n_items"], cfg["n_cats"])
with torch.no_grad():
    qvec = torch.cat([model.query(hist[s : s + 4096], w[s : s + 4096], cat[s : s + 4096]) for s in range(0, len(prepared), 4096)]).numpy()
np.save(TT / "eval_query_vectors.npy", qvec.astype(np.float32))
np.save(TT / "eval_query_sessions.npy", np.array([q["session"] for q in prepared], dtype=np.int64))
print(f"exported {qvec.shape[0]:,} query vectors of dimension {qvec.shape[1]}")
