"""Encode ranking queries with a trained Two-Tower query tower (own process: PyTorch cannot share one with FAISS or LightGBM on macOS).
Run: `uv run python scripts/tt_encode.py <model dir> <queries.parquet> <out.npy>`. The query vectors are unit length, one row per query in file order
(queries with no relevant label are dropped exactly as `prepare_eval` does, so the rows match `prepare_eval(queries)`)."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from evalset import prepare_eval

from ranking.models.two_tower import TwoTower, encode_queries

model_dir, queries_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
catalog_aids = np.load(model_dir / "catalog_aids.npy"); n_items = len(catalog_aids); index = {int(a): i for i, a in enumerate(catalog_aids)}
n_cats = len(json.loads((ROOT / "artifacts" / "category_names_v2.json").read_text())["categories"])
cats = pl.read_parquet(ROOT / "data" / "modeling" / "categories" / "item_category.parquet"); cat_of = dict(zip(cats["aid"].to_list(), cats["cluster"].to_list(), strict=True))
item_cat = torch.tensor([cat_of[int(a)] for a in catalog_aids]); state = torch.load(model_dir / "model.pt")
# item popularity buckets are part of the item tower, not the query tower, so the query vectors do not need them
model = TwoTower(n_items, n_cats, 16); model.load_state_dict(state); model.eval(); torch.set_num_threads(4)
prepared, _ = prepare_eval(pl.read_parquet(queries_path)); hist, w, cat_onehot, _ = encode_queries(prepared, index, n_items, n_cats)
with torch.no_grad():
    q = torch.cat([model.query(hist[s:s + 4096], w[s:s + 4096], cat_onehot[s:s + 4096]) for s in range(0, len(prepared), 4096)]).numpy()
np.save(out_path, q.astype(np.float32)); np.save(out_path.with_suffix(".sessions.npy"), np.array([p["session"] for p in prepared]))
print(f"encoded {len(prepared):,} queries -> {out_path}")
