"""Benchmark: train DCN-V2 + MMoE on CPU at a fixed size (issue #30). Run as its own process."""

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import polars as pl
import torch
from dcn import DCNMTL, predict, task_auc, train_model
from features import FEATURE_COLUMNS

torch.set_num_threads(4)
EPOCHS, BATCH, POOL = 3, 4096, 200
NUM = [c for c in FEATURE_COLUMNS if c not in ("category", "category_conf")]
t0 = time.time()
df = pl.read_parquet(ROOT / "data" / "features" / "train_asof11.parquet")
sessions = df["session"].unique().sort().to_numpy()
rng = np.random.default_rng(0)
val = set(rng.choice(sessions, size=len(sessions) // 10, replace=False).tolist())
is_val = df["session"].is_in(list(val)).to_numpy()
tr, va = df.filter(~pl.Series(is_val)), df.filter(pl.Series(is_val))
mean = tr.select(NUM).mean().to_numpy().ravel().astype(np.float32)
std = tr.select(NUM).std().to_numpy().ravel().astype(np.float32)
std[std < 1e-6] = 1.0


def tens(d):
    return (torch.from_numpy((d.select(NUM).to_numpy().astype(np.float32) - mean) / std),
            torch.from_numpy(d["category"].to_numpy().astype(np.int64)),
            torch.from_numpy(d.select(["y_click", "y_cart", "y_order"]).to_numpy().astype(np.float32)))


x_tr, c_tr, y_tr = tens(tr)
x_va, c_va, y_va = tens(va)
t_setup = time.time() - t0
model = DCNMTL(n_num=len(NUM), n_cats=int(df["category"].max()) + 1)
hist, best_epoch, seconds = train_model(model, x_tr, c_tr, y_tr, x_va, c_va, y_va, task_weights=(1.0, 2.0, 4.0), epochs=EPOCHS, batch=BATCH, label="bench")
auc = task_auc(predict(model, x_va, c_va), y_va.numpy())
print("RESULT " + json.dumps({
    "model": "DCN-V2 + MMoE", "device": "cpu", "threads": 4, "queries": len(x_tr) // POOL, "rows": len(x_tr),
    "epochs": EPOCHS, "batch": BATCH, "setup_seconds": round(t_setup), "train_seconds": round(seconds),
    "seconds_per_epoch": round(seconds / EPOCHS, 1), "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    "best_validation_loss": round(min(h["val"] for h in hist), 4), "validation_auc": [round(a, 3) for a in auc],
}))
