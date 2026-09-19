"""Benchmark: train the Two-Tower on CPU at a fixed size (issue #30). Run as its own process."""

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from tt_setup import score, setup, train

N_QUERIES, EPOCHS, BATCH = 500_000, 4, 2048
t0 = time.time()
ns = setup(ROOT / "data", ROOT / "artifacts", n_train_queries=N_QUERIES)
t_setup = time.time() - t0
model, hist, seconds = train(ns, hard=0, epochs=EPOCHS, batch_size=BATCH, label="bench")
res, rec, s, _, _ = score(ns, model, "bench")
best_val = min(h[2] for h in hist)
print("RESULT " + json.dumps({
    "model": "Two-Tower", "device": "cpu", "threads": 4, "queries": ns.n_train_queries, "examples": int(ns.n_examples),
    "epochs": EPOCHS, "batch": BATCH, "setup_seconds": round(t_setup), "train_seconds": round(seconds),
    "seconds_per_epoch": round(seconds / EPOCHS, 1), "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    "best_validation_loss": round(best_val, 4), "eval_recall100": round(rec[100], 4), "eval_ndcg10": round(s["ndcg10"][0], 4),
}))
