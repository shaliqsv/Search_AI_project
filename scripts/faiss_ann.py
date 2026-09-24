"""ANN choice for the Two-Tower item index (modelling Step 4). Run: `uv run python scripts/faiss_ann.py data/modeling/two_tower/<variant>`.

Exact search is the ground truth; IVF-PQ and HNSW are tested at several settings. Reports recall@100 against exact search, single-thread
latency (p50, p95) and index size on disk. Runs in its own process (FAISS and PyTorch cannot share one on macOS)."""

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

d = Path(sys.argv[1]); vecs = np.ascontiguousarray(np.load(d / "item_embeddings.npy"), dtype=np.float32); qall = np.load(d / "query_vectors_validation.npy").astype(np.float32)
n, dim = vecs.shape; rng = np.random.default_rng(0); Q = np.ascontiguousarray(qall[rng.choice(len(qall), size=min(5000, len(qall)), replace=False)])
out = d / "faiss"; out.mkdir(exist_ok=True); K = 100


def flat():
    ix = faiss.IndexFlatIP(dim); ix.add(vecs); return ix


def ivfpq(nlist=512, m=16, nbits=8):
    ix = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT); ix.train(vecs[rng.choice(n, size=min(n, 100_000), replace=False)]); ix.add(vecs); return ix


def hnsw(m=32, ef=200):
    ix = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT); ix.hnsw.efConstruction = ef; ix.add(vecs); return ix


built = {}
for name, fn in (("Flat (exact)", flat), ("IVF-PQ", ivfpq), ("HNSW", hnsw)):
    t = time.time(); built[name] = (fn(), time.time() - t); faiss.write_index(built[name][0], str(out / (name.split()[0].lower().replace("-", "") + ".index")))
_, exact = built["Flat (exact)"][0].search(Q, K)
rec = lambda ids: float(np.mean([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(ids, exact, strict=True)]))


def latency(ix, reps=800):
    faiss.omp_set_num_threads(1)
    for i in range(50): ix.search(Q[i:i + 1], K)
    ts = []
    for i in range(reps):
        t = time.perf_counter(); ix.search(Q[i:i + 1], K); ts.append((time.perf_counter() - t) * 1000)
    faiss.omp_set_num_threads(os.cpu_count()); return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


configs = [("Flat (exact)", "-", lambda ix: None)] + [("IVF-PQ", f"nprobe={p}", lambda ix, p=p: setattr(ix, "nprobe", p)) for p in (4, 16, 64)] + [("HNSW", f"efSearch={e}", lambda ix, e=e: setattr(ix.hnsw, "efSearch", e)) for e in (32, 64, 128, 256)]
rows = []
for name, params, setter in configs:
    ix, build_s = built[name]; setter(ix); _, ids = ix.search(Q, K); p50, p95 = latency(ix)
    rows.append({"index": name, "params": params, "recall_at_100_vs_exact": rec(ids), "p50_ms": p50, "p95_ms": p95, "size_mb": os.path.getsize(out / (name.split()[0].lower().replace("-", "") + ".index")) / 1e6, "build_seconds": build_s})
    print(rows[-1], flush=True)
(out / "results.json").write_text(json.dumps({"items": n, "dim": dim, "queries": len(Q), "results": rows}, indent=1))
