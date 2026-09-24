"""Inference latency and size of the trained DCN-V2 + MMoE for one query of 300 candidates, on 1 and on 4 threads (modelling Step 10). Own process."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from ranking.models.dcn_mmoe import DCNMTL

cfg = json.loads((ROOT / "configs" / "dcn.json").read_text()); path = ROOT / cfg["out"] / "mmoe_best.pt"; ck = torch.load(path, weights_only=False); p = ck["params"]
m = DCNMTL(n_num=ck["n_num"], n_cats=ck["n_cats"], n_tasks=3, mode="mmoe", n_cross=p["n_cross"], n_experts=p["n_experts"], expert_dim=p["expert_dim"]); m.load_state_dict(ck["state"]); m.eval()
x = torch.randn(cfg["pool"], ck["n_num"]); c = torch.zeros(cfg["pool"], dtype=torch.long); out = {"parameters": sum(q.numel() for q in m.parameters()), "file_mb": path.stat().st_size / 1e6}
for th in (1, 4):
    torch.set_num_threads(th)
    with torch.no_grad():
        for _ in range(30): m(x, c)
        ts = []
        for _ in range(300):
            t = time.perf_counter(); m(x, c); ts.append((time.perf_counter() - t) * 1000)
    out[f"p50_ms_{th}_threads"], out[f"p95_ms_{th}_threads"] = float(np.percentile(ts, 50)), float(np.percentile(ts, 95))
(ROOT / cfg["out"] / "latency.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
