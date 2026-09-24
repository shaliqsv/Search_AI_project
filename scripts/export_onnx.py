"""Export the neural models to ONNX and check parity with PyTorch (modelling Step 11). Run: `uv run python scripts/export_onnx.py <out dir> <Two-Tower model dir>`.
Exports DCN-V2 + MMoE (scores of a batch of candidates) and the Two-Tower query tower; each export is checked against the training framework on real-shaped
random inputs (maximum absolute difference must be at most 1e-4). Own process (PyTorch and onnxruntime)."""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from ranking.models.dcn_mmoe import DCNMTL
from ranking.models.two_tower import MAX_HIST, TwoTower

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); tt_dir = Path(sys.argv[2]); res = {}
cfg = json.loads((ROOT / "configs" / "dcn.json").read_text()); ck = torch.load(ROOT / cfg["out"] / "mmoe_best.pt", weights_only=False); p = ck["params"]
m = DCNMTL(n_num=ck["n_num"], n_cats=ck["n_cats"], n_tasks=3, mode="mmoe", n_cross=p["n_cross"], n_experts=p["n_experts"], expert_dim=p["expert_dim"]); m.load_state_dict(ck["state"]); m.eval()
x = torch.randn(64, ck["n_num"]); c = torch.randint(0, ck["n_cats"], (64,))
torch.onnx.export(m, (x, c), str(out / "dcn_mmoe.onnx"), input_names=["x_num", "category"], output_names=["logits"], dynamic_axes={"x_num": {0: "n"}, "category": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
s = ort.InferenceSession(str(out / "dcn_mmoe.onnx")); xb = torch.randn(300, ck["n_num"]); cb = torch.randint(0, ck["n_cats"], (300,))
with torch.no_grad(): ref = m(xb, cb).numpy()
res["dcn_mmoe_max_abs_diff"] = float(np.abs(s.run(None, {"x_num": xb.numpy(), "category": cb.numpy()})[0] - ref).max())
catalog = np.load(tt_dir / "catalog_aids.npy"); n_cats = len(json.loads((ROOT / "artifacts" / "category_names_v2.json").read_text())["categories"])
tt = TwoTower(len(catalog), n_cats, 16); tt.load_state_dict(torch.load(tt_dir / "model.pt")); tt.eval()
class QueryTower(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, hist_idx, hist_w, cat_dist): return self.m.query(hist_idx, hist_w, cat_dist)
qt = QueryTower(tt); h = torch.randint(0, len(catalog) + 1, (8, MAX_HIST)); w = torch.rand(8, MAX_HIST); cd = torch.zeros(8, n_cats); cd[:, 3] = 1
torch.onnx.export(qt, (h, w, cd), str(out / "two_tower_query.onnx"), input_names=["hist_idx", "hist_w", "cat_dist"], output_names=["query_vector"], dynamic_axes={"hist_idx": {0: "n"}, "hist_w": {0: "n"}, "cat_dist": {0: "n"}, "query_vector": {0: "n"}}, opset_version=17, dynamo=False)
s2 = ort.InferenceSession(str(out / "two_tower_query.onnx"))
with torch.no_grad(): ref2 = qt(h, w, cd).numpy()
res["two_tower_query_max_abs_diff"] = float(np.abs(s2.run(None, {"hist_idx": h.numpy(), "hist_w": w.numpy(), "cat_dist": cd.numpy()})[0] - ref2).max())
res["files_mb"] = {f.name: f.stat().st_size / 1e6 for f in out.glob("*.onnx")}; (out / "onnx_parity.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
assert res["dcn_mmoe_max_abs_diff"] <= 1e-4 and res["two_tower_query_max_abs_diff"] <= 1e-4
