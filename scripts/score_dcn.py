"""Score a feature table with the frozen DCN-V2 + MMoE (modelling Step 11). Run: `uv run python scripts/score_dcn.py <table.parquet> <out.parquet>`.
Uses the training normalisation stored with the weights, so the serving path and the training path see the same inputs. Own process (PyTorch)."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from ranking.models.dcn_mmoe import DCNMTL, predict

cfg = json.loads((ROOT / "configs" / "dcn.json").read_text()); ck = torch.load(ROOT / cfg["out"] / "mmoe_best.pt", weights_only=False); p = ck["params"]
model = DCNMTL(n_num=ck["n_num"], n_cats=ck["n_cats"], n_tasks=3, mode="mmoe", n_cross=p["n_cross"], n_experts=p["n_experts"], expert_dim=p["expert_dim"]); model.load_state_dict(ck["state"]); model.eval(); torch.set_num_threads(4)
FE = [f for f in cfg["features"] if f != "tt_missing"]; df = pl.read_parquet(sys.argv[1], columns=["session", "aid", "category"] + FE).sort("session", maintain_order=True)
x = df.select(FE).to_numpy().astype(np.float32); miss = np.isnan(x[:, FE.index("tt_sim")]).astype(np.float32); x = np.nan_to_num(x, nan=0.0)
for c in cfg["log_columns"]: x[:, FE.index(c)] = np.log1p(np.clip(x[:, FE.index(c)], 0, None))
x = np.hstack([x, miss[:, None]]); mean, std = np.array(ck["stats"][0], np.float32), np.array(ck["stats"][1], np.float32)
pr = predict(model, torch.from_numpy((x - mean) / std), torch.from_numpy(df["category"].to_numpy().astype(np.int64)))
pl.DataFrame({"session": df["session"], "aid": df["aid"], "p_click": pr[:, 0], "p_cart": pr[:, 1], "p_order": pr[:, 2]}).write_parquet(sys.argv[2]); print("scored", len(df))
