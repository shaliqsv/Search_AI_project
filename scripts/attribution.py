"""Attribution comparison for DCN-V2 with MMoE (modelling Step 9, D18): leave-one-feature-group-out deltas against integrated gradients.
Run: `uv run python scripts/attribution.py`. Own process (PyTorch). Sanity checks: IG completeness, the removal test (does removing the group a method
calls most important move the score more than removing a random group?), agreement between the methods, stability under small input noise, latency."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from ranking.models.dcn_mmoe import DCNMTL, OTTO_WEIGHTS

cfg = json.loads((ROOT / "configs" / "dcn.json").read_text()); ck = torch.load(ROOT / cfg["out"] / "mmoe_best.pt", weights_only=False); p = ck["params"]
model = DCNMTL(n_num=ck["n_num"], n_cats=ck["n_cats"], n_tasks=3, mode="mmoe", n_cross=p["n_cross"], n_experts=p["n_experts"], expert_dim=p["expert_dim"]); model.load_state_dict(ck["state"]); model.eval(); torch.set_num_threads(4)
FE = cfg["features"]; names = [f for f in FE if f != "tt_missing"] + ["tt_missing"]; mean, std = np.array(ck["stats"][0], np.float32), np.array(ck["stats"][1], np.float32)
GROUPS = {"co-visitation": ["rank_covis", "covis_max", "covis_mean", "covis_wsum"], "popularity": ["log_clicks", "log_clicks_3d", "pop_cat_pct", "item_cat_size"], "cart and order rates": ["cart_rate", "order_rate"],
          "recency": ["hours_since_last_seen"], "session": ["prefix_len", "prefix_n_cart", "prefix_n_order", "prefix_share_same_cat"], "two-tower similarity": ["tt_sim", "tt_missing", "rank_tt"]}
gidx = {g: [names.index(f) for f in fs if f in names] for g, fs in GROUPS.items()}; w = torch.tensor(OTTO_WEIGHTS)
va = pl.read_parquet(ROOT / cfg["validation_table"]).sort("session", maintain_order=True); sess = va["session"].unique().sort().to_numpy(); rng = np.random.default_rng(0); pick = set(rng.choice(sess, 200, replace=False).tolist())
sub = va.filter(pl.col("session").is_in(list(pick)))
x = sub.select([f for f in FE if f != "tt_missing"]).to_numpy().astype(np.float32); miss = np.isnan(x[:, [f for f in FE if f != "tt_missing"].index("tt_sim")]).astype(np.float32); x = np.nan_to_num(x, nan=0.0)
for c in cfg["log_columns"]:
    j = [f for f in FE if f != "tt_missing"].index(c); x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
x = torch.from_numpy((np.hstack([x, miss[:, None]]) - mean) / std); cat = torch.from_numpy(sub["category"].to_numpy().astype(np.int64))


def score(xx, cc): return (torch.sigmoid(model(xx, cc)) * w).sum(1)


with torch.no_grad(): s_all = score(x, cat).numpy()
sub = sub.with_columns(pl.Series("score", s_all)); top = sub.with_row_index("i").sort(["session", "score"], descending=[False, True]).group_by("session", maintain_order=True).head(5)["i"].to_numpy()     # top 5 results of each of the 200 queries
xs, cs = x[top], cat[top]; base = torch.zeros_like(xs)                                                                                       # the standardised training mean is the baseline
gnames = list(GROUPS)


def logo(xx, cc):
    with torch.no_grad():
        s0 = score(xx, cc); out = []
        for g in gnames:
            xr = xx.clone(); xr[:, gidx[g]] = 0.0; out.append((s0 - score(xr, cc)).numpy())
    return np.stack(out, 1)


def ig(xx, cc, steps=32):
    total = torch.zeros_like(xx)
    for k in range(1, steps + 1):
        z = (k / steps) * xx; z.requires_grad_(True); score(z, cc).sum().backward(); total += z.grad
    a = (xx * total / steps).detach().numpy(); return np.stack([a[:, gidx[g]].sum(1) for g in gnames], 1), a.sum(1)


t0 = time.time(); L = logo(xs, cs); t_logo = (time.time() - t0) / len(xs) * 1000; t0 = time.time(); I, tot = ig(xs, cs); t_ig = (time.time() - t0) / len(xs) * 1000
with torch.no_grad(): delta = (score(xs, cs) - score(base, cs)).numpy()
complete = float(np.median(np.abs(tot - delta) / (np.abs(delta) + 1e-9)))
def removal_lift(A):
    rng2 = np.random.default_rng(1); best = A.argmax(1); rand = rng2.integers(0, len(gnames), len(A))
    with torch.no_grad():
        s0 = score(xs, cs); drops = []
        for choice in (best, rand):
            xr = xs.clone()
            for i, g in enumerate(choice): xr[i, gidx[gnames[g]]] = 0.0
            drops.append(np.abs((s0 - score(xr, cs)).numpy()).mean())
    return float(drops[0] / max(drops[1], 1e-12))
agree = float(np.nanmean([spearmanr(L[i], I[i]).statistic for i in range(len(L))]))
noise = xs + 0.05 * torch.randn_like(xs); Ln, (In, _) = logo(noise, cs), ig(noise, cs)
stab = lambda A, B: float(np.nanmean([spearmanr(A[i], B[i]).statistic for i in range(len(A))])); res = {"results_explained": len(xs), "ig_completeness_median_relative_error": complete, "removal_lift_logo": removal_lift(L), "removal_lift_ig": removal_lift(I), "agreement_spearman": agree,
       "stability_logo": stab(L, Ln), "stability_ig": stab(I, In), "latency_ms_logo": t_logo, "latency_ms_ig": t_ig, "groups": gnames}
(ROOT / cfg["out"] / "attribution.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
