"""Train DCN-V2 with MMoE and its ablations (modelling Step 7). Run: `uv run python scripts/train_dcn.py configs/dcn.json`.

Own process (PyTorch cannot share one with LightGBM or FAISS on macOS). Reads the reranker-training table (days 8-9) and the validation table written in Step 6.
Every model family gets the same budget (`trials`): MMoE, shared-bottom multi-task, and single-task models (one per head). Trials are scored on the tuning
sessions only (never on the scoring sessions). Writes validation predictions, the best configurations, timings and the best MMoE weights."""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from ranking.models.dcn_mmoe import DCNMTL, OTTO_WEIGHTS, blend, ndcg10_pool, predict, train_model

cfg = json.loads(Path(sys.argv[1]).read_text()); torch.set_num_threads(4); SEED = cfg["seed"]; POOL = cfg["pool"]
FE, LOGCOLS, TASKS = cfg["features"], cfg["log_columns"], ["y_click", "y_cart", "y_order"]
out = ROOT / cfg["out"]; out.mkdir(parents=True, exist_ok=True); tune_sessions = set(json.loads((ROOT / cfg["tune_sessions"]).read_text()))


def load(path):
    df = pl.read_parquet(path, columns=["session", "aid", "category", "label"] + TASKS + [f for f in FE if f != "tt_missing"]).sort("session", maintain_order=True); return df


def matrix(df, stats=None):
    x = df.select([f for f in FE if f != "tt_missing"]).to_numpy().astype(np.float32); names = [f for f in FE if f != "tt_missing"]
    miss = np.isnan(x[:, names.index("tt_sim")]).astype(np.float32) if "tt_sim" in names else np.zeros(len(x), np.float32); x = np.nan_to_num(x, nan=0.0)
    for c in LOGCOLS:
        if c in names: x[:, names.index(c)] = np.log1p(np.clip(x[:, names.index(c)], 0, None))
    x = np.hstack([x, miss[:, None]])
    if stats is None: stats = (x.mean(0), x.std(0)); stats[1][stats[1] < 1e-6] = 1.0
    return torch.from_numpy((x - stats[0]) / stats[1]), stats


def tensors(df, stats=None):
    x, stats = matrix(df, stats); return x, torch.from_numpy(df["category"].to_numpy().astype(np.int64)), torch.from_numpy(df.select(TASKS).to_numpy().astype(np.float32)), stats


t0 = time.time(); tr_all, va = load(ROOT / cfg["train_table"]), load(ROOT / cfg["validation_table"])
rng = np.random.default_rng(SEED); sess = tr_all["session"].unique().sort().to_numpy(); hold = set(rng.choice(sess, size=len(sess) // 10, replace=False).tolist())
m_hold = tr_all["session"].is_in(list(hold)).to_numpy(); tr, ho = tr_all.filter(~pl.Series(m_hold)), tr_all.filter(pl.Series(m_hold))
x_tr, c_tr, y_tr, stats = tensors(tr); x_ho, c_ho, y_ho, _ = tensors(ho, stats); x_va, c_va, y_va, _ = tensors(va, stats)
va_tune = torch.from_numpy(va["session"].is_in(list(tune_sessions)).to_numpy()); n_cats = int(max(tr_all["category"].max(), va["category"].max())) + 1; n_num = x_tr.shape[1]
print(f"setup {time.time() - t0:.0f} s: train rows {len(x_tr):,}, early-stopping rows {len(x_ho):,}, validation rows {len(x_va):,}, features {n_num}", flush=True)
lab_va = va["label"].to_numpy()
def sample(r):
    tw = [1.0, 1.0, 1.0] if r.random() < 0.5 else [1.0, 2.0, 4.0]
    return {"n_cross": int(r.choice([1, 2, 3])), "n_experts": int(r.choice([2, 4, 8])), "expert_dim": int(r.choice([32, 64])), "lr": float(np.exp(r.uniform(np.log(1e-3), np.log(4e-3)))), "tw": tw}


def fit(mode, p, tasks):
    """tasks: list of head indices trained by this model (all three for MMoE / shared-bottom, one for a single-task model)."""
    torch.manual_seed(SEED); model = DCNMTL(n_num=n_num, n_cats=n_cats, n_tasks=len(tasks), mode=mode, n_cross=p["n_cross"], n_experts=p["n_experts"], expert_dim=p["expert_dim"])
    tw = [p["tw"][t] for t in tasks]; _hist, best_epoch, sec = train_model(model, x_tr, c_tr, y_tr[:, tasks], x_ho, c_ho, y_ho[:, tasks], task_weights=tw, epochs=cfg["epochs"], batch=4096, lr=p["lr"], seed=SEED, label=f"{mode}{tasks}")
    return model, sec, best_epoch, sum(q.numel() for q in model.parameters())


def probs_full(models):
    """Validation probabilities (N, 3) from a list of (model, tasks)."""
    out = np.zeros((len(x_va), 3), dtype=np.float32)
    for model, tasks in models: out[:, tasks] = predict(model, x_va, c_va)
    return out


def tuning_score(p3, w=OTTO_WEIGHTS):
    m = va_tune.numpy(); return ndcg10_pool(blend(p3[m], w), lab_va[m], pool=POOL)


results, trials, preds = {}, [], {}
FAMILIES = {"MMoE": ("mmoe", [[0, 1, 2]]), "shared-bottom multi-task": ("shared", [[0, 1, 2]]), "single-task (one model per head)": ("mmoe", [[0], [1], [2]])}
for fam, (mode, groups) in FAMILIES.items():
    r = np.random.default_rng(SEED + 7); best = None
    for trial in range(cfg["trials"]):
        p = sample(r); models, secs, npar = [], 0.0, 0
        for tasks in groups:
            m, sec, be, n = fit(mode if fam != "single-task (one model per head)" else "shared", p, tasks); models.append((m, tasks)); secs += sec; npar += n
        p3 = probs_full(models); sc = tuning_score(p3); trials.append({"family": fam, "trial": trial, "params": json.dumps(p), "ndcg10_tuning": sc, "train_seconds": secs, "parameters": npar}); print(f"{fam} trial {trial}: tuning NDCG@10 {sc:.4f} ({secs:.0f} s)", flush=True)
        if best is None or sc > best[0]: best = (sc, p, p3, models, secs, npar)
    results[fam] = {"params": best[1], "tuning_ndcg10": best[0], "train_seconds": best[4], "parameters": best[5], "seconds_per_epoch": best[4] / cfg["epochs"] / len(groups)}; preds[fam] = best[2]
    if fam == "MMoE": torch.save({"state": best[3][0][0].state_dict(), "params": best[1], "n_num": n_num, "n_cats": n_cats, "stats": [s.tolist() for s in stats]}, out / "mmoe_best.pt")
# blend weights for the best MMoE: OTTO default against a small grid tuned on the tuning part
grid = [(a, b, round(1 - a - b, 2)) for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01 - a, 0.1)]; a_ = np.round
gs = sorted(((tuning_score(preds["MMoE"], tuple(float(x) for x in w)), tuple(float(x) for x in w)) for w in grid), reverse=True)
results["blend"] = {"otto_default": OTTO_WEIGHTS, "otto_default_tuning_ndcg10": tuning_score(preds["MMoE"]), "best_weights": gs[0][1], "best_tuning_ndcg10": gs[0][0]}
for fam, p3 in preds.items(): pl.DataFrame({"session": va["session"], "aid": va["aid"], "p_click": p3[:, 0], "p_cart": p3[:, 1], "p_order": p3[:, 2]}).write_parquet(out / f"validation_predictions_{fam.split()[0].replace('-', '_')}.parquet")
# per-head validation AUC/BCE for the seesaw check (orders are rare)
from sklearn.metrics import roc_auc_score

seesaw = {fam: [float(roc_auc_score(y_va[:, t].numpy(), p3[:, t])) for t in range(3)] for fam, p3 in preds.items()}
(out / "results.json").write_text(json.dumps({"config": cfg, "results": results, "trials": trials, "head_auc": seesaw, "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, "rows": {"train": len(x_tr), "validation": len(x_va)}}, indent=1))
print("done", flush=True)
