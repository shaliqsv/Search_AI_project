"""Train and score the Two-Tower variants (modelling Step 4). Run: `uv run python scripts/train_two_tower.py configs/two_tower.json`.

Runs in its own process (PyTorch and FAISS/LightGBM cannot share one on macOS). Training window = days before `train_end_day`; scored on the
validation ranking queries of Step 2. Writes per-variant item vectors, query vectors, weights and a results.json (metrics, timings)."""

import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from evalset import TOP, evaluate, prepare_eval, summarize
from queries import build_ranking_queries

from ranking.data.split import day_to_ms
from ranking.eval.metrics import recall_at_k
from ranking.models.two_tower import TwoTower, encode_queries, sampled_softmax_loss


def setup(cfg):
    torch.manual_seed(cfg["seed"]); torch.set_num_threads(4)
    data = ROOT / "data"; d_train = day_to_ms(cfg["train_end_day"])
    events = pl.scan_parquet(data / "sample" / "events.parquet")
    categories = pl.read_parquet(data / "modeling" / "categories" / "item_category.parquet")
    n_cats = len(json.loads((ROOT / "artifacts" / "category_names_v2.json").read_text())["categories"])
    stats = events.filter(pl.col("ts") < d_train).group_by("aid").agg(pl.len().alias("n"), (pl.col("type") == 0).sum().alias("clicks")).collect()
    catalog = categories.join(stats.filter(pl.col("n") >= cfg["min_freq"]), on="aid", how="inner").sort("aid")
    catalog_aids = catalog["aid"].to_numpy(); n_items = len(catalog_aids); index = {int(a): i for i, a in enumerate(catalog_aids)}
    item_cat = torch.from_numpy(catalog["cluster"].to_numpy().astype(np.int64)); item_pop = torch.from_numpy(np.minimum(15, np.floor(np.log2(1 + catalog["clicks"].to_numpy())).astype(np.int64)))
    item_cluster = dict(zip(categories["aid"].to_list(), categories["cluster"].to_list(), strict=True))
    train_events = events.filter(pl.col("ts") < d_train).select("session", "aid", "ts", "type").sort("session", "ts").collect(); assert train_events["ts"].max() < d_train
    train_q, _ = build_ranking_queries(train_events, item_cluster, cap=10**9, seed=0); del train_events
    keep = np.sort(np.random.default_rng(cfg["seed"]).choice(train_q.height, size=min(cfg["n_train_queries"], train_q.height), replace=False))
    train_prepared, _ = prepare_eval(train_q[keep]); assert all(q["prefix_ts_last"] < d_train for q in train_prepared)
    q_of, item_of, rel_sets = [], [], []
    for qi, q in enumerate(train_prepared):
        rel = {index[int(a)] for a in set().union(*q["labels"].values()) if int(a) in index}; rel_sets.append(rel)
        for j in rel: q_of.append(qi); item_of.append(j)
    q_of, item_of = np.array(q_of), np.array(item_of)
    hist, w, cat_onehot, _ = encode_queries(train_prepared, index, n_items, n_cats)
    counts = np.bincount(item_of, minlength=n_items).astype(np.float64); log_q = torch.from_numpy(np.log((counts + 1) / (counts.sum() + n_items))).float()
    rng = np.random.default_rng(cfg["seed"]); val_q = set(rng.choice(len(train_prepared), size=len(train_prepared) // 20, replace=False).tolist())
    is_val = np.array([qi in val_q for qi in q_of]); tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    prepared, _ = prepare_eval(pl.read_parquet(data / "modeling" / "ranking_queries_validation.parquet"))
    e_hist, e_w, e_cat, e_cats = encode_queries(prepared, index, n_items, n_cats)
    ic = item_cat.numpy(); by_cat = {c: np.flatnonzero(ic == c) for c in range(n_cats)}; order = np.argsort(ic, kind="stable"); sizes = np.bincount(ic, minlength=n_cats); offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return SimpleNamespace(catalog_aids=catalog_aids, n_items=n_items, n_cats=n_cats, n_pop=16, index=index, item_cat=item_cat, item_pop=item_pop, hist=hist, w=w, cat=cat_onehot, q_of=q_of, item_of=item_of, rel_sets=rel_sets, log_q=log_q,
                           tr_idx=tr_idx, va_idx=va_idx, prepared=prepared, e_hist=e_hist, e_w=e_w, e_cat=e_cat, e_cats=e_cats, by_cat=by_cat, cat_order=order, cat_sizes=sizes, cat_offsets=offsets, n_train_queries=len(train_prepared), n_examples=len(q_of))


def hard_negatives(ns, pos_items, rng, per_row=1):
    cats = ns.item_cat.numpy()[pos_items]; pick = (rng.random((len(pos_items), per_row)) * ns.cat_sizes[cats][:, None]).astype(np.int64)
    return ns.cat_order[ns.cat_offsets[cats][:, None] + pick]


def loss_with_hard(q, i, item_idx, hard_vec, hard_idx, hard_ok, log_q, hard_log_q, temperature=0.07):
    in_batch = q @ i.T / temperature - log_q[item_idx].unsqueeze(0)
    same = item_idx.unsqueeze(0) == item_idx.unsqueeze(1); in_batch = in_batch.masked_fill(same & ~torch.eye(len(item_idx), dtype=torch.bool), float("-inf"))
    hard = torch.einsum("bd,bhd->bh", q, hard_vec) / temperature - hard_log_q; hard = hard.masked_fill(~hard_ok | (hard_idx == item_idx.unsqueeze(1)), float("-inf"))
    return torch.nn.functional.cross_entropy(torch.cat([in_batch, hard], dim=1), torch.arange(len(item_idx)))


def train(ns, cfg, hard, label, lr, bs, temp, epochs):
    seed = cfg["seed"]; torch.manual_seed(seed); model = TwoTower(ns.n_items, ns.n_cats, ns.n_pop); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng, hrng = np.random.default_rng(seed), np.random.default_rng(seed + 1); cat_log_size = torch.from_numpy(-np.log(ns.cat_sizes)).float()
    history, best, start = [], (float("inf"), None, 0), time.time()
    def step_loss(ids, hn):
        qi, it = torch.from_numpy(ns.q_of[ids]), torch.from_numpy(ns.item_of[ids])
        q = model.query(ns.hist[qi], ns.w[qi], ns.cat[qi]); i = model.item(it, ns.item_cat[it], ns.item_pop[it])
        if not hard: return sampled_softmax_loss(q, i, it, log_q=ns.log_q, temperature=temp)
        h = torch.from_numpy(hn); hv = model.item(h.reshape(-1), ns.item_cat[h.reshape(-1)], ns.item_pop[h.reshape(-1)]).reshape(len(ids), hard, -1)
        ok = torch.tensor([[int(h[r, k]) not in ns.rel_sets[int(qi[r])] for k in range(hard)] for r in range(len(ids))])
        return loss_with_hard(q, i, it, hv, h, ok, ns.log_q, cat_log_size[ns.item_cat[h]], temperature=temp)
    for epoch in range(epochs):
        model.train(); order = rng.permutation(ns.tr_idx); losses = []; t_ep = time.time()
        for s in range(0, len(order) - bs + 1, bs):
            ids = order[s:s + bs]; loss = step_loss(ids, hard_negatives(ns, ns.item_of[ids], hrng, hard) if hard else None); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vo = rng.permutation(ns.va_idx); v = [step_loss(vo[s:s + bs], hard_negatives(ns, ns.item_of[vo[s:s + bs]], hrng, hard) if hard else None).item() for s in range(0, min(len(vo), 20_000) - bs + 1, bs)]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": float(np.mean(v)), "seconds": time.time() - t_ep})
        if history[-1]["validation_loss"] < best[0]: best = (history[-1]["validation_loss"], {k: x.clone() for k, x in model.state_dict().items()}, epoch + 1)
        print(f"{label} epoch {epoch + 1}: train {history[-1]['train_loss']:.3f}, validation {history[-1]['validation_loss']:.3f} ({history[-1]['seconds']:.0f} s)", flush=True)
    model.load_state_dict(best[1]); return model, history, time.time() - start, best[2]


def retrieve(ns, model, topn=200):
    model.eval()
    with torch.no_grad():
        item_vec = model.item(torch.arange(ns.n_items), ns.item_cat, ns.item_pop)
        qvec = torch.cat([model.query(ns.e_hist[s:s + 4096], ns.e_w[s:s + 4096], ns.e_cat[s:s + 4096]) for s in range(0, len(ns.prepared), 4096)])
    out = {}
    for c in range(ns.n_cats):
        rows = np.flatnonzero(ns.e_cats == c)
        if not len(rows): continue
        members = ns.by_cat[c]; top = torch.topk(qvec[rows] @ item_vec[members].T, min(topn + 60, len(members)), dim=1).indices.numpy()
        for r, t in zip(rows, top, strict=True):
            prefix = ns.prepared[r]["prefix"]; out[ns.prepared[r]["session"]] = [a for a in (int(x) for x in ns.catalog_aids[members[t]]) if a not in prefix][:topn]
    return out, item_vec.numpy(), qvec.numpy()


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text()); out = ROOT / cfg["out"]; out.mkdir(parents=True, exist_ok=True); t0 = time.time(); ns = setup(cfg)
    print(f"setup {time.time() - t0:.0f} s: catalog {ns.n_items:,} items (min frequency {cfg['min_freq']}), {ns.n_train_queries:,} training queries, {ns.n_examples:,} examples, {len(ns.prepared):,} validation queries", flush=True)
    results = []
    for v in cfg["variants"]:
        model, history, seconds, best_epoch = train(ns, cfg, v["hard"], v["name"], v.get("lr", cfg["lr"]), v.get("batch_size", cfg["batch_size"]), v.get("temperature", 0.07), v.get("epochs", cfg["epochs"])); lists, item_vec, qvec = retrieve(ns, model)
        res = evaluate(ns.prepared, lambda q, lists=lists: lists[q["session"]][:TOP]); s = summarize(res)
        d = out / v["name"].replace(" ", "_").replace("+", "plus").replace("-", "_").replace("/", "_"); d.mkdir(exist_ok=True); res.write_parquet(d / "validation_results.parquet"); tune_ids = set(json.loads((ROOT / cfg["tune_sessions"]).read_text())) if cfg.get("tune_sessions") else set(); ndcg_tune = float(res.filter(pl.col("session").is_in(list(tune_ids)))["ndcg10"].mean()) if tune_ids else float("nan")
        rec = {k: float(np.mean([recall_at_k(lists[q["session"]], set().union(*q["labels"].values()), k=k) for q in ns.prepared])) for k in (20, 50, 100, 200)}
        np.save(d / "item_embeddings.npy", item_vec.astype(np.float32)); np.save(d / "query_vectors_validation.npy", qvec.astype(np.float32)); np.save(d / "catalog_aids.npy", ns.catalog_aids)
        np.save(d / "validation_sessions.npy", np.array([q["session"] for q in ns.prepared])); torch.save(model.state_dict(), d / "model.pt")
        results.append({"variant": v["name"], "hard": v["hard"], "params": {k: v[k] for k in ("lr", "batch_size", "temperature", "epochs") if k in v}, "ndcg10_tuning": ndcg_tune, "ndcg10": s["ndcg10"][0], "ndcg10_low": s["ndcg10"][1], "ndcg10_high": s["ndcg10"][2], **{f"recall{k}": x for k, x in rec.items()}, "train_seconds": seconds, "seconds_per_epoch": seconds / v.get("epochs", cfg["epochs"]), "best_epoch": best_epoch, "history": history, "dir": str(d.relative_to(ROOT))})
        print(f"{v['name']}: NDCG@10 {s['ndcg10'][0]:.4f} [{s['ndcg10'][1]:.4f}, {s['ndcg10'][2]:.4f}]; " + ", ".join(f"recall@{k} {x:.4f}" for k, x in rec.items()), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    (out / "results.json").write_text(json.dumps({"config": cfg, "catalog_items": ns.n_items, "train_queries": ns.n_train_queries, "examples": ns.n_examples, "validation_queries": len(ns.prepared), "peak_memory_gb": peak, "results": results}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
