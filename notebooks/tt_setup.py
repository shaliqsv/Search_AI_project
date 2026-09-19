"""Shared setup, training and scoring for the Two-Tower experiments (issues #24, #25, #26).

Import from a notebook in this folder with `from tt_setup import setup, train, score`.
Stays here, not in `src/ranking/`, until issue #52. It repeats what notebook 24 does, so that
later experiments use exactly the same data, catalog and evaluation.
"""

import json
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from evalset import TOP, evaluate, prepare_eval, summarize
from metrics import recall_at_k
from queries import build_ranking_queries
from split import day_to_ms
from twotower import TwoTower, encode_queries, sampled_softmax_loss

D14 = day_to_ms(14)
SEED = 0


def setup(data, artifacts, n_train_queries=400_000, seed=SEED):
    """Catalog, training examples, evaluation queries. Same choices as notebook 24."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    events = pl.scan_parquet(data / "sample" / "events.parquet")
    categories = pl.read_parquet(data / "categories" / "item_category.parquet")
    n_cats = len(json.loads((artifacts / "category_names.json").read_text())["categories"])
    stats = events.filter(pl.col("ts") < D14).group_by("aid").agg(pl.len().alias("n"), (pl.col("type") == 0).sum().alias("clicks")).collect()
    catalog = categories.join(stats.filter(pl.col("n") >= 5), on="aid", how="inner").sort("aid")
    catalog_aids = catalog["aid"].to_numpy()
    n_items = len(catalog_aids)
    index = {int(a): i for i, a in enumerate(catalog_aids)}
    item_cat = torch.from_numpy(catalog["cluster"].to_numpy().astype(np.int64))
    item_pop = torch.from_numpy(np.minimum(15, np.floor(np.log2(1 + catalog["clicks"].to_numpy())).astype(np.int64)))
    item_cluster = dict(zip(categories["aid"].to_list(), categories["cluster"].to_list(), strict=True))

    train_events = events.filter(pl.col("ts") < D14).select("session", "aid", "ts", "type").sort("session", "ts").collect()
    assert train_events["ts"].max() < D14
    train_q, _ = build_ranking_queries(train_events, item_cluster, cap=10**9, seed=0)
    del train_events
    keep = np.sort(np.random.default_rng(seed).choice(train_q.height, size=min(n_train_queries, train_q.height), replace=False))
    train_prepared, _ = prepare_eval(train_q[keep])
    assert all(q["prefix_ts_last"] < D14 for q in train_prepared)

    q_of, item_of, rel_sets = [], [], []
    for qi, q in enumerate(train_prepared):
        rel = {index[int(a)] for a in set().union(*q["labels"].values()) if int(a) in index}
        rel_sets.append(rel)
        for j in rel:
            q_of.append(qi)
            item_of.append(j)
    q_of, item_of = np.array(q_of), np.array(item_of)
    hist, w, cat_onehot, _ = encode_queries(train_prepared, index, n_items, n_cats)
    counts = np.bincount(item_of, minlength=n_items).astype(np.float64)
    log_q = torch.from_numpy(np.log((counts + 1) / (counts.sum() + n_items))).float()
    rng = np.random.default_rng(seed)
    val_q = set(rng.choice(len(train_prepared), size=len(train_prepared) // 20, replace=False).tolist())
    is_val = np.array([qi in val_q for qi in q_of])
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    eval_q = pl.read_parquet(data / "eval" / "ranking_queries.parquet")
    prepared, prep_stats = prepare_eval(eval_q)
    e_hist, e_w, e_cat, e_cats = encode_queries(prepared, index, n_items, n_cats)
    ic = item_cat.numpy()
    by_cat = {c: np.flatnonzero(ic == c) for c in range(n_cats)}
    order = np.argsort(ic, kind="stable")
    sizes = np.bincount(ic, minlength=n_cats)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return SimpleNamespace(
        catalog_aids=catalog_aids, n_items=n_items, n_cats=n_cats, n_pop=16, index=index,
        item_cat=item_cat, item_pop=item_pop, hist=hist, w=w, cat=cat_onehot,
        q_of=q_of, item_of=item_of, rel_sets=rel_sets, log_q=log_q, tr_idx=tr_idx, va_idx=va_idx,
        prepared=prepared, prep_stats=prep_stats, e_hist=e_hist, e_w=e_w, e_cat=e_cat, e_cats=e_cats,
        by_cat=by_cat, cat_order=order, cat_sizes=sizes, cat_offsets=offsets,
        n_train_queries=len(train_prepared), n_examples=len(q_of),
    )


def hard_negatives(ns, pos_items, rng, per_row=1):
    """Random items of the same category as each positive: (B, per_row) catalog indices."""
    cats = ns.item_cat.numpy()[pos_items]
    pick = (rng.random((len(pos_items), per_row)) * ns.cat_sizes[cats][:, None]).astype(np.int64)
    return ns.cat_order[ns.cat_offsets[cats][:, None] + pick]


def loss_with_hard(q, i, item_idx, hard_vec, hard_idx, hard_ok, log_q, hard_log_q, temperature=0.07):
    """In-batch softmax plus per-row hard negatives (hard_vec: (B, H, d)). hard_ok False = false negative."""
    in_batch = q @ i.T / temperature - log_q[item_idx].unsqueeze(0)
    same = item_idx.unsqueeze(0) == item_idx.unsqueeze(1)
    in_batch = in_batch.masked_fill(same & ~torch.eye(len(item_idx), dtype=torch.bool), float("-inf"))
    hard = torch.einsum("bd,bhd->bh", q, hard_vec) / temperature - hard_log_q
    hard = hard.masked_fill(~hard_ok | (hard_idx == item_idx.unsqueeze(1)), float("-inf"))
    logits = torch.cat([in_batch, hard], dim=1)
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(item_idx)))


def train(ns, hard=0, logq=True, epochs=4, batch_size=2048, lr=2e-3, seed=SEED, label=""):
    """Train a Two-Tower. hard = number of same-category hard negatives per example (0 = in-batch only)."""
    torch.manual_seed(seed)
    model = TwoTower(ns.n_items, ns.n_cats, ns.n_pop)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hrng = np.random.default_rng(seed + 1)
    log_q = ns.log_q if logq else torch.zeros_like(ns.log_q)
    cat_log_size = torch.from_numpy(-np.log(ns.cat_sizes)).float()   # uniform sampling inside a category
    history, best = [], (float("inf"), None, 0)
    start = time.time()

    def step_loss(ids, hn):
        qi = torch.from_numpy(ns.q_of[ids]); it = torch.from_numpy(ns.item_of[ids])
        q = model.query(ns.hist[qi], ns.w[qi], ns.cat[qi])
        i = model.item(it, ns.item_cat[it], ns.item_pop[it])
        if not hard:
            return sampled_softmax_loss(q, i, it, log_q=log_q if logq else None)
        h = torch.from_numpy(hn)
        hv = model.item(h.reshape(-1), ns.item_cat[h.reshape(-1)], ns.item_pop[h.reshape(-1)]).reshape(len(ids), hard, -1)
        ok = torch.tensor([[int(h[r, k]) not in ns.rel_sets[int(qi[r])] for k in range(hard)] for r in range(len(ids))])
        hlq = cat_log_size[ns.item_cat[h]] if logq else torch.zeros(len(ids), hard)
        return loss_with_hard(q, i, it, hv, h, ok, log_q, hlq)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(ns.tr_idx)
        losses = []
        for s in range(0, len(order) - batch_size + 1, batch_size):
            ids = order[s : s + batch_size]
            hn = hard_negatives(ns, ns.item_of[ids], hrng, hard) if hard else None
            loss = step_loss(ids, hn)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vo = rng.permutation(ns.va_idx)
            v = [step_loss(vo[s : s + batch_size], hard_negatives(ns, ns.item_of[vo[s : s + batch_size]], hrng, hard) if hard else None).item()
                 for s in range(0, min(len(vo), 20_000) - batch_size + 1, batch_size)]
        history.append((epoch + 1, float(np.mean(losses)), float(np.mean(v))))
        if history[-1][2] < best[0]:
            best = (history[-1][2], {k: x.clone() for k, x in model.state_dict().items()}, epoch + 1)
        print(f"{label} epoch {epoch + 1}: train loss {history[-1][1]:.3f}, validation loss {history[-1][2]:.3f}")
    model.load_state_dict(best[1])
    seconds = time.time() - start
    print(f"{label} best epoch {best[2]}, trained in {seconds:.0f} s")
    return model, history, seconds


def retrieve(ns, model, topn=200):
    model.eval()
    with torch.no_grad():
        item_vec = model.item(torch.arange(ns.n_items), ns.item_cat, ns.item_pop)
        qvec = torch.cat([model.query(ns.e_hist[s : s + 4096], ns.e_w[s : s + 4096], ns.e_cat[s : s + 4096]) for s in range(0, len(ns.prepared), 4096)])
    out = {}
    for c in range(ns.n_cats):
        rows = np.flatnonzero(ns.e_cats == c)
        if not len(rows):
            continue
        members = ns.by_cat[c]
        scores = qvec[rows] @ item_vec[members].T
        top = torch.topk(scores, min(topn + 60, len(members)), dim=1).indices.numpy()
        for r, t in zip(rows, top, strict=True):
            prefix = ns.prepared[r]["prefix"]
            lst = []
            for a in ns.catalog_aids[members[t]]:
                if int(a) not in prefix:
                    lst.append(int(a))
                    if len(lst) == topn:
                        break
            out[ns.prepared[r]["session"]] = lst
    return out, item_vec.numpy()


def score(ns, model, name, ks=(20, 50, 100, 200)):
    lists, item_vec = retrieve(ns, model)
    res = evaluate(ns.prepared, lambda q: lists[q["session"]][:TOP])
    rec = {k: float(np.mean([recall_at_k(lists[q["session"]], set().union(*q["labels"].values()), k=k) for q in ns.prepared])) for k in ks}
    s = summarize(res)
    print(f"{name}: NDCG@10 {s['ndcg10'][0]:.4f} [{s['ndcg10'][1]:.4f}, {s['ndcg10'][2]:.4f}]; recall " + ", ".join(f"@{k} {v:.4f}" for k, v in rec.items()))
    return res, rec, s, item_vec, lists
