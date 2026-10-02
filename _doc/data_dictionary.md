# Data dictionary — Phase 1 frozen snapshot

Version: see `data/frozen/manifest.json` (`version`, per-file `sha256`). Not committed to
git (`data/` is gitignored — too large to push reliably; regenerate with
`uv run python scripts/build_phase1_snapshot.py` against `data/sample/events.parquet`,
itself produced by `notebooks/kaggle_otto_sample`). Source: OTTO recsys dataset, CC BY 4.0
(`_doc/decisions/D1-otto-license.md`).

**Split is by timestamp, not session.** OTTO "sessions" can span the entire 4-week window
(median span ~7.4 days, 51% > 7 days — see Phase 1 issue #2's correction), so a session's
own id says nothing about which split it belongs to. Every session contributes exactly
one row to `ranking_examples.parquet` (its own leave-last-out example), and that row's
`split` is decided purely from `label_ts` against the cutoffs in `manifest.json`
(`split_cutoffs_ms`): train = first 2 weeks by ts, val = week 3, test = week 4.

## `events.parquet` (26,047,712 rows)
Real OTTO session events, session-hash-sampled at ~12% (`data/sample/sample_manifest.json`).

| column  | type  | meaning |
|---------|-------|---------|
| session | Int64 | session id (real OTTO id, not reassigned) |
| aid     | Int32 | item id |
| ts      | Int64 | Unix ms |
| type    | Int8  | 0=click, 1=cart, 2=order (`src/ranking/data/event_types.py`) |

## `item_categories.parquet` (1,414,089 rows — one per aid seen anywhere in the sample)
Synthetic categories (D4-equivalent): 80-means clustering of a truncated-SVD embedding of
co-visitation, fit on **events with ts before the train/val cutoff only** (any session —
13,064,647 events, frequency ≥ 5). Not a real taxonomy — a documented proxy (see Phase 0
issue #1, `_doc/outdate/plan.md` D4).

| column   | type  | meaning |
|----------|-------|---------|
| aid      | Int32 | item id |
| category | Int32 | 0-79, or **-1** = too sparse to place (< 5 occurrences before the cutoff, or never seen there). 72.0% of all 1,414,089 items catalog-wide; only 12.3% of train examples / 18.5% of val examples end up with an unknown query (held-out items skew popular; see Phase 3 issue for the train→val drift). Among placed items, categories are balanced (largest 2.6% of known items, median ~4,600) after fixing a clustering collapse — see Phase 3 issue and `categories.py`'s normalization. |

## `ranking_examples.parquet` (1,547,984 rows — one per session)
Unit of prediction (Phase 0 issue #1): leave-last-out per session. Held-out = the session's
last event by ts; query = that item's category; prefix = every earlier event.

| column         | type   | meaning |
|----------------|--------|---------|
| session        | Int64  | session id |
| prefix_last_ts | Int64  | ts of the last prefix event — nothing after this may be used as a feature |
| query_category | Int32  | category of the held-out item (-1 if unknown, 17.4% of examples) |
| label_aid      | Int32  | the held-out item |
| label_type     | Int8   | held-out event's type (0/1/2) |
| label_grade    | Int64  | graded relevance of `label_aid`: click=1, cart=2, order=3 |
| label_ts       | Int64  | the held-out event's own ts — **this, not session id, decides `split`** |
| prefix_len     | UInt32 | number of events in the prefix |
| split          | str    | "train" (512,161) / "val" (396,786) / "test" (639,037), from `label_ts` vs cutoffs |

## `train_covisitation.parquet` (17,292,646 rows)
Undirected item-item co-visitation weight, from events before the train/val cutoff only,
sliding window of 5 subsequent events per session, weight = mean of the two events' graded
weights. Not pruned to top-K per item yet — that's Phase 4 (feature engineering) work.

| column | type    | meaning |
|--------|---------|---------|
| aid_a  | Int32   | lower item id of the pair |
| aid_b  | Int32   | higher item id of the pair |
| weight | Float64 | summed co-visitation weight |
