"""Generate a small, fully synthetic OTTO-shaped dataset for local development.

Not real OTTO data - just a stand-in with the same schema (session, aid, ts, type)
so pipeline code (splitting, co-visitation, category clustering, features, eval)
can be built and tested without the 12GB real file or Kaggle compute.

Each item is assigned a hidden "true" category so co-visitation/clustering code
has real structure to recover during development. Real OTTO event-type rates
(clicks ~89.9%, carts ~7.8%, orders ~2.3%) are used to pick session length and
the type funnel (click -> maybe cart -> maybe order).

Usage: uv run python scripts/make_mock_otto.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

SEED = 0
N_SESSIONS = 2000
N_ITEMS = 300
N_HIDDEN_CATEGORIES = 20
N_WEEKS = 4
MS_PER_WEEK = 7 * 24 * 60 * 60 * 1000
START_TS = 1_659_312_000_000  # 2022-08-01T00:00:00Z, matches real OTTO's window

CART_AFTER_CLICK_P = 0.10
ORDER_AFTER_CART_P = 0.30

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"


def main() -> None:
    rng = np.random.default_rng(SEED)

    item_category = rng.integers(0, N_HIDDEN_CATEGORIES, size=N_ITEMS)
    items_by_category = {c: np.where(item_category == c)[0] for c in range(N_HIDDEN_CATEGORIES)}

    sessions = []
    for session_id in range(N_SESSIONS):
        primary_cat = int(rng.integers(0, N_HIDDEN_CATEGORIES))
        n_events = int(rng.integers(3, 30))
        week = int(rng.integers(0, N_WEEKS))
        ts = START_TS + week * MS_PER_WEEK + int(rng.integers(0, MS_PER_WEEK))

        events = []
        for _ in range(n_events):
            if rng.random() < 0.8:
                aid = int(rng.choice(items_by_category[primary_cat]))
            else:
                aid = int(rng.integers(0, N_ITEMS))

            ts += int(rng.integers(1_000, 300_000))  # 1s-5min between events
            events.append({"aid": aid, "ts": ts, "type": "clicks"})

            if rng.random() < CART_AFTER_CLICK_P:
                ts += int(rng.integers(1_000, 60_000))
                events.append({"aid": aid, "ts": ts, "type": "carts"})
                if rng.random() < ORDER_AFTER_CART_P:
                    ts += int(rng.integers(1_000, 60_000))
                    events.append({"aid": aid, "ts": ts, "type": "orders"})

        sessions.append({"session": session_id, "events": events})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUT_DIR / "mock_train.jsonl"
    with jsonl_path.open("w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")

    # jsonl keeps string type names (matches OTTO's raw format); the parquet uses
    # int8 codes (matches notebooks/kaggle_otto_sample's output - see event_types.py)
    type_code = {"clicks": 0, "carts": 1, "orders": 2}
    rows = [
        {"session": s["session"], "aid": e["aid"], "ts": e["ts"], "type": type_code[e["type"]]}
        for s in sessions
        for e in s["events"]
    ]
    events_df = pl.DataFrame(rows, schema_overrides={"type": pl.Int8})
    events_df.write_parquet(OUT_DIR / "mock_events.parquet")

    categories_df = pl.DataFrame(
        {"aid": np.arange(N_ITEMS), "true_hidden_category": item_category}
    )
    categories_df.write_parquet(OUT_DIR / "mock_true_categories.parquet")

    print(f"sessions: {len(sessions)}, events: {len(events_df)}, items: {N_ITEMS}")
    print(f"type counts:\n{events_df['type'].value_counts()}")
    print(f"wrote: {jsonl_path}, mock_events.parquet, mock_true_categories.parquet")


if __name__ == "__main__":
    main()
