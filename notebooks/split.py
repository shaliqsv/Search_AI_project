"""Time-based split of the OTTO sample (issue #6).

Import from a notebook in this folder with `from split import split_by_time, WINDOWS, ...`.
Stays here, not in `src/ranking/`, until issue #52.

Rules (they keep the future out of training)
- Train = every event with ts < train_end. A session that crosses train_end is cut there,
  so only its past is used.
- Eval sessions = sessions whose FIRST event is in [eval_start, eval_end). Their events are
  cut at eval_end. A session that started before eval_start is never an eval session, so no
  session is in both sets.
"""

import polars as pl

DAY_MS = 86_400_000
START_MS = 1_659_304_800_000  # start of day 0 (2022-07-31 22:00 UTC), the first day in the data

# Day numbers count from START_MS. Week 1 = days 0-6, week 2 = 7-13, week 3 = 14-20, week 4 = 21-27.
# Every window evaluates on the 2 days directly after its training data.
WINDOWS = {
    "w0 initial (weeks 1-2)": {"train_end_day": 14, "eval_start_day": 14, "eval_end_day": 16},
    "w1 week 3 arrives": {"train_end_day": 21, "eval_start_day": 21, "eval_end_day": 23},
    # Week 4 arrives, but its last 2 days are held back for evaluation.
    "w2 week 4 arrives": {"train_end_day": 26, "eval_start_day": 26, "eval_end_day": 28},
}


def day_to_ms(day):
    return START_MS + int(day * DAY_MS)


def split_by_time(events, train_end, eval_start, eval_end):
    """Split events (LazyFrame or DataFrame with session, ts) into (train, eval).

    Times are Unix milliseconds. Requires train_end <= eval_start < eval_end.
    """
    if not (train_end <= eval_start < eval_end):
        raise ValueError("need train_end <= eval_start < eval_end")
    lazy = events.lazy() if isinstance(events, pl.DataFrame) else events
    train = lazy.filter(pl.col("ts") < train_end)
    starts = lazy.group_by("session").agg(pl.col("ts").min().alias("first_ts"))
    eval_sessions = starts.filter(
        (pl.col("first_ts") >= eval_start) & (pl.col("first_ts") < eval_end)
    ).select("session")
    evaluation = lazy.join(eval_sessions, on="session", how="inner").filter(pl.col("ts") < eval_end)
    return train, evaluation


def split_window(events, name):
    w = WINDOWS[name]
    return split_by_time(
        events,
        day_to_ms(w["train_end_day"]),
        day_to_ms(w["eval_start_day"]),
        day_to_ms(w["eval_end_day"]),
    )
