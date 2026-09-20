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


# ---- Level 0 split (modelling guide, Step 1; decision D3, Provisional; see docs/decisions/D3-split-rule.md) ----
# Everything inside weeks 1-2 (days 0-13). Weeks 3 and 4 are Level 1 arrivals and modelling never touches them.
LEVEL0 = {
    "train": {"start_day": 0, "end_day": 10},              # events with ts < day 10 (sessions that cross day 10 are cut there)
    "validation": {"start_day": 10, "end_day": 12},        # sessions whose FIRST event is on days 10-11; events cut at day 12
    "test": {"start_day": 12, "end_day": 14},              # sessions whose FIRST event is on days 12-13; events cut at day 14
}
MAX_MODELLING_DAY = 14                                     # first day of week 3: never read for modelling


def level0_slice(events, name):
    """Return the events of one Level 0 slice. Training: every event before the training end. Validation and test: the whole
    session (cut at the slice end) of every session that STARTS inside the slice, so no session is in two slices."""
    s = LEVEL0[name]
    end = day_to_ms(s["end_day"])
    if s["end_day"] > MAX_MODELLING_DAY:
        raise ValueError("modelling must not read week 3 or later")
    if name == "train":
        return (events.lazy() if isinstance(events, pl.DataFrame) else events).filter(pl.col("ts") < end)
    _, evaluation = split_by_time(events, day_to_ms(s["start_day"]), day_to_ms(s["start_day"]), end)
    return evaluation


def arrival_slice(events, train_end_day, days=2):
    """The same rule for a Level 1 arrival: the training window ends at `train_end_day` and the evaluation slice is the `days`
    days right after it."""
    return split_by_time(events, day_to_ms(train_end_day), day_to_ms(train_end_day), day_to_ms(train_end_day + days))
