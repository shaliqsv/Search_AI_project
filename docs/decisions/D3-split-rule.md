# D3. Level 0 time split (Provisional)

Status: Provisional (proposed in modelling Step 1, issue #123). Plan D3 fixed only the layout: weeks 1-2 as the initial training window, weeks 3 and 4 as Level 1 arrivals, evaluation on the days right after a training window.

## Decision
Everything for Level 0 lives inside weeks 1-2 (days 0-13); modelling never reads week 3 or later.

| Slice | Days | What is in it |
|---|---|---|
| Training | 0-9 | every event with `ts` before day 10; a session that crosses day 10 is cut there, so only its past is used |
| Validation | 10-11 | every session whose FIRST event is on days 10-11, with its events cut at the end of day 11 |
| Test | 12-13 | every session whose FIRST event is on days 12-13, with its events cut at the end of day 13; scored once, in Step 11 |

A session belongs to at most one slice, decided by its first event. The same function applies to a Level 1 arrival: the training window ends at `train_end_day` and the evaluation slice is the two days right after it (`ranking.data.split.arrival_slice`).

Categories, the co-visitation graph, embeddings, popularity and every other learned quantity use the training window (days 0-9) only (D4). Item-level features for a validation or test session are computed point in time, as of the day before the session, from past events only (D25).

## Why
Time-ordered slices mimic deployment and stop future behaviour from leaking into features or categories. Cutting crossing sessions at the boundary keeps the future out of training. Assigning a session by its first event keeps slices session-disjoint. Measured on the sample (weeks 1-2, 13,064,647 events): training 9,060,280 events (69%), validation 661,000 events in 107,467 sessions (8,036 with an order), test 833,544 events in 127,458 sessions (9,526 with an order). Even the rarest head (orders) has thousands of evaluation sessions, enough for stable bootstrap intervals, while weeks 3 and 4 stay free for Level 1.

## Alternatives considered
- Training on days 0-11 and testing on days 12-13 (test right after training, but no validation slice left for tuning).
- A random session split (leaks the future).
- Weeks 1-2 for training and days 14-15 for evaluation, as in the first phase (touches week 3, which is reserved for Level 1).
- Refitting on training plus validation before the test (would make the test score right after training but breaks "categories from the training window only"); the test score is therefore slightly pessimistic for a model retrained through day 11.

## Consequences
The first-phase results (issues #4-#33) used days 14-15 and one shared feature table and are history, not results of this phase.
