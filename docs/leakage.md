# Leakage audit

Checked by `notebooks/12_leakage_audit.ipynb` on window w0 (train days 0-13, eval days 14-15). Re-run the notebook after every task listed at its end.

| Place | Status | Evidence |
|---|---|---|
| Split boundaries | checked automatically | Last train event is before the first eval event. Eval events stay inside the eval window. 0 sessions are in both sets |
| Co-visitation matrix | checked automatically | Item set equals the items with 5 or more events before the train end (396,630 items). 41,281 items would differ if the eval days were counted, so the check can tell |
| Categories | checked automatically | All 394,663 categorised items are inside the co-visitation item set, which is training window only |
| Query construction: time order | checked automatically | All query sessions start in the eval window. Prefix ends before the labels. 300 random queries match the raw events |
| Query construction: category from the label | **known leak (accepted)** | The query category is the category of the held-out click. Ranking inside it is easier than real search. #36 measures the effect of predicted categories |
| Sessions crossing the boundary | checked automatically | 525,031 sessions cross the train end. Their later events are dropped from train and none of them is an eval session |
| Session sampling | not applicable | Sessions are chosen by a hash of the session ID only, never by events or labels |
| Items never seen in training | not a leak, a coverage limit | 3.9% of label events are on items with no training events. 11.8% are on items without a category. Models cannot retrieve them |
| Popularity and rate features | pending | Call `check_counts_from_window` on the counts each ranker or feature uses. Re-run after #17 and #22 |

## The deliberate-leak demonstration
`check_counts_from_window(counts, events, train_end, name)` recomputes counts from events before the train end and compares. The notebook passes it counts over all four weeks and shows that it fails, so the check is real.

## Where to re-run
After #17 (popularity), #18 (co-visitation scoring), #22 (features as of a cutoff), #23 to #28 (any training set built from sessions) and #36 (predicted categories).
