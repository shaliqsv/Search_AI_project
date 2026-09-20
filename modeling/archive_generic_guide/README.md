# Archived: exploratory work under the earlier, generic modelling guide

This folder holds what was done before `_doc/modeling_guide.md` was rewritten around the project plan. It is kept as exploratory
evidence, not as the phase deliverable. It ran a generic tabular slate on the candidate table `data/features/train_asof11.parquet`
(dev queries, features as of day 11) and found that no learned model beats the co-visitation rank by more than noise (best tuned
extra trees +0.0007 NDCG@10 over it). That finding is consistent with the first-phase comparison and is worth checking again in the
plan-based steps. Its "fresh holdout" (sessions starting on days 16-17) touched week 3, which the new guide reserves for Level 1
arrivals, so it was deleted unscored. Nothing here is a decision of the current phase.
