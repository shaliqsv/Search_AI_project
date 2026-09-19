# Feature dictionary

Columns of `data/features/*.parquet`, built by `notebooks/features.py` (issue #22). One row per (session, candidate item). Item-level values use only events before the cutoff.

| Column | Definition |
|---|---|
| `session` | Session ID (not a feature). |
| `category` | Query category (synthetic cluster ID), the same for every candidate of a query. Categorical. |
| `aid` | Candidate item ID (not a feature). |
| `label` | Graded label: 0 not found later, 1 clicked, 2 carted, 3 ordered (highest wins). Target for ranking. |
| `y_click` | 1 if the item was clicked in the held-out events, else 0. |
| `y_cart` | 1 if the item was carted in the held-out events, else 0. |
| `y_order` | 1 if the item was ordered in the held-out events, else 0. |
| `rank_covis` | Rank (1-200) of the item in the co-visitation list, 201 if that retriever did not return it. |
| `rank_pop` | Rank (1-200) of the item in the category popularity list, 201 if not returned. |
| `rrf` | Reciprocal rank fusion score of the item in the candidate pool. |
| `log_clicks` | log(1 + clicks on the item before the cutoff). |
| `log_clicks_3d` | log(1 + clicks on the item in the 3 days before the cutoff). |
| `cart_rate` | Smoothed carts per click of the item before the cutoff: (carts + 10 * global rate) / (clicks + 10). |
| `order_rate` | Smoothed orders per click of the item before the cutoff, same smoothing. |
| `pop_cat_pct` | Popularity rank of the item inside its category divided by the category size (small = popular). |
| `item_cat_size` | Number of categorised items in the item's category. |
| `item_seen` | 1 if the item had any event before the cutoff, else 0. |
| `hours_since_last_seen` | Hours between the item's last event before the cutoff and the end of the session prefix. Filled with 24 * (days before cutoff) for unseen items. |
| `covis_max` | Largest co-visitation weight between the item and any of the last 30 prefix items. |
| `covis_mean` | Mean co-visitation weight between the item and the last 30 prefix events (zeros count). |
| `covis_wsum` | Sum of co-visitation weights, weighted by event type (click 1, cart 3, order 6) and recency (0.8 per step back). |
| `prefix_len` | Number of events in the session prefix. |
| `prefix_n_cart` | Number of cart events in the prefix. |
| `prefix_n_order` | Number of order events in the prefix. |
| `prefix_share_same_cat` | Share of categorised prefix items that are in the query category. |
| `category_conf` | Confidence of the query category. 1.0 for now (true category); a classifier value from #34 later. |
