"""Canonical event-type encoding, matching the real OTTO sample (int8 codes).

The sampling notebook (notebooks/kaggle_otto_sample) writes `type` as int8:
0 = click, 1 = cart, 2 = order. Every module reading sampled events uses these
codes directly rather than re-encoding to strings, to match the frozen data on disk.
"""

CLICK = 0
CART = 1
ORDER = 2

# graded relevance, consistent with the NDCG gains used across the project (D2 issue #1)
GRADE = {CLICK: 1, CART: 2, ORDER: 3}
