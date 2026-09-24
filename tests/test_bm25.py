"""Step 3 of the modelling guide: the hand-written BM25 equals the rank_bm25 library on toy cases."""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from ranking.models.bm25 import BM25, tokenize

CORPUS = [
    "summer dress light dress", "winter coat wool", "wool sweater winter", "light summer sandals", "leather boots winter",
    "cotton shirt summer", "dress shoes leather", "wool scarf", "sandals beach", "coat rain light",
]
QUERIES = ["summer dress", "winter wool", "leather", "sandals beach summer", "unknown word", "light coat rain"]


@pytest.mark.parametrize("query", QUERIES)
def test_scores_equal_the_library(query):
    docs = [tokenize(d) for d in CORPUS]
    ours, lib = BM25(docs, k1=1.5, b=0.75, epsilon=0.25), BM25Okapi(docs, k1=1.5, b=0.75, epsilon=0.25)
    assert np.allclose(ours.scores(tokenize(query)), lib.get_scores(tokenize(query)), atol=1e-9)


def test_other_parameters_also_match():
    docs = [tokenize(d) for d in CORPUS]
    ours, lib = BM25(docs, k1=0.9, b=0.4, epsilon=0.25), BM25Okapi(docs, k1=0.9, b=0.4, epsilon=0.25)
    assert np.allclose(ours.scores(["summer", "wool"]), lib.get_scores(["summer", "wool"]), atol=1e-9)
