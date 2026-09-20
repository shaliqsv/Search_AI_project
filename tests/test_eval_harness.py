"""Step 1 of the modelling guide: the metrics match scikit-learn on toy cases; the bootstrap interval covers the true mean."""

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from ranking.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from ranking.eval.metrics import (
    DEFAULT_GAINS,
    OTTO_WEIGHTS,
    ndcg_at_k,
    otto_weighted_recall_session,
    recall_at_k,
)


def _case(rng):
    n_items = 30
    grades = np.zeros(n_items)
    for gain in DEFAULT_GAINS.values():
        idx = rng.choice(n_items, size=rng.integers(0, 4), replace=False)
        grades[idx] = np.maximum(grades[idx], gain)
    scores = rng.random(n_items)
    ranking = list(np.argsort(-scores))
    labels = {"click": set(np.flatnonzero(grades == 1)), "cart": set(np.flatnonzero(grades == 2)), "order": set(np.flatnonzero(grades == 3))}
    return grades, scores, ranking, labels


@pytest.mark.parametrize("seed", range(8))
def test_ndcg_matches_scikit_learn(seed):
    grades, scores, ranking, labels = _case(np.random.default_rng(seed))
    if grades.sum() == 0:
        assert ndcg_at_k(ranking, labels, k=10) == 0.0
        return
    expected = ndcg_score(grades[None, :], scores[None, :], k=10)
    assert abs(ndcg_at_k([int(i) for i in ranking], {k: {int(i) for i in v} for k, v in labels.items()}, k=10) - expected) < 1e-9


def test_recall_and_weighted_recall_by_hand():
    labels = {"click": {1, 2}, "cart": {3}, "order": {4}}
    ranking = [1, 9, 3, 8, 7, 4] + list(range(100, 120))
    assert recall_at_k(ranking, {1, 2, 3, 4}, k=20) == 0.75
    weighted = otto_weighted_recall_session(ranking, labels, k=20)
    expected = OTTO_WEIGHTS["click"] * 0.5 + OTTO_WEIGHTS["cart"] * 1.0 + OTTO_WEIGHTS["order"] * 1.0
    assert abs(weighted - expected) < 1e-12


def test_bootstrap_interval_covers_the_true_mean():
    rng = np.random.default_rng(0)
    covered = 0
    for rep in range(200):
        sample = rng.beta(2, 5, size=300)                      # true mean 2/7
        _, low, high = bootstrap_ci(sample, n_boot=300, seed=rep)
        covered += low <= 2 / 7 <= high
    assert 0.90 <= covered / 200 <= 0.99


def test_paired_bootstrap_detects_a_real_difference_and_a_tie():
    rng = np.random.default_rng(1)
    a = rng.random(500)
    _, low, high = paired_bootstrap_diff(a + 0.1, a, n_boot=300)
    assert low > 0
    _, low, high = paired_bootstrap_diff(a + rng.normal(0, 0.05, 500), a, n_boot=300)
    assert low < 0 < high
