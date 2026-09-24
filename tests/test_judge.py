"""Step 9 of the modelling guide: the judge agreement and bias statistics."""

import numpy as np

from ranking.llm.judge import position_effect, spearman, verbosity_effect, weighted_kappa


def test_kappa_is_one_for_identical_ratings_and_zero_for_independent_ones():
    a = np.array([0, 1, 2, 3, 4] * 20)
    assert weighted_kappa(a, a) == 1.0
    rng = np.random.default_rng(0)
    assert abs(weighted_kappa(rng.integers(0, 5, 2000), rng.integers(0, 5, 2000))) < 0.08


def test_kappa_penalises_far_disagreement_more_than_near_disagreement():
    a = np.array([0, 1, 2, 3, 4] * 20)
    assert weighted_kappa(a, np.clip(a + 1, 0, 4)) > weighted_kappa(a, 4 - a)


def test_spearman_and_bias_checks():
    x = np.arange(50)
    assert abs(spearman(x, x * 2) - 1.0) < 1e-12 and abs(spearman(x, -x) + 1.0) < 1e-12
    means, spread = position_effect([1, 1, 3, 3], [0, 0, 1, 1])
    assert means == {0: 1.0, 1: 3.0} and spread == 2.0
    assert abs(verbosity_effect(x, x) - 1.0) < 1e-12
