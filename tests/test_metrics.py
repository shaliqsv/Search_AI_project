import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from ranking.eval.bootstrap import bootstrap_ci, paired_bootstrap_pvalue
from ranking.eval.metrics import ndcg_at_k, otto_weighted_score, recall_at_20_micro


@pytest.mark.parametrize(
    "gains",
    [
        [3, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 3],
        [2, 3, 1, 0, 0],
    ],
)
def test_ndcg_matches_sklearn(gains):
    got = ndcg_at_k(gains, k=10)
    want = ndcg_score(np.asarray([gains]), np.asarray([gains]), k=10) if any(gains) else 0.0
    # sklearn's ndcg_score expects true relevance and predicted scores separately;
    # since `gains` here is already in predicted-rank order, feed it as both
    # y_true (ideal ordering is computed internally) and a monotonically
    # decreasing y_score so predicted order matches input order.
    y_score = list(range(len(gains), 0, -1))
    want = ndcg_score(np.asarray([gains]), np.asarray([y_score]), k=10)
    assert got == pytest.approx(want, abs=1e-9)


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k([3, 2, 1], k=10) == pytest.approx(1.0)


def test_ndcg_worst_order_is_low():
    assert ndcg_at_k([0, 1, 3], k=10) < ndcg_at_k([3, 1, 0], k=10)


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], k=10) == 0.0


def test_ndcg_truncates_at_k():
    # a relevant item beyond k contributes nothing
    within_k = ndcg_at_k([0, 0, 0, 0, 0, 5], k=5)
    assert within_k == 0.0


def test_recall_at_20_perfect_and_partial():
    assert recall_at_20_micro([[1, 2, 3]], [[1, 2, 3]]) == 1.0
    # 2 of 3 ground-truth items recovered
    assert recall_at_20_micro([[1, 2]], [[1, 2, 3]]) == pytest.approx(2 / 3)


def test_recall_at_20_is_micro_averaged_not_a_mean():
    # session A: perfect (1/1). session B: total miss (0/2).
    # a naive mean of per-session recalls would be 0.5; micro-averaging gives 1/3.
    got = recall_at_20_micro([[1], []], [[1], [1, 2]])
    assert got == pytest.approx(1 / 3)
    assert got != pytest.approx(0.5)


def test_recall_at_20_truncates_predictions_at_20():
    pred = list(range(25))
    assert recall_at_20_micro([pred], [[19]]) == 1.0  # index 19 -> within top 20
    assert recall_at_20_micro([pred], [[24]]) == 0.0  # index 24 -> beyond top 20


def test_recall_at_20_more_than_20_ground_truth_caps_denominator():
    # official spec: predicting 20 correctly out of >20 ground truth scores 1.0
    assert recall_at_20_micro([list(range(20))], [list(range(30))]) == 1.0


def test_recall_at_20_empty_ground_truth_is_zero_not_error():
    assert recall_at_20_micro([[1, 2]], [[]]) == 0.0


def test_otto_weighted_score_matches_official_weights():
    assert otto_weighted_score(1.0, 0.0, 0.0) == pytest.approx(0.10)
    assert otto_weighted_score(0.0, 1.0, 0.0) == pytest.approx(0.30)
    assert otto_weighted_score(0.0, 0.0, 1.0) == pytest.approx(0.60)
    assert otto_weighted_score(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_bootstrap_ci_recovers_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.05, size=500)
    mean, lower, upper = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert lower < mean < upper
    assert mean == pytest.approx(0.5, abs=0.02)


def test_paired_bootstrap_detects_clear_winner():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.6, scale=0.05, size=300)
    b = rng.normal(loc=0.4, scale=0.05, size=300)
    diff, p_value = paired_bootstrap_pvalue(a, b, n_resamples=2000, seed=1)
    assert diff > 0
    assert p_value < 0.05


def test_paired_bootstrap_no_difference_is_not_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.5, scale=0.05, size=300)
    b = a.copy()
    diff, p_value = paired_bootstrap_pvalue(a, b, n_resamples=2000, seed=1)
    assert diff == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_pvalue([1, 2, 3], [1, 2])
