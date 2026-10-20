import numpy as np

from credit_risk.cost import COST_FN, COST_FP, bayes_threshold, best_threshold, expected_cost


def test_expected_cost_counts_errors():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])  # one FN and one FP at t=0.5
    assert expected_cost(y, p, 0.5) == (COST_FN + COST_FP) / 4


def test_bayes_threshold():
    assert abs(bayes_threshold() - 1 / 6) < 1e-9


def test_best_threshold_beats_naive_when_fn_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)
    thr, cost = best_threshold(y, p)
    assert thr < 0.5 and cost <= expected_cost(y, p, 0.5)
