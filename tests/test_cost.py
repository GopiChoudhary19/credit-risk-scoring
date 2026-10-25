import numpy as np

from credit_risk.cost import COST_FN, COST_FP, bayes_threshold, best_threshold, bootstrap_cost_ci, expected_cost


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


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 400), 0, 1)
    point = expected_cost(y, p, 0.2)
    ci = bootstrap_cost_ci(y, p, 0.2, baseline_threshold=0.5, n_boot=500)
    lo, hi = ci["cost_ci95"]
    assert lo <= point <= hi
    assert ci["saving_vs_baseline_ci95"][0] > 0  # 0.2 is clearly better than 0.5 when misses cost 5x


def test_bootstrap_is_reproducible():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    assert bootstrap_cost_ci(y, p, 0.3, seed=7) == bootstrap_cost_ci(y, p, 0.3, seed=7)
