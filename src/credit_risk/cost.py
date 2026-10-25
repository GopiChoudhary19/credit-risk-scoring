import numpy as np

# Statlog cost matrix: approving a bad applicant costs 5, rejecting a good one costs 1.
COST_FN = 5.0
COST_FP = 1.0


def expected_cost(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    pred = (proba >= threshold).astype(int)
    fn = np.sum((pred == 0) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    return float((COST_FN * fn + COST_FP * fp) / len(y_true))


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91)
    costs = [expected_cost(y_true, proba, t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def bayes_threshold() -> float:
    """Decision-theoretic threshold for a calibrated model: reject when p(bad) > c_fp / (c_fp + c_fn)."""
    return COST_FP / (COST_FP + COST_FN)


def bootstrap_cost_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                      baseline_threshold: float | None = None, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """95% bootstrap interval for the cost per applicant, and optionally for the saving over a second threshold.

    Both thresholds are scored on the same resampled applicants, so the saving interval is paired.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    costs, savings = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        costs[b] = expected_cost(y_true[ix], proba[ix], threshold)
        if baseline_threshold is not None:
            savings[b] = expected_cost(y_true[ix], proba[ix], baseline_threshold) - costs[b]
    out = {"cost_ci95": [round(float(v), 4) for v in np.percentile(costs, [2.5, 97.5])]}
    if baseline_threshold is not None:
        out["saving_vs_baseline_ci95"] = [round(float(v), 4) for v in np.percentile(savings, [2.5, 97.5])]
    return out
