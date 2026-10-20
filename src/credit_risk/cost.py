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
