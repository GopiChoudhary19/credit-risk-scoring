import numpy as np
import pandas as pd

from credit_risk.models import MODELS


def test_pipelines_fit_and_return_probabilities():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": pd.Categorical(rng.choice(["x", "y", "z"], 200))})
    y = (X["a"] + (X["b"] == "x") > 0.5).astype(int)
    for build in MODELS.values():
        p = build(["a"], ["b"]).fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (200,) and np.all((p >= 0) & (p <= 1))
