import joblib
import numpy as np
import pandas as pd
import pytest

from credit_risk.models import MODELS
from credit_risk.predict import score


@pytest.fixture
def model_path(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": pd.Categorical(rng.choice(["x", "y"], 200))})
    y = (X["a"] > 0).astype(int)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": MODELS["logistic_regression"](["a"], ["b"]).fit(X, y), "threshold": 0.3,
                 "columns": ["a", "b"]}, path)
    return str(path)


def test_score_adds_probability_and_decision(model_path):
    out = score(pd.DataFrame({"a": [-3.0, 3.0], "b": ["x", "y"]}), model_path)
    assert list(out["decision"]) == ["approve", "reject"]
    assert out["p_bad"].between(0, 1).all()


def test_score_names_missing_columns(model_path):
    with pytest.raises(ValueError, match="missing 1 required column"):
        score(pd.DataFrame({"a": [0.0]}), model_path)
