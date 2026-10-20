import pandas as pd
from sklearn.datasets import fetch_openml


def load_credit(cache_dir: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Statlog German Credit data (1000 applicants, 20 features).

    Returns features and a binary target where 1 = bad credit (default risk).
    """
    bunch = fetch_openml("credit-g", version=1, as_frame=True, data_home=cache_dir)
    X = bunch.data.copy()
    y = (bunch.target == "bad").astype(int).rename("bad")
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if X[c].dtype.name in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    return num, cat
