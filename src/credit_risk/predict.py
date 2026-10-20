"""Score new applicants from a CSV: python -m credit_risk.predict applicants.csv"""
import sys

import joblib
import pandas as pd


def score(df: pd.DataFrame, path: str = "artifacts/credit_model.joblib") -> pd.DataFrame:
    bundle = joblib.load(path)
    p = bundle["model"].predict_proba(df[bundle["columns"]])[:, 1]
    decision = ["reject" if v >= bundle["threshold"] else "approve" for v in p]
    return df.assign(p_bad=p.round(4), decision=decision)


if __name__ == "__main__":
    print(score(pd.read_csv(sys.argv[1])).to_string(index=False))
