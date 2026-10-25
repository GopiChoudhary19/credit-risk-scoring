"""Cross-validated model comparison, cost-aware thresholding and a held-out test report."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .cost import bayes_threshold, best_threshold, bootstrap_cost_ci, expected_cost
from .data import load_credit, split_columns
from .models import MODELS, SEED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def metrics(y, p, thr):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "cost_per_applicant": round(expected_cost(y, p, thr), 4),
        "threshold": round(thr, 3),
    }


def main(out_dir: str = "reports", model_dir: str = "artifacts") -> dict:
    out, mdir = Path(out_dir), Path(model_dir)
    out.mkdir(exist_ok=True)
    mdir.mkdir(exist_ok=True)
    X, y = load_credit()
    num, cat = split_columns(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, oof = {}, {}
    for name, build in MODELS.items():
        model = CalibratedClassifierCV(build(num, cat), method="sigmoid", cv=3)
        p = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr, _ = best_threshold(y_tr.values, p)
        oof[name] = p
        results[name] = metrics(y_tr.values, p, thr)
        print(name, results[name])

    # model and threshold are both chosen on out-of-fold predictions; the test split is touched once
    best = min(results, key=lambda k: results[k]["cost_per_applicant"])
    thr = results[best]["threshold"]
    final = CalibratedClassifierCV(MODELS[best](num, cat), method="sigmoid", cv=5).fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    report = {
        "selected_model": best,
        "cv": results,
        "test": metrics(y_te.values, p_te, thr),
        "test_bayes_threshold": metrics(y_te.values, p_te, bayes_threshold()),
        "test_naive_0.5": metrics(y_te.values, p_te, 0.5),
        "baseline_approve_everyone_cost": round(expected_cost(y_te.values, np.zeros(len(y_te)), 0.5), 4),
    }
    # 200 test applicants is a small sample, so report how much the headline cost could move
    report["test"].update(bootstrap_cost_ci(y_te.values, p_te, thr, baseline_threshold=0.5, seed=SEED))
    joblib.dump({"model": final, "threshold": thr, "columns": list(X.columns)}, mdir / "credit_model.joblib")

    imp = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1][:10]
    report["top_features"] = {X.columns[i]: round(float(imp.importances_mean[i]), 4) for i in order}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y_tr, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_tr, p):.3f})")
        frac, mean = calibration_curve(y_tr, p, n_bins=8)
        ax[1].plot(mean, frac, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (5-fold out-of-fold)", xlabel="FPR", ylabel="TPR")
    ax[0].legend()
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set(title="Calibration (out-of-fold)", xlabel="predicted p(bad)", ylabel="observed rate")
    ax[1].legend()
    grid = np.linspace(0.05, 0.95, 91)
    ax[2].plot(grid, [expected_cost(y_tr.values, oof[best], t) for t in grid])
    ax[2].axvline(thr, color="r", ls="--", label=f"chosen {thr:.2f}")
    ax[2].axvline(0.5, color="gray", ls=":", label="naive 0.5")
    ax[2].set(title=f"Expected cost vs threshold ({best})", xlabel="threshold", ylabel="cost / applicant")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=120)

    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--model-dir", default="artifacts")
    a = ap.parse_args()
    main(a.out_dir, a.model_dir)
