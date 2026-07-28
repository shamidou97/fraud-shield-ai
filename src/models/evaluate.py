"""
Evaluation metrics and interpretability reporting.
"""
import shap
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
    }


def print_report(y_true, y_pred) -> None:
    print(classification_report(y_true, y_pred, target_names=["legit", "fraud"]))


def explain_model(model, X_sample):
    """Returns SHAP values for interpretability reporting."""
    explainer = shap.Explainer(model, X_sample)
    return explainer(X_sample)
