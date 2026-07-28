"""
Hybrid ensemble: stacks supervised and deep learning model probabilities
into a meta-learner for the final fraud prediction.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def build_meta_features(*model_probabilities: np.ndarray) -> np.ndarray:
    """Stack per-model fraud probabilities into a meta-feature matrix."""
    return np.column_stack(model_probabilities)


def train_meta_model(meta_features: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """A simple, interpretable meta-learner combining base model outputs."""
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(meta_features, y_true)
    return meta_model


def predict(meta_model: LogisticRegression, *model_probabilities: np.ndarray) -> np.ndarray:
    meta_features = build_meta_features(*model_probabilities)
    return meta_model.predict_proba(meta_features)[:, 1]
