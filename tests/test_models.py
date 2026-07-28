"""Unit tests for supervised model training."""
import numpy as np
from src.models.supervised import train_logistic_regression


def test_train_logistic_regression_fits():
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    model = train_logistic_regression(X, y)
    preds = model.predict(X)
    assert len(preds) == 50
