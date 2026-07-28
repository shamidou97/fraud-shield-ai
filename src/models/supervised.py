"""
Supervised baseline models: Logistic Regression, Random Forest, XGBoost.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def train_logistic_regression(X_train, y_train, **kwargs) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", **kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, scale_pos_weight: float = 1.0, **kwargs) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
