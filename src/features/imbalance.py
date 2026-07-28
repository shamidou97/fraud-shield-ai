"""
Class imbalance handling for the highly skewed is_fraud label.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X, y)


def apply_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X, y)


def compute_class_weights(y: pd.Series) -> dict:
    """Inverse-frequency class weights, for cost-sensitive training (e.g. XGBoost scale_pos_weight)."""
    counts = y.value_counts()
    total = len(y)
    return {cls: total / (len(counts) * count) for cls, count in counts.items()}
