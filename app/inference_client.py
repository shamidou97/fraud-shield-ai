"""
Local inference client for the Fraud Shield hybrid ensemble.

Loads the five trained base models (Logistic Regression, Random Forest,
XGBoost, FNN, LSTM) plus the feature scaler and meta-model saved by
`03_baseline_models.ipynb`, `04_deep_learning_models.ipynb`, and
`05_hybrid_ensemble.ipynb`, and runs the same ensemble logic used there
against new transaction data.

This talks to the models directly on disk rather than a deployed SageMaker
endpoint. Once `src/pipeline/deploy.py` has a real endpoint running, swap
`predict_batch` to call `invoke_endpoint` instead -- the function signature
(DataFrame in, DataFrame with `is_fraud_pred`/`fraud_probability` out) can
stay the same so `streamlit_app.py` doesn't need to change.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.deep_learning import FraudFNN, FraudLSTM
from src.models.ensemble import build_meta_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "models"
SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature columns expected in incoming data, in the order the models were
# trained on. Must match `feature_cols` in 02_feature_engineering.ipynb /
# 03_baseline_models.ipynb exactly.
ID_COLS = ["transaction_id", "event_time", "cc_num"]
FEATURE_COLS = [
    "amt", "city_pop", "geo_distance_km", "txn_velocity", "amt_zscore",
    "amt_category_zscore",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net",
    "category_misc_pos", "category_personal_care", "category_shopping_net",
    "category_shopping_pos", "category_travel", "gender_F", "gender_M",
]

_models = None  # lazy-loaded cache, see _load_models()


def _load_models():
    """Load all artifacts once and cache them for the life of the process."""
    global _models
    if _models is not None:
        return _models

    logreg = joblib.load(MODELS_DIR / "logreg.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
    meta_model = joblib.load(MODELS_DIR / "meta_model.joblib")
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")

    fnn = FraudFNN(input_dim=len(FEATURE_COLS)).to(DEVICE)
    fnn.load_state_dict(torch.load(MODELS_DIR / "fnn.pt", map_location=DEVICE))
    fnn.eval()

    lstm = FraudLSTM(input_dim=len(FEATURE_COLS)).to(DEVICE)
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm.pt", map_location=DEVICE))
    lstm.eval()

    _models = {
        "logreg": logreg,
        "rf": rf,
        "xgb": xgb_model,
        "fnn": fnn,
        "lstm": lstm,
        "meta_model": meta_model,
        "scaler": scaler,
    }
    return _models


def _build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = SEQ_LEN):
    """
    Same trailing-window sequence construction used in 04/05: for each
    transaction, build a (seq_len, n_features) array of that cardholder's
    most recent transactions (zero-padded if fewer than seq_len exist),
    using only cc_num + event_time already present in the input.
    """
    sorted_df = df.sort_values(["cc_num", "event_time"]).reset_index()
    sorted_df = sorted_df.rename(columns={"index": "orig_index"})

    lag_arrays = [sorted_df[feature_cols].values]
    for lag in range(1, seq_len):
        shifted = sorted_df.groupby("cc_num")[feature_cols].shift(lag).fillna(0).values
        lag_arrays.append(shifted)
    lag_arrays = lag_arrays[::-1]

    sequences = np.stack(lag_arrays, axis=1)
    pos_lookup = pd.Series(np.arange(len(sorted_df)), index=sorted_df["orig_index"])
    return sequences, pos_lookup


def predict_batch(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Run the full hybrid ensemble against a batch of transactions.

    `df` must contain FEATURE_COLS plus `cc_num` and `event_time` (needed
    for the LSTM's sequence construction). Extra columns are ignored.

    Returns the original DataFrame with two columns appended:
    `fraud_probability` (the ensemble's raw probability) and
    `is_fraud_pred` (0/1, using `threshold`).
    """
    models = _load_models()
    missing = [c for c in FEATURE_COLS + ["cc_num", "event_time"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Supervised branch
    logreg_proba = models["logreg"].predict_proba(X)[:, 1]
    rf_proba = models["rf"].predict_proba(X)[:, 1]
    xgb_proba = models["xgb"].predict_proba(X)[:, 1]

    # Deep learning branch
    scaler = models["scaler"]
    X_scaled = scaler.transform(X.values)

    sequences, pos_lookup = _build_sequences(df, FEATURE_COLS, SEQ_LEN)
    positions = pos_lookup.loc[df.index].values
    X_seq_raw = sequences[positions]
    n, seq_len, n_features = X_seq_raw.shape
    X_seq_scaled = scaler.transform(X_seq_raw.reshape(-1, n_features)).reshape(n, seq_len, n_features)

    with torch.no_grad():
        fnn_proba = models["fnn"](
            torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
        lstm_proba = models["lstm"](
            torch.tensor(X_seq_scaled, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    # Hybrid ensemble
    meta_features = build_meta_features(logreg_proba, rf_proba, xgb_proba, fnn_proba, lstm_proba)
    ensemble_proba = models["meta_model"].predict_proba(meta_features)[:, 1]

    result = df.copy()
    result["fraud_probability"] = ensemble_proba
    result["is_fraud_pred"] = (ensemble_proba >= threshold).astype(int)
    return result
