"""
Feature engineering: transaction velocity, merchant/cardholder geo-distance,
and per-cardholder spending profile features.
"""
import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between cardholder and merchant coordinates."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["geo_distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    return df


def add_transaction_velocity(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    """Rolling count of transactions per card number within a time window."""
    df = df.sort_values(["cc_num", "datetime"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df["txn_velocity"] = (
        df.groupby("cc_num")["amt"]
        .rolling(window)
        .count()
        .reset_index(level=0, drop=True)
    )
    return df.reset_index()


def add_spending_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cardholder historical mean/std of transaction amount, for anomaly scoring."""
    df = df.copy()
    profile = df.groupby("cc_num")["amt"].agg(["mean", "std"]).rename(
        columns={"mean": "cc_avg_amt", "std": "cc_std_amt"}
    )
    df = df.merge(profile, on="cc_num", how="left")
    df["amt_zscore"] = (df["amt"] - df["cc_avg_amt"]) / df["cc_std_amt"].replace(0, np.nan)
    return df
