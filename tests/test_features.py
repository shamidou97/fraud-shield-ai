"""Unit tests for feature engineering functions."""
import pandas as pd
from src.features.engineering import haversine_distance, add_geo_distance


def test_haversine_distance_zero_for_same_point():
    assert haversine_distance(40.0, -73.0, 40.0, -73.0) == 0


def test_add_geo_distance_column_created():
    df = pd.DataFrame({
        "lat": [40.0], "long": [-73.0],
        "merch_lat": [40.1], "merch_long": [-73.1],
    })
    result = add_geo_distance(df)
    assert "geo_distance_km" in result.columns
    assert result["geo_distance_km"].iloc[0] > 0
