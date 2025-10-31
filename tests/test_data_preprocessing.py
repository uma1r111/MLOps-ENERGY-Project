import pandas as pd
from data_pre_processing.data_preprocessing import (
    create_temporal_features,
    create_interaction_features,
    apply_log_transforms,
    scale_features,
)


def test_temporal_features():
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=3, freq="h")})
    result = create_temporal_features(df)
    assert "hour" in result.columns
    assert "hour_sin" in result.columns


def test_interaction_features():
    df = pd.DataFrame(
        {
            "uk_gen_wind_%": [20, 30, 40],
            "uk_gen_solar_%": [10, 15, 5],
            "uk_gen_gas_%": [50, 45, 55],
            "temperature_C": [15, 25, 10],
            "solar_radiation_Wm2": [100, 200, 50],
            "carbon_intensity_actual": [200, 250, 180],
            "retail_price_£_per_kWh": [0.15, 0.18, 0.2],
        }
    )
    result = create_interaction_features(df)
    assert "renewable_pct" in result.columns
    assert "carbon_per_price" in result.columns


def test_log_transformations():
    df = pd.DataFrame({"so2": [1, 10, 100]})
    result = apply_log_transforms(df)
    assert "log_so2" in result.columns


def test_scaling_features():
    df = pd.DataFrame(
        {
            "temperature_C": [10, 15, 20],
            "wind_speed_mps": [5, 10, 15],
            "humidity_%": [30, 40, 50],
            "carbon_intensity_actual": [200, 250, 300],
            "uk_gen_wind_%": [20, 30, 40],
            "uk_gen_gas_%": [50, 60, 70],
        }
    )
    result = scale_features(df)
    assert any(col.startswith("scaled_") for col in result.columns)
