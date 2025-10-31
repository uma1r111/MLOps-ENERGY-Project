import os
import pandas as pd
import pytest
from data_selection.data_selection import update_selected_features, SELECTED_FEATURES


@pytest.fixture
def setup_temp_files(tmp_path):
    """Create temporary CSVs for testing."""
    engineered = tmp_path / "engineered_features.csv"
    selected = tmp_path / "selected_features.csv"

    # Sample engineered data
    data = {
        "datetime": pd.date_range("2024-10-30", periods=5, freq="h", tz="UTC"),
        "scaled_renewable_pct": range(5),
        "scaled_price_lag_1h": range(5),
        "scaled_price_lag_24h": range(5),
        "hour": range(5),
        "hour_sin": range(5),
        "scaled_carbon_lag_1h": range(5),
        "hour_cos": range(5),
        "scaled_log_solar_radiation_Wm2": range(5),
        "scaled_carbon_intensity_actual": range(5),
        "scaled_log_no2": range(5),
        "scaled_uk_gen_biomass_%": range(5),
        "is_peak_hour": [0, 1, 0, 1, 0],
        "scaled_wind_lag_24h": range(5),
        "scaled_carbon_per_price": range(5),
        "scaled_fossil_pct": range(5),
        "scaled_uk_gen_wind_%": range(5),
        "scaled_log_uk_gen_solar_%": range(5),
        "scaled_carbon_rolling_24h_mean": range(5),
        "scaled_uk_gen_nuclear_%": range(5),
        "scaled_uk_gen_gas_%": range(5),
        "scaled_carbon_intensity_forecast": range(5),
        "scaled_wind_speed_mps": range(5),
        "scaled_temperature_C": range(5),
        "scaled_price_rolling_24h_mean": range(5),
        "scaled_wind_lag_1h": range(5),
        "retail_price_£_per_kWh": range(5),
    }
    df = pd.DataFrame(data)
    df.to_csv(engineered, index=False)

    return engineered, selected


def test_creates_new_selected_file(monkeypatch, tmp_path, setup_temp_files):
    """Test creation of new selected_features.csv when none exists."""
    engineered, selected = setup_temp_files
    monkeypatch.setattr(
        "data_selection.data_selection.ENGINEERED_PATH", str(engineered)
    )
    monkeypatch.setattr("data_selection.data_selection.SELECTED_PATH", str(selected))
    update_selected_features()
    assert os.path.exists(selected)
    result = pd.read_csv(selected)
    assert list(result.columns) == SELECTED_FEATURES


def test_updates_existing_selected_file(monkeypatch, tmp_path, setup_temp_files):
    """Test update when new rows are added."""
    engineered, selected = setup_temp_files
    monkeypatch.setattr(
        "data_selection.data_selection.ENGINEERED_PATH", str(engineered)
    )
    monkeypatch.setattr("data_selection.data_selection.SELECTED_PATH", str(selected))

    # Create initial smaller CSV
    df = pd.read_csv(engineered).iloc[:3]
    df.to_csv(selected, index=False)

    update_selected_features()
    result = pd.read_csv(selected)
    assert len(result) >= 5  # updated file should have all 5 rows
