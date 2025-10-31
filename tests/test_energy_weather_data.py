import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from data_collection.energy_weather_data import (
    fetch_weather_data,
    fetch_carbon_intensity,
    merge_all_sources,
)


# ---------- MOCK FIXTURES ----------
@pytest.fixture
def mock_weather_data():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-10-29", periods=3, freq="h", tz="UTC"),
            "temperature_C": [12.5, 13.1, 11.9],
            "humidity_%": [80, 78, 85],
            "wind_speed_mps": [4.2, 3.8, 5.1],
            "cloud_cover_%": [40, 60, 50],
            "solar_radiation_Wm2": [120, 80, 0],
        }
    )


@pytest.fixture
def mock_carbon_data():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-10-29", periods=3, freq="h", tz="UTC"),
            "carbon_intensity_actual": [210, 215, 205],
            "carbon_intensity_forecast": [212, 217, 208],
            "carbon_index": ["moderate", "moderate", "low"],
        }
    )


@pytest.fixture
def mock_price_data():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-10-29", periods=3, freq="h", tz="UTC"),
            "retail_price_£_per_kWh": [0.25, 0.26, 0.27],
        }
    )


@pytest.fixture
def mock_genmix_data():
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-10-29T00:00:00Z")],
            "uk_gen_biomass_%": [8.0],
            "uk_gen_gas_%": [45.0],
            "uk_gen_wind_%": [30.0],
        }
    )


@pytest.fixture
def mock_aqi_data():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-10-29", periods=3, freq="h", tz="UTC"),
            "pm10": [20, 22, 19],
            "pm2_5": [10, 11, 9],
            "co": [0.3, 0.4, 0.35],
            "no2": [15, 18, 14],
            "so2": [2, 1.8, 1.9],
            "o3": [25, 27, 24],
            "aqi_us": [42, 45, 40],
        }
    )


# ---------- TESTS ----------
def test_fetch_weather_data_success(monkeypatch):
    """Test successful API response for fetch_weather_data()"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-10-29T00:00Z", "2024-10-29T01:00Z"],
            "temperature_2m": [10, 11],
            "relative_humidity_2m": [80, 82],
            "wind_speed_10m": [5, 6],
            "cloudcover": [40, 50],
            "shortwave_radiation": [100, 120],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_weather_data()
        assert isinstance(df, pd.DataFrame)
        assert "temperature_C" in df.columns
        assert len(df) == 2


def test_fetch_carbon_intensity_api_failure(monkeypatch):
    """Test API failure handling in fetch_carbon_intensity()"""

    def mock_get(*args, **kwargs):
        raise requests.exceptions.RequestException("Mocked error")

    with patch("requests.get", mock_get):
        df = fetch_carbon_intensity()
        assert isinstance(df, pd.DataFrame)
        assert "carbon_intensity_actual" in df.columns or len(df) == 0


def test_merge_all_sources(
    mock_weather_data,
    mock_aqi_data,
    mock_carbon_data,
    mock_genmix_data,
    mock_price_data,
):
    """Test merge_all_sources() merges data correctly"""
    merged = merge_all_sources(
        mock_weather_data,
        mock_aqi_data,
        mock_carbon_data,
        mock_genmix_data,
        mock_price_data,
    )
    assert not merged.empty
    assert "temperature_C" in merged.columns
    assert "pm10" in merged.columns
    assert "carbon_intensity_actual" in merged.columns
    assert "retail_price_£_per_kWh" in merged.columns
