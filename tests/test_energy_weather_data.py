import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
from data_collection.energy_weather_data import (
    fetch_weather_data,
    fetch_air_quality,
    fetch_carbon_intensity,
    fetch_carbon_generation_mix,
    fetch_octopus_prices,
    merge_all_sources,
)

# ---------- FIXTURES ----------


@pytest.fixture
def yesterday_iso():
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    return yesterday.isoformat() + "T00:00Z"


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
def mock_price_data():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-10-29", periods=3, freq="h", tz="UTC"),
            "retail_price_£_per_kWh": [0.25, 0.26, 0.27],
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


# ---------- FETCH CARBON INTENSITY TESTS ----------


def test_fetch_carbon_intensity_missing_intensity(yesterday_iso):
    """Test handling of None or missing intensity safely."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"from": yesterday_iso, "intensity": None},
            {"from": yesterday_iso},
            {
                "from": yesterday_iso,
                "intensity": {"actual": 120, "forecast": 130, "index": "moderate"},
            },
        ]
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_carbon_intensity()

    valid_rows = df[df["carbon_intensity_actual"].notnull()]
    assert len(valid_rows) >= 1

    missing_rows = df[df["carbon_intensity_actual"].isnull()]
    for col in ["carbon_intensity_actual", "carbon_intensity_forecast", "carbon_index"]:
        assert missing_rows[col].isnull().all()


def test_fetch_carbon_intensity_empty_response():
    """Test handling of empty API response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_carbon_intensity()
        assert df.empty
        assert list(df.columns) == [
            "datetime",
            "carbon_intensity_actual",
            "carbon_intensity_forecast",
            "carbon_index",
        ]


def test_fetch_carbon_intensity_api_failure():
    """Test handling of API failure."""
    with patch(
        "requests.get", side_effect=requests.exceptions.RequestException("API Error")
    ):
        df = fetch_carbon_intensity()
        assert df.empty


# ---------- FETCH WEATHER DATA TESTS ----------


def test_fetch_weather_data_column_types():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-10-29T00:00Z"],
            "temperature_2m": [10.5],
            "relative_humidity_2m": [75],
            "wind_speed_10m": [3.2],
            "cloudcover": [50],
            "shortwave_radiation": [100],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_weather_data()
        assert df["temperature_C"].dtype.kind in ("f", "i")
        assert df["humidity_%"].dtype.kind in ("f", "i")
        assert df["wind_speed_mps"].dtype.kind in ("f", "i")
        assert df["cloud_cover_%"].dtype.kind in ("f", "i")
        assert df["solar_radiation_Wm2"].dtype.kind in ("f", "i")


def test_fetch_weather_data_api_exception():
    """Test fetch_weather_data handles API exception safely."""

    def raise_exc(*args, **kwargs):
        raise requests.exceptions.Timeout("Timeout!")

    with patch("requests.get", side_effect=raise_exc):
        with pytest.raises(requests.exceptions.Timeout):
            fetch_weather_data()


def test_fetch_weather_data_custom_location():
    """Test fetch_weather_data with custom coordinates."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-10-29T00:00Z"],
            "temperature_2m": [15.0],
            "relative_humidity_2m": [70],
            "wind_speed_10m": [5.0],
            "cloudcover": [30],
            "shortwave_radiation": [200],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response) as mock_get:
        df = fetch_weather_data(lat=40.7128, lon=-74.0060)
        assert not df.empty
        call_url = mock_get.call_args[0][0]
        assert "latitude=40.7128" in call_url
        # Longitude might be formatted as -74.006 (trailing zero removed)
        assert "longitude=-74.006" in call_url


# ---------- FETCH AIR QUALITY TEST ----------


def test_fetch_air_quality():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-10-29T00:00Z"],
            "pm10": [20],
            "pm2_5": [10],
            "carbon_monoxide": [0.3],
            "nitrogen_dioxide": [15],
            "sulphur_dioxide": [2],
            "ozone": [25],
            "us_aqi": [42],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_air_quality()
        assert df["pm10"].iloc[0] == 20
        assert df["co"].iloc[0] == 0.3


def test_fetch_air_quality_custom_location():
    """Test fetch_air_quality with custom coordinates."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2024-10-29T00:00Z"],
            "pm10": [25],
            "pm2_5": [12],
            "carbon_monoxide": [0.5],
            "nitrogen_dioxide": [20],
            "sulphur_dioxide": [3],
            "ozone": [30],
            "us_aqi": [50],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response) as mock_get:
        df = fetch_air_quality(lat=40.7128, lon=-74.0060)
        assert not df.empty
        call_url = mock_get.call_args[0][0]
        assert "latitude=40.7128" in call_url
        # Longitude might be formatted as -74.006 (trailing zero removed)
        assert "longitude=-74.006" in call_url


# ---------- FETCH GENERATION MIX ----------


def test_fetch_carbon_generation_mix():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "from": "2024-10-29T00:00Z",
            "generationmix": [
                {"fuel": "Biomass", "perc": 8},
                {"fuel": "Gas", "perc": 45},
                {"fuel": "Wind", "perc": 30},
                {"fuel": "Solar", "perc": 10},
            ],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_carbon_generation_mix()
        for col in [
            "uk_gen_biomass_%",
            "uk_gen_gas_%",
            "uk_gen_wind_%",
            "uk_gen_solar_%",
        ]:
            assert col in df.columns


def test_fetch_carbon_generation_mix_all_fuels():
    """Test that all expected fuel types are captured."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "from": "2024-10-29T00:00Z",
            "generationmix": [
                {"fuel": "Biomass", "perc": 8},
                {"fuel": "Coal", "perc": 5},  # Should be filtered
                {"fuel": "Imports", "perc": 10},
                {"fuel": "Gas", "perc": 45},
                {"fuel": "Nuclear", "perc": 20},
                {"fuel": "Other", "perc": 2},  # Should be filtered
                {"fuel": "Solar", "perc": 5},
                {"fuel": "Wind", "perc": 15},
            ],
        }
    }
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        df = fetch_carbon_generation_mix()
        # Check that only specified fuels are included
        expected_cols = [
            "datetime",
            "uk_gen_biomass_%",
            "uk_gen_imports_%",
            "uk_gen_gas_%",
            "uk_gen_nuclear_%",
            "uk_gen_solar_%",
            "uk_gen_wind_%",
        ]
        assert set(expected_cols).issubset(set(df.columns))
        # Check that filtered fuels are not included
        assert "uk_gen_coal_%" not in df.columns
        assert "uk_gen_other_%" not in df.columns


# ---------- FETCH OCTOPUS PRICES ----------


def test_fetch_octopus_prices():
    """Test fetching Octopus Energy prices with proper mocking."""
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    product_data = {
        "results": [
            {
                "code": "AGILE-18-02-21",
                "links": [
                    {
                        "href": "https://api.octopus.energy/v1/electricity-tariffs/E-1R-AGILE-18-02-21-A/"
                    }
                ],
            }
        ]
    }

    # Create rates data with yesterday's timestamp
    rates_data = {
        "results": [
            {"valid_from": f"{yesterday}T00:00:00Z", "value_inc_vat": 25.0},
            {"valid_from": f"{yesterday}T01:00:00Z", "value_inc_vat": 26.0},
            {"valid_from": f"{yesterday}T02:00:00Z", "value_inc_vat": 27.0},
        ]
    }

    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        if "products/" in url and "electricity-tariffs" not in url:
            mock_resp.json.return_value = product_data
        elif "standard-unit-rates" in url:
            mock_resp.json.return_value = rates_data
        else:
            raise ValueError(f"Unexpected URL: {url}")
        return mock_resp

    with patch("requests.get", side_effect=mock_get):
        df = fetch_octopus_prices()
        assert not df.empty
        assert "datetime" in df.columns
        assert "retail_price_£_per_kWh" in df.columns
        assert df["retail_price_£_per_kWh"].iloc[0] == 0.25
        # All returned data should be from yesterday
        assert all(df["datetime"].dt.date == yesterday)


def test_fetch_octopus_prices_no_agile_products():
    """Test handling when no Agile products are found."""
    product_data = {"results": [{"code": "FIXED-12-MONTH", "links": []}]}

    mock_response = MagicMock()
    mock_response.json.return_value = product_data
    mock_response.raise_for_status = lambda: None

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="No Agile tariffs found"):
            fetch_octopus_prices()


# ---------- MERGE ALL SOURCES ----------


def test_merge_all_sources_complete(
    mock_weather_data,
    mock_aqi_data,
    mock_carbon_data,
    mock_genmix_data,
    mock_price_data,
):
    merged = merge_all_sources(
        mock_weather_data,
        mock_aqi_data,
        mock_carbon_data,
        mock_genmix_data,
        mock_price_data,
    )
    assert not merged.empty
    for col in [
        "temperature_C",
        "pm10",
        "carbon_intensity_actual",
        "retail_price_£_per_kWh",
    ]:
        assert col in merged.columns


def test_merge_all_sources_misalignment(
    mock_weather_data,
    mock_aqi_data,
    mock_carbon_data,
    mock_genmix_data,
    mock_price_data,
):
    shifted_aqi = mock_aqi_data.copy()
    shifted_aqi["datetime"] += pd.Timedelta(hours=1)
    merged = merge_all_sources(
        mock_weather_data,
        shifted_aqi,
        mock_carbon_data,
        mock_genmix_data,
        mock_price_data,
    )
    assert merged.isnull().any().any()


# ---------- HELPER FUNCTIONS ----------


def test_get_yesterday_dates():
    """Test get_yesterday_dates returns correct format."""
    from data_collection.energy_weather_data import get_yesterday_dates

    start, end = get_yesterday_dates()

    # Should be in YYYY-MM-DD format
    assert len(start) == 10
    assert len(end) == 10
    assert start == end  # Start and end should be the same day

    # Should be yesterday
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    assert start == yesterday


# ---------- APPEND TO HISTORICAL ----------


def test_append_to_historical_new_file(tmp_path, mock_weather_data):
    """Test creating a new file."""
    from data_collection.energy_weather_data import append_to_historical

    result = append_to_historical(
        mock_weather_data, save_dir=str(tmp_path), file_name="test.csv"
    )

    assert not result.empty
    assert (tmp_path / "test.csv").exists()
    assert len(result) == len(mock_weather_data)


def test_append_to_historical_existing_file(tmp_path, mock_weather_data):
    """Test appending to existing file."""
    from data_collection.energy_weather_data import append_to_historical

    # Create initial file
    initial_data = mock_weather_data.iloc[:1].copy()
    initial_data.to_csv(tmp_path / "test.csv", index=False)

    # Append new data (ensure it's for yesterday)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    new_data = mock_weather_data.copy()
    new_data["datetime"] = pd.date_range(
        start=yesterday, periods=len(new_data), freq="h", tz="UTC"
    )

    result = append_to_historical(
        new_data, save_dir=str(tmp_path), file_name="test.csv"
    )

    assert len(result) >= len(new_data)
    assert (tmp_path / "test.csv").exists()


def test_append_to_historical_duplicate_removal(tmp_path):
    """Test that duplicate datetimes keep only the most recent."""
    from data_collection.energy_weather_data import append_to_historical

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    # Create initial data
    initial = pd.DataFrame(
        {
            "datetime": pd.date_range(start=yesterday, periods=2, freq="h", tz="UTC"),
            "temperature_C": [10.0, 11.0],
        }
    )
    initial.to_csv(tmp_path / "test.csv", index=False)

    # Create duplicate data with different values
    duplicate = pd.DataFrame(
        {
            "datetime": pd.date_range(start=yesterday, periods=2, freq="h", tz="UTC"),
            "temperature_C": [15.0, 16.0],
        }
    )

    result = append_to_historical(
        duplicate, save_dir=str(tmp_path), file_name="test.csv"
    )

    # Should have 2 unique datetimes with updated values
    assert len(result) == 2
    assert result["temperature_C"].iloc[0] == 15.0


# ---------- FULL INTEGRATION TEST ----------


def test_collect_and_append_yesterday_with_errors(tmp_path, capsys):
    """Test handling of API errors during collection."""
    from data_collection.energy_weather_data import collect_and_append_yesterday

    with patch("requests.get", side_effect=Exception("API Error")):
        with pytest.raises(Exception):
            collect_and_append_yesterday(save_dir=str(tmp_path), file_name="test.csv")


def test_fetch_octopus_prices_no_tariff_link():
    """Test handling when product has no tariff links."""
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    product_data = {"results": [{"code": "AGILE-18-02-21", "links": []}]}  # No links

    rates_data = {
        "results": [
            {"valid_from": f"{yesterday}T00:00:00Z", "value_inc_vat": 25.0},
        ]
    }

    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        if "products/" in url and "electricity-tariffs" not in url:
            mock_resp.json.return_value = product_data
        elif "E-1R-AGILE-18-02-21-A" in url:
            # Falls back to constructed tariff code
            mock_resp.json.return_value = rates_data
        else:
            raise ValueError(f"Unexpected URL: {url}")
        return mock_resp

    with patch("requests.get", side_effect=mock_get):
        df = fetch_octopus_prices()
        assert not df.empty
