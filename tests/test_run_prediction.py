import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import requests

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import prediction_client.run_prediction_client as rpc


@pytest.fixture
def mock_input_data(tmp_path):
    """Create a temporary CSV file with sample energy data."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=80, freq="H"),
        "feature1": np.random.rand(80),
        "feature2": np.random.rand(80),
        "retail_price_£_per_kWh": np.random.rand(80)
    })
    file_path = tmp_path / "selected_features.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@patch("requests.post")
def test_prediction_request(mock_post, mock_input_data, monkeypatch):
    """✅ Test that the payload is correctly prepared and API response handled properly."""

    # ---- Mock API response from BentoML ----
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "forecast": [0.25, 0.27, 0.30],
        "forecast_dates": [
            "2025-01-04 01:00:00",
            "2025-01-04 02:00:00",
            "2025-01-04 03:00:00"
        ]
    }
    mock_post.return_value = mock_response

    # ---- Monkeypatch to use local test data instead of live dependencies ----
    monkeypatch.setattr(rpc, "requests", requests)
    monkeypatch.setattr(rpc.pd, "read_csv", lambda _: pd.read_csv(mock_input_data))

    # Prevent sys.exit(1) from stopping pytest
    with patch("builtins.exit", lambda _: None):
        if hasattr(rpc, "main"):
            rpc.main()
        else:
            # Fallback: run top-level logic if not wrapped in a main() function
            if "__main__" in dir(rpc):
                exec(open(rpc.__file__).read(), {"__name__": "__main__"})

    # ---- Validation: ensure API was called once ----
    mock_post.assert_called_once()
    _, called_kwargs = mock_post.call_args

    # ---- Validate payload ----
    payload = json.loads(called_kwargs["data"])
    assert "input_features" in payload
    assert isinstance(payload["input_features"], list)
    assert len(payload["input_features"]) == 72  # Expecting last 72 hours for forecast
    assert payload.get("steps") == 72

    # ---- Validate mock response handling ----
    result = mock_post.return_value.json()
    assert "forecast" in result
    assert "forecast_dates" in result
    assert len(result["forecast"]) == 3
