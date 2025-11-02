import sys
from unittest.mock import MagicMock

# ---------- MOCK HEAVY DEPENDENCIES FIRST ----------
# These must come *before* importing your service
sys.modules["bentoml"] = MagicMock()
sys.modules["tensorflow"] = MagicMock()
sys.modules["keras"] = MagicMock()

# ---------- THEN IMPORT OTHER LIBRARIES ----------
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch

# Import your functions after mocking
from model_serving.service import forecast, health_check


# ---------- FIXTURES ----------
@pytest.fixture
def mock_input_data():
    """Mock input data for forecasting."""
    return {
        "input_features": np.random.rand(48, 10).tolist(),  # 48 time steps, 10 features
        "steps": 5,
        "last_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------- FORECAST TESTS ----------
@patch("model_serving.service.model_runner")
def test_forecast_success(mock_runner, mock_input_data):
    """Check that forecast runs successfully with valid input."""
    # Mock model output (one value per forecast step)
    mock_runner.run.return_value = np.random.rand(mock_input_data["steps"], 1)

    response = forecast(mock_input_data)

    assert isinstance(response, dict)
    assert response["status"] == "success"
    assert len(response["forecast"]) == mock_input_data["steps"]
    assert len(response["forecast_dates"]) == mock_input_data["steps"]
    assert all(isinstance(x, float) for x in response["forecast"])


def test_forecast_missing_features():
    """Handle case when features are missing."""
    input_data = {
        "steps": 5,
        "last_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    response = forecast(input_data)

    assert isinstance(response, dict)
    assert "error" in response["status"].lower()


def test_forecast_missing_timestamp():
    """Handle case when timestamp is missing."""
    input_data = {
        "input_features": np.random.rand(24, 10).tolist(),
        "steps": 5,
    }

    response = forecast(input_data)

    assert isinstance(response, dict)
    assert "error" in response["status"].lower()


# ---------- HEALTH CHECK TEST ----------
def test_health_check():
    """Health check endpoint should return healthy status."""
    response = health_check({})

    assert isinstance(response, dict)
    assert "status" in response
    assert response["status"] == "healthy"
    assert "runner_ready" in response
    assert isinstance(response["timestamp"], str)
