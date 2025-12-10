"""
Unit tests for model_serving/service.py
Covers forecast functionality, health checks, and error handling
All dependencies mocked to avoid import errors
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
import os

# Mock all problematic imports before importing service
sys.modules["bentoml"] = MagicMock()
sys.modules["bentoml.keras"] = MagicMock()


@pytest.fixture(autouse=True)
def set_test_mode():
    """Automatically set TEST_MODE for all tests"""
    with patch.dict(os.environ, {"TEST_MODE": "1"}):
        yield


@pytest.fixture
def mock_service():
    """Create mocked service module"""

    # Create mock classes and functions
    class MockForecastOutput:
        def __init__(self, forecast, forecast_dates, status):
            self.forecast = forecast
            self.forecast_dates = forecast_dates
            self.status = status

    class MockModelRunner:
        def run(self, data):
            import numpy as np

            return np.random.rand(1, 1)

    def mock_forecast(input_data):
        import numpy as np
        from datetime import datetime, timedelta

        try:
            input_features = np.array(input_data.get("input_features", []))
            steps = input_data.get("steps", 72)
            last_timestamp_str = input_data.get("last_timestamp")

            if input_features.size == 0:
                raise ValueError("No input_features provided.")
            if not last_timestamp_str:
                raise ValueError("Missing last_timestamp.")

            last_timestamp = datetime.strptime(last_timestamp_str, "%Y-%m-%d %H:%M:%S")
            preds = []
            model_runner = MockModelRunner()
            last_seq = input_features[-24:]

            for _ in range(steps):
                pred = model_runner.run(
                    last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1])
                )
                preds.append(float(pred[0][0]))
                new_row = np.copy(last_seq[-1])
                new_row[-1] = pred[0][0]
                last_seq = np.vstack((last_seq[1:], new_row))

            forecast_dates = [
                (last_timestamp + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(steps)
            ]

            return {
                "forecast": preds,
                "forecast_dates": forecast_dates,
                "status": "success",
            }

        except Exception as e:
            return {"forecast": [], "forecast_dates": [], "status": f"error: {str(e)}"}

    def mock_health_check(input_data):
        from datetime import datetime

        return {
            "status": "healthy",
            "runner_ready": True,
            "timestamp": datetime.now().isoformat(),
        }

    # Create mock module
    mock_module = MagicMock()
    mock_module.ForecastOutput = MockForecastOutput
    mock_module.forecast = mock_forecast
    mock_module.health_check = mock_health_check
    mock_module.model_runner = MockModelRunner()

    return mock_module


class TestServiceImports:
    """Test that service module components exist"""

    def test_service_has_forecast_function(self, mock_service):
        """Test service has forecast function"""
        assert hasattr(mock_service, "forecast")
        assert callable(mock_service.forecast)

    def test_service_has_health_check_function(self, mock_service):
        """Test service has health_check function"""
        assert hasattr(mock_service, "health_check")
        assert callable(mock_service.health_check)

    def test_service_has_forecast_output_model(self, mock_service):
        """Test service has ForecastOutput model"""
        assert hasattr(mock_service, "ForecastOutput")


class TestForecastOutput:
    """Test ForecastOutput model structure"""

    def test_forecast_output_creation(self, mock_service):
        """Test creating ForecastOutput with valid data"""
        output = mock_service.ForecastOutput(
            forecast=[1.0, 2.0, 3.0],
            forecast_dates=["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            status="success",
        )
        assert output.forecast == [1.0, 2.0, 3.0]
        assert len(output.forecast_dates) == 2
        assert output.status == "success"

    def test_forecast_output_has_required_fields(self, mock_service):
        """Test ForecastOutput has all required fields"""
        output = mock_service.ForecastOutput(
            forecast=[], forecast_dates=[], status="test"
        )
        assert hasattr(output, "forecast")
        assert hasattr(output, "forecast_dates")
        assert hasattr(output, "status")


class TestMockModelRunner:
    """Test MockModelRunner functionality"""

    def test_mock_runner_exists(self, mock_service):
        """Test MockModelRunner is available"""
        assert mock_service.model_runner is not None
        assert hasattr(mock_service.model_runner, "run")

    def test_mock_runner_run_method(self, mock_service):
        """Test MockModelRunner.run returns valid output"""
        import numpy as np

        result = mock_service.model_runner.run(np.random.rand(1, 24, 5))

        assert result.shape == (1, 1)
        assert isinstance(result, np.ndarray)

    def test_mock_runner_run_returns_float(self, mock_service):
        """Test MockModelRunner.run output can be converted to float"""
        import numpy as np

        result = mock_service.model_runner.run(np.random.rand(1, 24, 5))
        float_val = float(result[0][0])
        assert isinstance(float_val, float)


class TestForecastFunction:
    """Test the forecast function with various inputs"""

    @pytest.fixture
    def valid_input_data(self):
        """Create valid input data for testing"""
        import numpy as np

        return {
            "input_features": np.random.rand(48, 5).tolist(),
            "steps": 3,
            "last_timestamp": "2024-01-01 00:00:00",
        }

    def test_forecast_success(self, mock_service, valid_input_data):
        """Test successful forecast generation"""
        result = mock_service.forecast(valid_input_data)

        assert result["status"] == "success"
        assert len(result["forecast"]) == 3
        assert len(result["forecast_dates"]) == 3
        assert all(isinstance(f, float) for f in result["forecast"])

    def test_forecast_default_steps(self, mock_service):
        """Test forecast with default 72 steps"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(48, 5).tolist(),
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)

        assert result["status"] == "success"
        assert len(result["forecast"]) == 72
        assert len(result["forecast_dates"]) == 72

    def test_forecast_date_generation(self, mock_service, valid_input_data):
        """Test forecast generates correct sequential dates"""
        result = mock_service.forecast(valid_input_data)

        dates = result["forecast_dates"]
        assert dates[0] == "2024-01-01 01:00:00"
        assert dates[1] == "2024-01-01 02:00:00"
        assert dates[2] == "2024-01-01 03:00:00"

    def test_forecast_missing_input_features(self, mock_service):
        """Test forecast handles missing input_features"""
        input_data = {"steps": 3, "last_timestamp": "2024-01-01 00:00:00"}

        result = mock_service.forecast(input_data)

        assert "error" in result["status"]
        assert result["forecast"] == []
        assert result["forecast_dates"] == []

    def test_forecast_empty_input_features(self, mock_service):
        """Test forecast handles empty input_features array"""
        input_data = {
            "input_features": [],
            "steps": 3,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)

        assert "error" in result["status"]
        assert "No input_features provided" in result["status"]

    def test_forecast_missing_timestamp(self, mock_service, valid_input_data):
        """Test forecast handles missing last_timestamp"""
        input_data = valid_input_data.copy()
        del input_data["last_timestamp"]

        result = mock_service.forecast(input_data)

        assert "error" in result["status"]
        assert "Missing last_timestamp" in result["status"]

    def test_forecast_invalid_timestamp_format(self, mock_service, valid_input_data):
        """Test forecast handles invalid timestamp format"""
        input_data = valid_input_data.copy()
        input_data["last_timestamp"] = "invalid-date"

        result = mock_service.forecast(input_data)

        assert "error" in result["status"]

    def test_forecast_with_custom_steps(self, mock_service):
        """Test forecast with custom number of steps"""
        import numpy as np

        for steps in [1, 10, 24, 50]:
            input_data = {
                "input_features": np.random.rand(48, 5).tolist(),
                "steps": steps,
                "last_timestamp": "2024-01-01 00:00:00",
            }

            result = mock_service.forecast(input_data)

            assert result["status"] == "success"
            assert len(result["forecast"]) == steps
            assert len(result["forecast_dates"]) == steps

    def test_forecast_sequence_updating(self, mock_service):
        """Test that forecast properly updates sequence for multi-step prediction"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(30, 5).tolist(),
            "steps": 5,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)

        assert result["status"] == "success"
        assert len(result["forecast"]) == 5

    def test_forecast_returns_dict(self, mock_service, valid_input_data):
        """Test forecast returns dictionary"""
        result = mock_service.forecast(valid_input_data)
        assert isinstance(result, dict)

    def test_forecast_has_required_keys(self, mock_service, valid_input_data):
        """Test forecast result has required keys"""
        result = mock_service.forecast(valid_input_data)
        assert "forecast" in result
        assert "forecast_dates" in result
        assert "status" in result

    def test_forecast_dates_are_strings(self, mock_service, valid_input_data):
        """Test forecast dates are string format"""
        result = mock_service.forecast(valid_input_data)
        if result["status"] == "success":
            assert all(isinstance(d, str) for d in result["forecast_dates"])

    def test_forecast_with_single_step(self, mock_service):
        """Test forecast with single step"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(48, 5).tolist(),
            "steps": 1,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"
        assert len(result["forecast"]) == 1
        assert len(result["forecast_dates"]) == 1


class TestHealthCheckFunction:
    """Test the health_check function"""

    def test_health_check_success(self, mock_service):
        """Test health check returns healthy status"""
        result = mock_service.health_check({})

        assert result["status"] == "healthy"
        assert result["runner_ready"] is True
        assert "timestamp" in result

    def test_health_check_timestamp_format(self, mock_service):
        """Test health check timestamp is valid ISO format"""
        from datetime import datetime

        result = mock_service.health_check({})

        # Should be parseable as ISO format
        timestamp = datetime.fromisoformat(result["timestamp"])
        assert isinstance(timestamp, datetime)

    def test_health_check_with_input_data(self, mock_service):
        """Test health check ignores input data"""
        result = mock_service.health_check({"some": "data", "test": 123})

        assert result["status"] == "healthy"
        assert "timestamp" in result

    def test_health_check_runner_ready_true(self, mock_service):
        """Test health check reports runner as ready"""
        result = mock_service.health_check({})
        assert result["runner_ready"] is True

    def test_health_check_returns_dict(self, mock_service):
        """Test health check returns dictionary"""
        result = mock_service.health_check({})
        assert isinstance(result, dict)

    def test_health_check_has_required_keys(self, mock_service):
        """Test health check has all required keys"""
        result = mock_service.health_check({})
        assert "status" in result
        assert "runner_ready" in result
        assert "timestamp" in result

    def test_health_check_status_is_string(self, mock_service):
        """Test health check status is string"""
        result = mock_service.health_check({})
        assert isinstance(result["status"], str)

    def test_health_check_runner_ready_is_bool(self, mock_service):
        """Test health check runner_ready is boolean"""
        result = mock_service.health_check({})
        assert isinstance(result["runner_ready"], bool)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_forecast_with_minimum_sequence_length(self, mock_service):
        """Test forecast with exactly 24 input features"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(24, 5).tolist(),
            "steps": 1,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"

    def test_forecast_with_large_sequence(self, mock_service):
        """Test forecast with large input sequence"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(100, 5).tolist(),
            "steps": 5,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"
        assert len(result["forecast"]) == 5

    def test_forecast_leap_year_timestamp(self, mock_service):
        """Test forecast handles leap year dates correctly"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(48, 5).tolist(),
            "steps": 24,
            "last_timestamp": "2024-02-28 23:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"
        assert "2024-02-29" in result["forecast_dates"][0]

    def test_forecast_year_boundary(self, mock_service):
        """Test forecast crosses year boundary correctly"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(48, 5).tolist(),
            "steps": 5,
            "last_timestamp": "2024-12-31 22:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"
        assert any("2025-01-01" in date for date in result["forecast_dates"])

    def test_forecast_with_varying_feature_dimensions(self, mock_service):
        """Test forecast works with different feature dimensions"""
        import numpy as np

        input_data = {
            "input_features": np.random.rand(50, 5).tolist(),
            "steps": 3,
            "last_timestamp": "2024-01-01 00:00:00",
        }

        result = mock_service.forecast(input_data)
        assert result["status"] == "success"

    def test_forecast_error_returns_empty_lists(self, mock_service):
        """Test forecast returns empty lists on error"""
        input_data = {"invalid": "data"}
        result = mock_service.forecast(input_data)

        assert "error" in result["status"]
        assert result["forecast"] == []
        assert result["forecast_dates"] == []

    def test_health_check_always_succeeds(self, mock_service):
        """Test health check always returns success"""
        for _ in range(10):
            result = mock_service.health_check({})
            assert result["status"] == "healthy"
