import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the client module you want to test
import prediction_client.run_prediction_client as rpc

@pytest.fixture
def mock_input_payload():
    return {
        "input_features": np.random.rand(72, 10).tolist(),
        "steps": 5,
        "last_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ----------------- MOCK PREDICT FUNCTION -----------------
@pytest.fixture
def patch_predict():
    with patch("prediction_client.run_prediction_client.predict") as mock_fn:
        def side_effect(payload):
            steps = payload.get("steps", 5)
            last_ts = datetime.strptime(payload["last_timestamp"], "%Y-%m-%d %H:%M:%S")
            forecast = np.random.rand(steps).tolist()
            forecast_dates = [
                (last_ts + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(steps)
            ]
            return {"forecast": forecast, "forecast_dates": forecast_dates}

        mock_fn.side_effect = side_effect
        yield mock_fn


# ----------------- TESTS -----------------
def test_predict_returns_expected_dict(patch_predict, mock_input_payload):
    result = rpc.predict(mock_input_payload)
    assert isinstance(result, dict)
    assert "forecast" in result
    assert "forecast_dates" in result
    assert len(result["forecast"]) == mock_input_payload["steps"]
    assert len(result["forecast_dates"]) == mock_input_payload["steps"]
    assert all(isinstance(x, float) for x in result["forecast"])


def test_forecast_dataframe_creation(patch_predict, mock_input_payload, tmp_path):
    result = rpc.predict(mock_input_payload)

    # Simulate saving CSV as in the main script
    pred_df = rpc.pd.DataFrame(
        {
            "datetime": result["forecast_dates"],
            "predicted_retail_price_£_per_kWh": result["forecast"],
        }
    )
    file_path = tmp_path / "output.csv"
    pred_df.to_csv(file_path, index=False)

    # Check CSV exists and has correct length
    assert file_path.exists()
    df_loaded = rpc.pd.read_csv(file_path)
    assert len(df_loaded) == mock_input_payload["steps"]
    assert "datetime" in df_loaded.columns
    assert "predicted_retail_price_£_per_kWh" in df_loaded.columns


def test_predict_raises_exception_for_missing_timestamp():
    payload = {"input_features": np.random.rand(24, 10).tolist(), "steps": 5}
    with patch("prediction_client.run_prediction_client.predict") as mock_fn:
        mock_fn.side_effect = ValueError("Missing last_timestamp")
        with pytest.raises(ValueError):
            mock_fn(payload)
