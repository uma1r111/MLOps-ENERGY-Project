import bentoml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Dict, Any
import tensorflow as tf
import os

# Define forecast response
class ForecastOutput(BaseModel):
    forecast: List[float]
    forecast_dates: List[str]
    status: str

# Load BentoML model
try:
    model_ref = bentoml.models.get("energy_model:latest")
    model_runner = bentoml.keras.get("energy_model:latest").to_runner()
    model = model_runner.model
except Exception as e:
    raise ValueError(f"❌ Failed to load Energy model: {str(e)}")

svc = bentoml.Service("energy_forecaster")

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def forecast(input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        input_features = np.array(input_data.get("input_features", []))
        steps = input_data.get("steps", 72)
        last_timestamp_str = input_data.get("last_timestamp")

        if input_features.size == 0:
            raise ValueError("No input_features provided")

        if not last_timestamp_str:
            raise ValueError("Missing last_timestamp")

        last_timestamp = datetime.strptime(last_timestamp_str, "%Y-%m-%d %H:%M:%S")

        preds = []
        last_seq = input_features[-24:]  # Assuming model uses 24-step sequences

        for _ in range(steps):
            pred = model.predict(last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1]))
            preds.append(pred[0, 0])
            last_seq = np.vstack((last_seq[1:], np.append(last_seq[-1, 1:], pred[0, 0])))

        forecast_dates = [(last_timestamp + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:%M:%S") for i in range(steps)]

        return {"forecast": preds, "forecast_dates": forecast_dates, "status": "success"}

    except Exception as e:
        return {"forecast": [], "forecast_dates": [], "status": f"error: {str(e)}"}

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health_check(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "healthy", "model_loaded": model is not None, "timestamp": datetime.now().isoformat()}
