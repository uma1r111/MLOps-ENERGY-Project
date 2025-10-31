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

# Load the latest saved BentoML model
try:
    model_runner = bentoml.keras.get("energy_model:latest").to_runner()
except Exception as e:
    raise ValueError(f"❌ Failed to load Energy model runner: {str(e)}")

svc = bentoml.Service("energy_forecaster", runners=[model_runner])

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def forecast(input_data: Dict[str, Any]) -> Dict[str, Any]:
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

        # Assuming model expects 24-step input sequence
        last_seq = input_features[-24:]

        for _ in range(steps):
            # Run prediction using BentoML runner
            pred = model_runner.run(last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1]))
            preds.append(float(pred[0][0]))

            # Shift window for next step
            new_row = np.copy(last_seq[-1])
            new_row[-1] = pred[0][0]
            last_seq = np.vstack((last_seq[1:], new_row))

        forecast_dates = [
            (last_timestamp + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(steps)
        ]

        return {"forecast": preds, "forecast_dates": forecast_dates, "status": "success"}

    except Exception as e:
        return {"forecast": [], "forecast_dates": [], "status": f"error: {str(e)}"}


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health_check(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "healthy",
        "runner_ready": model_runner is not None,
        "timestamp": datetime.now().isoformat()
    }
