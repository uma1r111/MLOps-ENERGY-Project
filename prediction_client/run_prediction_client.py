import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

# --- CONFIG: MOCK or REAL ---
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

print("\n🔹 Loading and preparing input data...")

# --- DATA PREP ---
# Use real CSV if available, otherwise fallback to mock
if os.path.exists("data/selected_features.csv"):
    df = pd.read_csv("data/selected_features.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    feature_cols = [
        c for c in df.columns if c not in ["datetime", "retail_price_£_per_kWh"]
    ]
    last_72_data = df[feature_cols].tail(72).values.tolist()
    last_timestamp = df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
else:
    # MOCK DATA
    feature_cols = [f"feature_{i}" for i in range(10)]
    last_72_data = np.random.rand(72, len(feature_cols)).tolist()
    last_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"✅ Prepared {len(last_72_data)} rows of input features.")
print(f"Last timestamp: {last_timestamp}")

# --- BUILD PAYLOAD ---
input_payload = {
    "input_features": last_72_data,
    "steps": 72,
    "last_timestamp": last_timestamp,
}


# --- PREDICTION FUNCTION ---
def predict(payload):
    if USE_MOCK:
        # Mock response
        forecast_values = np.random.rand(payload["steps"]).tolist()
        forecast_dates = [
            (
                datetime.strptime(payload["last_timestamp"], "%Y-%m-%d %H:%M:%S")
                + timedelta(hours=i + 1)
            ).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(payload["steps"])
        ]
        return {"forecast": forecast_values, "forecast_dates": forecast_dates}
    else:
        # Real API call
        response = requests.post(
            url="http://localhost:3000/forecast",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()


# --- RUN PREDICTION ---
try:
    result = predict(input_payload)
    print("✅ Prediction successful.")

    pred_df = pd.DataFrame(
        {
            "datetime": result["forecast_dates"],
            "predicted_retail_price_£_per_kWh": result["forecast"],
        }
    )
    pred_df.to_csv("bentoml_forecast_output.csv", index=False)
    print("💾 Saved predictions to bentoml_forecast_output.csv")

    print(f"\nAverage price: {np.mean(result['forecast']):.2f}")
    print(f"Min price: {np.min(result['forecast']):.2f}")
    print(f"Max price: {np.max(result['forecast']):.2f}")

except Exception as e:
    print("❌ Prediction error:", e)
