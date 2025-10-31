import requests
import pandas as pd
import numpy as np
import json

print("\n🔹 Loading and preparing input data...")

try:
    df = pd.read_csv("data/selected_features.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Select all model features except datetime and target
    feature_cols = [
        c for c in df.columns if c not in ["datetime", "retail_price_£_per_kWh"]
    ]
    last_72_data = df[feature_cols].tail(72).values.tolist()

    if len(last_72_data) < 72:
        raise ValueError("Insufficient data: need at least 72 recent rows of features.")

    last_timestamp = df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

    print(f"✅ Prepared {len(last_72_data)} rows of input features.")
    print(f"Last timestamp: {last_timestamp}")

except Exception as e:
    print("❌ Error preparing input data:", e)
    exit(1)

# Build request payload
input_payload = {
    "input_features": last_72_data,
    "steps": 72,
    "last_timestamp": last_timestamp,
}

print("\n🚀 Sending request to BentoML API...")
try:
    response = requests.post(
        url="http://localhost:3000/forecast",
        headers={"Content-Type": "application/json"},
        data=json.dumps(input_payload),
    )

    if response.status_code == 200:
        result = response.json()
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

    else:
        print(f"❌ Request failed ({response.status_code}):", response.text)

except Exception as e:
    print("❌ Error connecting to BentoML service:", e)
    exit(1)
