import mlflow
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ---------------- CONFIG ----------------
MLFLOW_TRACKING_URI = "http://54.226.40.241:8000/"
EXPERIMENT_NAME = "UK Energy - ML Model Training "
DATA_PATH = "../data/selected_features.csv"
TARGET_COL = "retail_price_£_per_kWh"
PREDICT_HORIZON = 72  # next 3 days (hourly)    
ROLLING_WINDOW = 7 * 24  # 7-day rolling validation window

# Fixed SARIMAX parameters
best_params = {
    "p": 0,
    "d": 0,
    "q": 0,
    "P": 2,
    "D": 0,
    "Q": 2,
    "seasonal_period": 6
}

# ---------------- SETUP ----------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[(df["datetime"] >= "2025-08-02") & (df["datetime"] <= "2025-10-27")]
df = df.sort_values("datetime").reset_index(drop=True)

target = df[TARGET_COL]
exog = df.drop(columns=["datetime", TARGET_COL])
print(f"Loaded {len(df)} rows from {df['datetime'].min()} to {df['datetime'].max()}")

# ---------------- WALK-FORWARD VALIDATION ----------------
def walk_forward_validation(params):
    rmse_scores = []
    horizon = ROLLING_WINDOW

    for i in range(0, len(df) - horizon, horizon):
        train_end = i + horizon
        test_end = train_end + horizon
        if test_end > len(df):
            break

        train_y = target.iloc[:train_end]
        test_y = target.iloc[train_end:test_end]
        train_exog = exog.iloc[:train_end].values
        test_exog = exog.iloc[train_end:test_end].values

        try:
            model = SARIMAX(
                endog=train_y,
                exog=train_exog,
                order=(params["p"], params["d"], params["q"]),
                seasonal_order=(params["P"], params["D"], params["Q"], params["seasonal_period"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            preds = fitted.forecast(steps=len(test_y), exog=test_exog)
            rmse = sqrt(mean_squared_error(test_y, preds))
            rmse_scores.append(rmse)
        except Exception as e:
            print("Error in fold:", e)
            rmse_scores.append(np.nan)

    return np.nanmean(rmse_scores)

# ---------------- FINAL MODEL TRAINING ----------------
with mlflow.start_run(run_name="Final_SARIMAX_FixedParams"):
    mlflow.log_params(best_params)

    model = SARIMAX(
        endog=target,
        exog=exog,
        order=(best_params["p"], best_params["d"], best_params["q"]),
        seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], best_params["seasonal_period"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    print("Model fitted successfully.")

    # Forecast next 72 hours
    last_exog = exog.iloc[-PREDICT_HORIZON:].values
    preds = fitted_model.forecast(steps=PREDICT_HORIZON, exog=last_exog)
    future_dates = pd.date_range(start=df["datetime"].iloc[-1] + pd.Timedelta(hours=1),
                                 periods=PREDICT_HORIZON, freq="H")

    future_df = pd.DataFrame({"datetime": future_dates, TARGET_COL: preds})
    future_df.to_csv("future_retail_price_predictions.csv", index=False)

    mlflow.log_artifact("future_retail_price_predictions.csv")

    final_rmse = walk_forward_validation(best_params)
    mlflow.log_metric("final_rmse", final_rmse)
    mlflow.set_tag("model_type", "SARIMAX_WalkForward")
    mlflow.set_tag("param_mode", "Fixed")

print("✅ Walk-forward SARIMAX training completed successfully.")
