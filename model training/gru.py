import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
MLFLOW_TRACKING_URI = "http://54.226.40.241:8000/"
EXPERIMENT_NAME = "UK Energy - Model Training"
DATA_PATH = "../data/selected_features.csv"
TARGET_COL = "retail_price_£_per_kWh"
PREDICT_HORIZON = 72  # next 3 days (hourly)
LOOKBACK = 24  # past 24 hours for training

# ---------------- MODEL PARAMS ----------------
# --- Use this block for Final_GRU_MultiOutput_Keras ---
run_name = "Final_GRU_MultiOutput_Keras"
params = {
    "n_units": 96,
    "n_gru_layers": 1,
    "activation": "tanh",
    "dropout": 0.5,
    "recurrent_dropout": 0.1,
    "n_dense_units": 32,
    "learning_rate": 0.00808139279466219
}

# --- Uncomment this block for Final_GRU_MultiOutput_Optuna ---
# run_name = "Final_GRU_MultiOutput_Optuna"
# params = {
#     "n_units": 34,
#     "n_dense_units": 44,
#     "dropout": 0.10903765421610254,
#     "recurrent_dropout": 0.07197859168062842,
#     "learning_rate": 0.0025316396529251263,
#     "activation": "relu",
#     "n_gru_layers": 1
# }

# ---------------- SETUP ----------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[(df["datetime"] >= "2025-08-02") & (df["datetime"] <= "2025-10-27")]
df = df.sort_values("datetime").reset_index(drop=True)

target = df[TARGET_COL].values.reshape(-1, 1)
features = df.drop(columns=["datetime", TARGET_COL]).values
print(f"Loaded {len(df)} rows from {df['datetime'].min()} to {df['datetime'].max()}")

# ---------------- DATA PREPARATION ----------------
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_features = scaler_x.fit_transform(features)
scaled_target = scaler_y.fit_transform(target)

X, y = [], []
for i in range(LOOKBACK, len(scaled_features) - PREDICT_HORIZON):
    X.append(scaled_features[i - LOOKBACK:i])
    y.append(scaled_target[i:i + PREDICT_HORIZON].flatten())

X, y = np.array(X), np.array(y)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------- MODEL BUILDING ----------------
def build_gru_model(input_shape, params):
    model = Sequential()
    for i in range(params["n_gru_layers"]):
        return_sequences = i < params["n_gru_layers"] - 1
        model.add(GRU(params["n_units"],
                      activation=params["activation"],
                      dropout=params["dropout"],
                      recurrent_dropout=params["recurrent_dropout"],
                      return_sequences=return_sequences,
                      input_shape=input_shape if i == 0 else None))
    model.add(Dense(params["n_dense_units"], activation=params["activation"]))
    model.add(Dense(PREDICT_HORIZON))
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

# ---------------- TRAINING & MLflow LOGGING ----------------
with mlflow.start_run(run_name=run_name):
    mlflow.log_params(params)

    model = build_gru_model((LOOKBACK, X.shape[2]), params)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_split=0.2, verbose=1)

    preds = model.predict(X_test)
    preds_inv = scaler_y.inverse_transform(preds)
    y_test_inv = scaler_y.inverse_transform(y_test)

    rmse = sqrt(mean_squared_error(y_test_inv.flatten(), preds_inv.flatten()))
    mae = mean_absolute_error(y_test_inv.flatten(), preds_inv.flatten())

    mlflow.log_metric("final_rmse", rmse)
    mlflow.log_metric("final_mae", mae)

    # Save model
    model.save("gru_model.h5")
    mlflow.log_artifact("gru_model.h5")

    mlflow.set_tag("model_type", "GRU_MultiOutput")
    mlflow.set_tag("data_window", f"{LOOKBACK}h_lookback_{PREDICT_HORIZON}h_forecast")

print("✅ GRU model training and MLflow logging completed successfully.")