import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from datetime import timedelta
import bentoml

# ----------------------
# Step 1: Load Data
# ----------------------
df = pd.read_csv("data/selected_features.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Filter last 90 days from latest date
latest_date = df["datetime"].max()
start_date = latest_date - pd.Timedelta(days=90)
df = df[df["datetime"] > start_date].sort_values("datetime")

print(f"Filtered data rows: {len(df)}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# ----------------------
# Step 2: Prepare Data
# ----------------------
target_col = "retail_price_£_per_kWh"
target = df[target_col].values.reshape(-1, 1)
features = df.drop(columns=["datetime", target_col])

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

# Sequence creation
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

# Split train/val
train_size = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")

# ----------------------
# Step 3: Define Models
# ----------------------
lstm_params = {
    "n_units": 128,
    "n_lstm_layers": 1,
    "activation": "tanh",
    "dropout": 0.5,
    "recurrent_dropout": 0.1,
    "n_dense_units": 128,
    "learning_rate": 0.004902441025672476
}

gru_params = {
    "n_units": 96,
    "n_gru_layers": 1,
    "activation": "tanh",
    "dropout": 0.5,
    "recurrent_dropout": 0.1,
    "n_dense_units": 32,
    "learning_rate": 0.00808139279466219
}

models = {}

# ----------------------
# Step 4: Build and Train
# ----------------------
def build_lstm_model(params, input_shape):
    model = Sequential()
    for _ in range(params["n_lstm_layers"]):
        model.add(LSTM(params["n_units"], activation=params["activation"],
                       dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"],
                       return_sequences=False, input_shape=input_shape))
    model.add(Dense(params["n_dense_units"], activation=params["activation"]))
    model.add(Dense(1))
    opt = Adam(learning_rate=params["learning_rate"])
    model.compile(loss="mse", optimizer=opt)
    return model


def build_gru_model(params, input_shape):
    model = Sequential()
    for _ in range(params["n_gru_layers"]):
        model.add(GRU(params["n_units"], activation=params["activation"],
                      dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"],
                      return_sequences=False, input_shape=input_shape))
    model.add(Dense(params["n_dense_units"], activation=params["activation"]))
    model.add(Dense(1))
    opt = Adam(learning_rate=params["learning_rate"])
    model.compile(loss="mse", optimizer=opt)
    return model


# Train both
for name, (build_func, params) in {
    "Final_LSTM": (build_lstm_model, lstm_params),
    "Final_GRU": (build_gru_model, gru_params)
}.items():
    print(f"\n🔧 Training {name} model...")
    model = build_func(params, (SEQ_LEN, X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1,
        shuffle=False
    )
    y_pred = model.predict(X_val)
    y_true_inv = scaler_y.inverse_transform(y_val)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    models[name] = {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "params": params
    }
    print(f"{name} MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ----------------------
# Step 5: Select Best Model
# ----------------------
best_model_name = min(models.keys(), key=lambda x: models[x]["mae"])
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")
print(f"MAE: {best_model['mae']:.4f}, RMSE: {best_model['rmse']:.4f}")

# ----------------------
# Step 6: Retrain Best Model on Full Dataset
# ----------------------
print("\nRetraining best model on full dataset...")
full_model = build_lstm_model(lstm_params, (SEQ_LEN, X_train.shape[2])) if "LSTM" in best_model_name else build_gru_model(gru_params, (SEQ_LEN, X_train.shape[2]))
full_model.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=1, shuffle=False)

# ----------------------
# Step 7: Save Model with BentoML (FIXED)
# ----------------------
try:
    # Make the model inference-only by calling it once to build the graph
    dummy_input = np.zeros((1, SEQ_LEN, X_scaled.shape[1]), dtype=np.float32)
    _ = full_model.predict(dummy_input, verbose=0)

    # Use keras.save_model instead of deprecated tensorflow.save_model
    saved_model = bentoml.keras.save_model(
        name="energy_model",
        model=full_model,
        signatures={"__call__": {"batchable": True}},
        metadata={
            "model_type": best_model_name,
            "best_params": best_model["params"],
            "mae": float(best_model["mae"]),
            "rmse": float(best_model["rmse"]),
            "seq_length": SEQ_LEN,
            "n_features": X_scaled.shape[1],
            "features": list(features.columns),
            "target": target_col,
            "training_date": str(pd.Timestamp.now()),
            "data_date_range": f"{df['datetime'].min()} to {df['datetime'].max()}"
        }
    )

    print(f"   ✅ Model saved to BentoML: {saved_model.tag}")
    print(f"      Model name: {saved_model.tag.name}")
    print(f"      Version: {saved_model.tag.version}")

except Exception as e:
    print(f"   ❌ Failed to save model: {e}")
    import traceback
    traceback.print_exc()
    raise  # Re-raise to fail the workflow if model save fails


# ----------------------
# Step 8: Forecast Next 72 Hours
# ----------------------
PREDICT_HORIZON = 72
print(f"\nForecasting next {PREDICT_HORIZON} hours...")

last_seq = X_scaled[-SEQ_LEN:]
predictions = []

for _ in range(PREDICT_HORIZON):
    pred = full_model.predict(last_seq.reshape(1, SEQ_LEN, X_scaled.shape[1]), verbose=0)
    predictions.append(pred[0, 0])
    new_row = np.append(last_seq[-1, 1:], pred[0, 0])
    last_seq = np.vstack((last_seq[1:], new_row))

future_preds_inv = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
future_dates = pd.date_range(start=df["datetime"].iloc[-1] + pd.Timedelta(hours=1), periods=PREDICT_HORIZON, freq="H")

output_df = pd.DataFrame({"datetime": future_dates, "predicted_retail_price_£_per_kWh": future_preds_inv.flatten()})
output_df.to_csv("data/predictions.csv", index=False)

print("✅ retail_price per kWh predictions for next 3 days saved to predictions.csv")
print(f"Average predicted Retail price per kwh: {future_preds_inv.mean():.2f}")
print(f"Min predicted Retail price per kwh: {future_preds_inv.min():.2f}")
print(f"Max predicted Retail price per kwh: {future_preds_inv.max():.2f}")
print(f"Prediction period: {future_dates[0]} → {future_dates[-1]}")

print("\n🎉 LSTM-GRU training and prediction workflow completed successfully!")
