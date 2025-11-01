import os
import pandas as pd
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Define file paths ---
ENGINEERED_PATH = os.path.join("data", "engineered_features.csv")
SELECTED_PATH = os.path.join("data", "selected_features.csv")

# --- Define selected features (as identified from Optuna + models) ---
SELECTED_FEATURES = [
    "datetime",
    "scaled_renewable_pct",
    "scaled_price_lag_1h",
    "scaled_price_lag_24h",
    "hour",
    "hour_sin",
    "scaled_carbon_lag_1h",
    "hour_cos",
    "scaled_log_solar_radiation_Wm2",
    "scaled_carbon_intensity_actual",
    "scaled_log_no2",
    "scaled_uk_gen_biomass_%",
    "is_peak_hour",
    "scaled_wind_lag_24h",
    "scaled_carbon_per_price",
    "scaled_fossil_pct",
    "scaled_uk_gen_wind_%",
    "scaled_log_uk_gen_solar_%",
    "scaled_carbon_rolling_24h_mean",
    "scaled_uk_gen_nuclear_%",
    "scaled_uk_gen_gas_%",
    "scaled_carbon_intensity_forecast",
    "scaled_wind_speed_mps",
    "scaled_temperature_C",
    "scaled_price_rolling_24h_mean",
    "scaled_wind_lag_1h",
    "retail_price_£_per_kWh"  # target column
]


def update_selected_features():
    """Update selected_features.csv with new rows from engineered_features.csv."""
    if not os.path.exists(ENGINEERED_PATH):
        print(f"❌ Missing file: {ENGINEERED_PATH}")
        return

    # --- Load engineered features ---
    full_df = pd.read_csv(ENGINEERED_PATH)
    full_df['datetime'] = pd.to_datetime(full_df['datetime'], utc=True)

    # --- If selected_features.csv does not exist, create it ---
    if not os.path.exists(SELECTED_PATH):
        print("selected_features.csv not found — creating a new one.")
        selected_df = full_df[SELECTED_FEATURES].copy()
        selected_df.to_csv(SELECTED_PATH, index=False)
        print(f"✅ Created selected_features.csv with {len(selected_df)} rows.")
        return

    # --- Load existing selected features ---
    feature_df = pd.read_csv(SELECTED_PATH)
    feature_df['datetime'] = pd.to_datetime(feature_df['datetime'], utc=True)

    # --- Find new rows ---
    last_datetime = feature_df['datetime'].max()
    new_data = full_df[full_df['datetime'] > last_datetime]

    if new_data.empty:
        print("✅ No new rows detected. selected_features.csv is already up-to-date.")
        return

    # --- Append new rows and remove duplicates ---
    new_feature_data = new_data[SELECTED_FEATURES]
    updated_df = pd.concat([feature_df, new_feature_data], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset="datetime", keep="last")

    # --- Save the updated file ---
    updated_df['datetime'] = updated_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    updated_df.to_csv(SELECTED_PATH, index=False)

    print(f"✅ Added {len(new_feature_data)} new rows to selected_features.csv (Total: {len(updated_df)})")

if __name__ == "__main__":
    update_selected_features()
