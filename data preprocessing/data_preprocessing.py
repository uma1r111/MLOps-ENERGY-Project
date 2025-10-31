import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ========== FILE PATHS ==========
RAW_CSV = "data/uk_energy_data.csv"
FE_CSV = "data/engineered_features.csv"

# ========== LOAD DATA ==========
print("Loading raw data...")
raw_df = pd.read_csv(RAW_CSV)
raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
raw_df.sort_values("datetime", inplace=True)
print(f"Raw data: {raw_df.shape}")

# Load previous feature-engineered data if exists
if os.path.exists(FE_CSV):
    print(f"Loading existing feature-engineered data from {FE_CSV}...")
    prev_df = pd.read_csv(FE_CSV)
    prev_df['datetime'] = pd.to_datetime(prev_df['datetime'])
    prev_df.sort_values("datetime", inplace=True)
    print(f"Previous FE data: {prev_df.shape}")
else:
    print("No previous feature-engineered data found. Creating new...")
    prev_df = pd.DataFrame()

# Identify new rows to process
if not prev_df.empty:
    new_df = raw_df[~raw_df['datetime'].isin(prev_df['datetime'])].copy()
    print(f"New rows to process: {len(new_df)}")
else:
    new_df = raw_df.copy()
    print(f"Processing all {len(new_df)} rows (first run)")

if new_df.empty:
    print("✅ No new data to process. Exiting.")
    exit()

# ========== 1. TEMPORAL FEATURES ==========
print("\n1️⃣ Creating temporal features...")
new_df['hour'] = new_df['datetime'].dt.hour
new_df['day_of_week'] = new_df['datetime'].dt.dayofweek
new_df['month'] = new_df['datetime'].dt.month
new_df['day_of_month'] = new_df['datetime'].dt.day
new_df['week_of_year'] = new_df['datetime'].dt.isocalendar().week
new_df['is_weekend'] = (new_df['day_of_week'] >= 5).astype(int)
new_df['is_peak_hour'] = new_df['hour'].between(17, 21).astype(int)
new_df['is_night'] = (new_df['hour'].between(23, 23) | new_df['hour'].between(0, 6)).astype(int)

# Cyclical encoding
new_df['hour_sin'] = np.sin(2 * np.pi * new_df['hour'] / 24)
new_df['hour_cos'] = np.cos(2 * np.pi * new_df['hour'] / 24)
new_df['month_sin'] = np.sin(2 * np.pi * new_df['month'] / 12)
new_df['month_cos'] = np.cos(2 * np.pi * new_df['month'] / 12)

# ========== 2. INTERACTION FEATURES ==========
print("2️⃣ Creating interaction features...")
new_df['renewable_pct'] = new_df['uk_gen_wind_%'] + new_df['uk_gen_solar_%']
new_df['fossil_pct'] = new_df['uk_gen_gas_%']
new_df['heating_demand'] = (18 - new_df['temperature_C']).clip(lower=0)
new_df['cooling_demand'] = (new_df['temperature_C'] - 22).clip(lower=0)
new_df['wind_solar_combined'] = new_df['uk_gen_wind_%'] * new_df['solar_radiation_Wm2']
new_df['carbon_per_price'] = new_df['carbon_intensity_actual'] / (new_df['retail_price_£_per_kWh'] + 1e-6)

# ========== 3. LOG TRANSFORMATIONS ==========
print("3️⃣ Applying log transformations...")
log_features = [
    "so2", "pm2_5", "co", "no2", "pm10",
    "solar_radiation_Wm2", "uk_gen_solar_%", "aqi_us"
]

for col in log_features:
    if col in new_df.columns:
        new_df[f"log_{col}"] = np.log1p(new_df[col].fillna(0))

# ========== 4. LAG & ROLLING FEATURES (COMBINED DATA) ==========
print("4️⃣ Computing lag and rolling features...")

# Combine previous and new data for lag calculations
if not prev_df.empty:
    # Get necessary columns from previous data
    lag_cols = ['datetime', 'retail_price_£_per_kWh', 'carbon_intensity_actual',
                'temperature_C', 'uk_gen_wind_%']
    combined_df = pd.concat([prev_df[lag_cols], new_df[lag_cols]])
else:
    combined_df = new_df[['datetime', 'retail_price_£_per_kWh', 'carbon_intensity_actual',
                           'temperature_C', 'uk_gen_wind_%']].copy()

combined_df.sort_values("datetime", inplace=True)
combined_df.set_index("datetime", inplace=True)

# Lag features
combined_df['price_lag_1h'] = combined_df['retail_price_£_per_kWh'].shift(1)
combined_df['price_lag_24h'] = combined_df['retail_price_£_per_kWh'].shift(24)
combined_df['carbon_lag_1h'] = combined_df['carbon_intensity_actual'].shift(1)
combined_df['carbon_lag_24h'] = combined_df['carbon_intensity_actual'].shift(24)
combined_df['temp_lag_1h'] = combined_df['temperature_C'].shift(1)
combined_df['temp_lag_24h'] = combined_df['temperature_C'].shift(24)
combined_df['wind_lag_1h'] = combined_df['uk_gen_wind_%'].shift(1)
combined_df['wind_lag_24h'] = combined_df['uk_gen_wind_%'].shift(24)

# Rolling features
combined_df['price_rolling_24h_mean'] = combined_df['retail_price_£_per_kWh'].rolling(24, min_periods=1).mean()
combined_df['price_rolling_24h_std'] = combined_df['retail_price_£_per_kWh'].rolling(24, min_periods=1).std()
combined_df['temp_rolling_24h_mean'] = combined_df['temperature_C'].rolling(24, min_periods=1).mean()
combined_df['temp_rolling_24h_std'] = combined_df['temperature_C'].rolling(24, min_periods=1).std()
combined_df['carbon_rolling_24h_mean'] = combined_df['carbon_intensity_actual'].rolling(24, min_periods=1).mean()
combined_df['carbon_rolling_24h_std'] = combined_df['carbon_intensity_actual'].rolling(24, min_periods=1).std()
combined_df['wind_rolling_24h_std'] = combined_df['uk_gen_wind_%'].rolling(24, min_periods=1).std()

combined_df.reset_index(inplace=True)

# Merge lag/rolling features back to new_df
lag_rolling_cols = ['datetime', 'price_lag_1h', 'price_lag_24h',
                    'carbon_lag_1h', 'carbon_lag_24h', 'temp_lag_1h', 'temp_lag_24h',
                    'wind_lag_1h', 'wind_lag_24h', 'price_rolling_24h_mean',
                    'price_rolling_24h_std',
                    'temp_rolling_24h_mean', 'temp_rolling_24h_std',
                    'carbon_rolling_24h_mean', 'carbon_rolling_24h_std',
                    'wind_rolling_24h_std']

new_df = new_df.merge(combined_df[lag_rolling_cols], on='datetime', how='left')

# ========== 5. SCALING ==========
print("5️⃣ Scaling features with StandardScaler...")

scale_features = [
    # Weather features
    "temperature_C", "wind_speed_mps", "humidity_%", "cloud_cover_%", "o3",

    # Carbon intensity
    "carbon_intensity_actual", "carbon_intensity_forecast",

    # Generation mix
    "uk_gen_wind_%", "uk_gen_imports_%", "uk_gen_biomass_%",
    "uk_gen_nuclear_%", "uk_gen_gas_%",

    # Log-transformed features
    "log_so2", "log_pm2_5", "log_co", "log_no2", "log_pm10",
    "log_solar_radiation_Wm2", "log_uk_gen_solar_%", "log_aqi_us",

    # Engineered features
    "renewable_pct", "fossil_pct", "heating_demand", "cooling_demand",
    "carbon_per_price",

    # Lag features
    "price_lag_1h", "price_lag_24h",
    "carbon_lag_1h", "carbon_lag_24h", "temp_lag_1h", "temp_lag_24h",
    "wind_lag_1h", "wind_lag_24h",

    # Rolling features
    "price_rolling_24h_mean", "price_rolling_24h_std",
    "temp_rolling_24h_mean", "temp_rolling_24h_std",
    "carbon_rolling_24h_mean", "carbon_rolling_24h_std", "wind_rolling_24h_std"
]

# Only scale features that exist in new_df
scale_features = [col for col in scale_features if col in new_df.columns]

# Create scaler and fit/transform
scaler = StandardScaler()
scaled_values = scaler.fit_transform(new_df[scale_features].fillna(0))
scaled_df = pd.DataFrame(
    scaled_values,
    columns=[f"scaled_{col}" for col in scale_features],
    index=new_df.index
)

# Join scaled features to new_df
new_df = new_df.join(scaled_df)

# ========== 6. COMBINE AND SAVE ==========
print("\n6️⃣ Combining and saving final dataset...")

# Combine previous and new data
if not prev_df.empty:
    final_df = pd.concat([prev_df, new_df], ignore_index=True)
else:
    final_df = new_df.copy()

# Remove duplicates (keep latest)
final_df.drop_duplicates(subset="datetime", keep="last", inplace=True)
final_df.sort_values("datetime", inplace=True)

# Save to CSV
os.makedirs(os.path.dirname(FE_CSV), exist_ok=True)
final_df.to_csv(FE_CSV, index=False)

print(f"\n✅ Feature engineering complete!")
print(f"   Saved: {FE_CSV}")
print(f"   Final shape: {final_df.shape}")
print(f"   Date range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
print(f"   New features added: {len(new_df)}")
print(f"   Total features: {final_df.shape[1]}")
print(f"\nFeature columns: {final_df.shape[1]} total")
print(f"   - Original: 23")
print(f"   - Temporal: 13")
print(f"   - Interactions: 6")
print(f"   - Log transforms: 8")
print(f"   - Lag features: 9")
print(f"   - Rolling features: 8")
print(f"   - Scaled features: {len([col for col in final_df.columns if col.startswith('scaled_')])}")
