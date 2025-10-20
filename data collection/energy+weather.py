import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EIA_API_KEY = os.getenv("EIA_API_KEY")

# ---------- 1️⃣ OPEN-METEO WEATHER DATA ----------
def fetch_weather_data(lat=51.5072, lon=-0.1276):  # Default: London
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly="
        f"temperature_2m,relative_humidity_2m,wind_speed_10m,cloudcover,shortwave_radiation"
    )
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Weather API request failed: {response.status_code}")
    
    data = response.json()
    
    # Safety check
    if "hourly" not in data or "time" not in data["hourly"]:
        print("⚠️ Unexpected response from Open-Meteo API:")
        print(data)
        raise KeyError("Missing 'hourly' data in Open-Meteo response.")
    
    df_weather = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature_C": data["hourly"]["temperature_2m"],
        "humidity_%": data["hourly"]["relative_humidity_2m"],
        "wind_speed_mps": data["hourly"]["wind_speed_10m"],
        "cloud_cover_%": data["hourly"]["cloudcover"],
        "solar_radiation_Wm2": data["hourly"]["shortwave_radiation"],
    })
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    return df_weather


# ---------- 2️⃣ UK CARBON INTENSITY API ----------
def fetch_carbon_intensity():
    url = "https://api.carbonintensity.org.uk/intensity"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Carbon Intensity API request failed: {response.status_code}")
    
    data = response.json()
    records = data.get("data", [])
    
    if not records:
        raise ValueError("No data returned from Carbon Intensity API.")
    
    # Flatten nested structure
    flat_records = []
    for r in records:
        flat_records.append({
            "datetime": r.get("from"),
            "carbon_intensity_actual": r.get("intensity", {}).get("actual"),
            "carbon_intensity_forecast": r.get("intensity", {}).get("forecast"),
            "carbon_index": r.get("intensity", {}).get("index")
        })
    
    df_carbon = pd.DataFrame(flat_records)
    df_carbon["datetime"] = pd.to_datetime(df_carbon["datetime"])
    return df_carbon



# ---------- 3️⃣ EIA ELECTRICITY DATA ----------
def fetch_eia_data():
    """Fetch electricity generation data from EIA API (v2) with error handling."""
    try:
        url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={EIA_API_KEY}&frequency=hourly&data[0]=value"
        response = requests.get(url)
        data = response.json()

        # Debug print if structure unexpected
        if "response" not in data:
            print("⚠️ Unexpected EIA response format:", data)
            raise KeyError("'response' key not found in EIA API output.")

        records = data["response"]["data"]

        if not records:
            print("⚠️ No data returned from EIA.")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Parse and clean
        if "period" in df.columns:
            df["datetime"] = pd.to_datetime(df["period"])
        df = df.rename(columns={"value": "electricity_value"})
        df = df[["datetime", "electricity_value"]]

        print(f"✅ EIA data fetched successfully: {len(df)} records")
        return df

    except Exception as e:
        print(f"❌ Error fetching EIA data: {e}")
        return pd.DataFrame()


# ---------- 4️⃣ MERGE & SAVE ----------
def collect_and_merge_data():
    print("Fetching weather data...")
    weather_df = fetch_weather_data()
    print("Fetching carbon intensity data...")
    carbon_df = fetch_carbon_intensity()
    print("Fetching EIA electricity data...")
    eia_df = fetch_eia_data()

    # Standardize datetime formats
    for df in [weather_df, carbon_df, eia_df]:
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)

    print("Merging datasets...")
    df_merged = pd.merge_asof(
        weather_df.sort_values("datetime"),
        carbon_df.sort_values("datetime"),
        on="datetime",
        direction="nearest"
    )
    df_merged = pd.merge_asof(
        df_merged.sort_values("datetime"),
        eia_df.sort_values("datetime"),
        on="datetime",
        direction="nearest"
    )

    df_merged.to_csv("merged_energy_weather_data.csv", index=False)
    print("✅ Data successfully merged and saved!")


if __name__ == "__main__":
    collect_and_merge_data()
