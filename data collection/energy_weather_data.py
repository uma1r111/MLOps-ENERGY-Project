import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ---------- DATE SETUP ----------
def get_yesterday_dates():
    """Return yesterday's start and end date (UTC)."""
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    return str(yesterday), str(yesterday)


# ---------- 1️⃣ OPEN-METEO WEATHER DATA ----------
def fetch_weather_data(lat=51.5072, lon=-0.1276):
    """Fetch yesterday's hourly weather data for given coordinates."""
    start_date, end_date = get_yesterday_dates()
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"cloudcover,shortwave_radiation"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    df_weather = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature_C": data["hourly"]["temperature_2m"],
        "humidity_%": data["hourly"]["relative_humidity_2m"],
        "wind_speed_mps": data["hourly"]["wind_speed_10m"],
        "cloud_cover_%": data["hourly"]["cloudcover"],
        "solar_radiation_Wm2": data["hourly"]["shortwave_radiation"],
    })
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"], utc=True)
    return df_weather


# ---------- 2️⃣ OPEN-METEO AIR QUALITY DATA ----------
def fetch_air_quality(lat=51.5072, lon=-0.1276):
    """Fetch yesterday's air quality and pollutant data."""
    start_date, end_date = get_yesterday_dates()
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone,us_aqi"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    df_aqi = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"], utc=True),
        "pm10": data["hourly"]["pm10"],
        "pm2_5": data["hourly"]["pm2_5"],
        "co": data["hourly"]["carbon_monoxide"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "so2": data["hourly"]["sulphur_dioxide"],
        "o3": data["hourly"]["ozone"],
        "aqi_us": data["hourly"]["us_aqi"],
    })
    return df_aqi


# ---------- 3️⃣ UK CARBON INTENSITY ----------
def fetch_carbon_intensity():
    """Fetch UK carbon intensity for yesterday."""
    start_date, _ = get_yesterday_dates()
    url = f"https://api.carbonintensity.org.uk/intensity/date/{start_date}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("data", [])

    records = [{
        "datetime": r.get("from"),
        "carbon_intensity_actual": r.get("intensity", {}).get("actual"),
        "carbon_intensity_forecast": r.get("intensity", {}).get("forecast"),
        "carbon_index": r.get("intensity", {}).get("index"),
    } for r in data]

    df_carbon = pd.DataFrame(records)
    df_carbon["datetime"] = pd.to_datetime(df_carbon["datetime"], utc=True)
    return df_carbon


# ---------- 4️⃣ UK GENERATION MIX ----------
def fetch_carbon_generation_mix():
    """Fetch yesterday's UK electricity generation mix (fuel type %)."""
    url = "https://api.carbonintensity.org.uk/generation"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    timestamp = pd.to_datetime(data["data"]["from"], utc=True)
    gen_mix = data["data"]["generationmix"]

    mix_dict = {"datetime": timestamp}
    for item in gen_mix:
        fuel = item["fuel"].lower().replace(" ", "_")
        if fuel in ["biomass", "imports", "gas", "nuclear", "solar", "wind"]:
            mix_dict[f"uk_gen_{fuel}_%"] = item["perc"]

    return pd.DataFrame([mix_dict])


# ---------- 5️⃣ OCTOPUS ENERGY PRICES ----------
def fetch_octopus_prices():
    """Fetch retail electricity prices (Agile Octopus)."""
    products_url = "https://api.octopus.energy/v1/products/"
    response = requests.get(products_url)
    response.raise_for_status()

    products_data = response.json()
    agile_products = [p for p in products_data.get("results", []) if "AGILE" in p["code"]]
    if not agile_products:
        raise ValueError("No Agile tariffs found")

    latest_agile = agile_products[0]
    product_code = latest_agile["code"]

    tariff_code = None
    for link in latest_agile.get("links", []):
        if "electricity-tariffs" in link.get("href", ""):
            tariff_code = link["href"].split("/")[-2]
            break
    if not tariff_code:
        tariff_code = f"E-1R-{product_code}-A"

    rates_url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
    response = requests.get(rates_url)
    response.raise_for_status()
    data = response.json().get("results", [])

    df_prices = pd.DataFrame(data)
    df_prices["datetime"] = pd.to_datetime(df_prices["valid_from"], utc=True)
    df_prices["retail_price_£_per_kWh"] = df_prices["value_inc_vat"] / 100
    df_prices = df_prices[df_prices["datetime"].dt.date == (datetime.now(timezone.utc).date() - timedelta(days=1))]
    return df_prices[["datetime", "retail_price_£_per_kWh"]]


# ---------- 6️⃣ MERGE ALL SOURCES ----------
def merge_all_sources(weather_df, aqi_df, carbon_df, carbon_gen_df, prices_df):
    """Merge all data sources on datetime."""
    merged = weather_df.merge(aqi_df, on="datetime", how="outer")
    merged = merged.merge(carbon_df, on="datetime", how="outer")
    merged = merged.merge(prices_df, on="datetime", how="outer")

    for col in carbon_gen_df.columns:
        if col != "datetime":
            merged[col] = carbon_gen_df[col].iloc[0]

    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged


# ---------- 7️⃣ APPEND TO EXISTING FILE ----------
def append_to_historical(new_data, save_dir="data", file_name="uk_energy_data.csv"):
    """Append new data to existing historical CSV (dedup by datetime)."""
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"], utc=True)

        pd.set_option('future.no_silent_downcasting', True)  # log  

        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
        new_data = new_data[new_data["datetime"].dt.date == yesterday]

        combined = pd.concat([existing_df, new_data], ignore_index=True)

        combined = (
            combined.sort_values(["datetime"])
            .groupby("datetime", as_index=False, group_keys=False)
            .apply(lambda g: g.ffill().bfill().iloc[-1])
            .reset_index(drop=True)
        )
        combined = combined.drop_duplicates(subset=["datetime"], keep="first")


        combined = combined.infer_objects(copy=False)
        combined = combined.sort_values("datetime").reset_index(drop=True)

        combined.to_csv(save_path, index=False)
        print(f"✅ Appended data to {save_path}")
        return combined
    else:
        new_data.to_csv(save_path, index=False)
        print(f"🆕 Created new file: {save_path}")
        return new_data


# ---------- 8️⃣ DAILY COLLECTION ----------
def collect_and_append_yesterday(save_dir="data", file_name="uk_energy_data.csv"):
    """Fetch yesterday’s data and append to historical CSV."""
    try:
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        print(f"\n{'='*55}")
        print(f"⚡ Collecting Data for {yesterday}")
        print(f"{'='*55}\n")

        weather_df = fetch_weather_data()
        aqi_df = fetch_air_quality()
        carbon_df = fetch_carbon_intensity()
        carbon_gen_df = fetch_carbon_generation_mix()
        prices_df = fetch_octopus_prices()

        merged_df = merge_all_sources(weather_df, aqi_df, carbon_df, carbon_gen_df, prices_df)

        # 🧹 Clean duplicates and only keep full-hour data
        merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], utc=True)
        merged_df = merged_df[merged_df["datetime"].dt.minute == 0]
        merged_df = merged_df.drop_duplicates(subset=["datetime"], keep="first")
        merged_df = merged_df.sort_values("datetime").reset_index(drop=True)

        final_df = append_to_historical(merged_df, save_dir, file_name)

        print(f"\n✅ Collection Complete for {yesterday}")
        print(f"📅 Range in file: {final_df['datetime'].min()} → {final_df['datetime'].max()}")
        print(f"{'='*55}\n")

        return final_df

    except Exception as e:
        print(f"❌ Error during daily data collection: {e}")
        raise


if __name__ == "__main__":
    collect_and_append_yesterday()
