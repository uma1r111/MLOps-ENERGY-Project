import pandas as pd
import numpy as np
from datetime import timedelta
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import os

# === CONFIG ===
DATA_PATH = "data/selected_features.csv"
PROMETHEUS_PUSHGATEWAY = ""
JOB_NAME = "data_drift_monitor"


def calculate_drift(reference_df, current_df, feature_cols):
    """
    Compute simple drift metrics between reference (monthly) and current (weekly) datasets.
    Returns a dict of drift scores per feature.
    """
    drift_results = {}

    for col in feature_cols:
        if col == "datetime":  # Skip datetime
            continue
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()

        if ref.empty or cur.empty:
            continue

        # Compute summary statistics
        mean_diff = abs(cur.mean() - ref.mean())
        std_diff = abs(cur.std() - ref.std())
        corr = np.corrcoef(
            ref[: min(len(ref), len(cur))], cur[: min(len(ref), len(cur))]
        )[0, 1]

        drift_results[col] = {
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "corr": corr,
        }

    return drift_results


def push_metrics_to_prometheus(drift_results, registry):
    """
    Push drift metrics to Prometheus Pushgateway.
    """
    for feature, metrics in drift_results.items():
        for metric_name, value in metrics.items():
            g = Gauge(
                f"data_drift_{feature}_{metric_name}",
                f"Drift metric {metric_name} for feature {feature}",
                registry=registry,
            )
            g.set(float(value))

    # Push metrics
    push_to_gateway(PROMETHEUS_PUSHGATEWAY, job=JOB_NAME, registry=registry)


def monitor_drift():
    """
    Main function to monitor drift between latest week and previous month.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Missing dataset: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    latest_date = df["datetime"].max()

    # --- Define windows ---
    current_start = latest_date - timedelta(days=7)
    reference_start = current_start - timedelta(days=30)
    reference_end = current_start - timedelta(days=1)

    reference_df = df[
        (df["datetime"] >= reference_start) & (df["datetime"] <= reference_end)
    ]
    current_df = df[df["datetime"] >= current_start]

    print(
        f"Reference window: {reference_start.date()} → {reference_end.date()} ({len(reference_df)} rows)"
    )
    print(
        f"Current window: {current_start.date()} → {latest_date.date()} ({len(current_df)} rows)"
    )

    if reference_df.empty or current_df.empty:
        print("Not enough data for comparison — skipping drift check.")
        return

    # --- Select numeric columns only ---
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- Calculate drift ---
    drift_results = calculate_drift(reference_df, current_df, feature_cols)

    # --- Log to console ---
    for feature, metrics in drift_results.items():
        print(
            f"{feature} → mean_diff={metrics['mean_diff']:.4f}, std_diff={metrics['std_diff']:.4f}, corr={metrics['corr']:.4f}"
        )

    # --- Push metrics to Prometheus ---
    registry = CollectorRegistry()
    push_metrics_to_prometheus(drift_results, registry)

    print("Drift metrics pushed to Prometheus Pushgateway.")


if __name__ == "__main__":
    monitor_drift()
