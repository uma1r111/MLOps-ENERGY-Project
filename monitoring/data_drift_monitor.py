"""
monitoring/data_drift_monitor.py

Evidently (Report + DataDriftPreset) script compatible with evidently >=0.4 and <0.7.

Usage:
  1) Make sure to install a compatible version inside your container:
       pip install "evidently>=0.4,<0.7"
  2) Run this script to generate the report and start a local web server.
  3) Open http://localhost:8002 to view the Evidently dashboard.
"""

import os
import sys
from datetime import timedelta
import pandas as pd
from flask import Flask, send_file

# ✅ Evidently imports (works for versions >=0.4,<0.7)
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
except Exception as exc:
    print("❌ ERROR: Unable to import Evidently (expected API for evidently >=0.4,<0.7).")
    print("👉 Fix by installing a compatible version:")
    print('   pip install "evidently>=0.4,<0.7"')
    print(f"Import error: {exc}")
    sys.exit(1)

# === CONFIG ===
DATA_PATH = "data/selected_features.csv"
REPORT_PATH = "monitoring/evidently_drift_report.html"
PORT = 8002


def build_windows(df):
    """Split data into reference (30 days) and current (7 days) windows."""
    df["datetime"] = pd.to_datetime(df["datetime"])
    latest = df["datetime"].max()

    current_start = latest - timedelta(days=7)
    reference_start = current_start - timedelta(days=30)
    reference_end = current_start - timedelta(days=1)

    reference_df = df[(df["datetime"] >= reference_start) & (df["datetime"] <= reference_end)].copy()
    current_df = df[df["datetime"] >= current_start].copy()

    return reference_df, current_df, reference_start, reference_end, current_start, latest


def generate_report(reference_df, current_df, report_path=REPORT_PATH):
    """Generate Evidently Data Drift report."""
    column_mapping = ColumnMapping()
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.save_html(report_path)
    return report


def monitor_and_generate():
    """Load data, build windows, and generate the report."""
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset missing at {DATA_PATH}")
        return False

    df = pd.read_csv(DATA_PATH)
    reference_df, current_df, ref_start, ref_end, cur_start, latest = build_windows(df)

    print(f"📅 Reference window: {ref_start.date()} → {ref_end.date()} ({len(reference_df)} rows)")
    print(f"📅 Current window:   {cur_start.date()} → {latest.date()} ({len(current_df)} rows)")

    if reference_df.empty or current_df.empty:
        print("⚠️ Not enough data for comparison — skipping.")
        return False

    generate_report(reference_df, current_df)
    print(f"✅ Evidently report saved to: {REPORT_PATH}")
    return True


# === Flask web server to serve report ===
app = Flask(__name__)

@app.route("/")
def serve_dashboard():
    abs_report_path = os.path.abspath(REPORT_PATH)
    if not os.path.exists(abs_report_path):
        return (
            f"Report not found at {abs_report_path}. "
            "Run the generator first (python monitoring/data_drift_monitor.py)."
        ), 404
    return send_file(abs_report_path)


if __name__ == "__main__":
    ok = monitor_and_generate()
    if not ok:
        print("⚠️ Report generation failed or skipped; server will still serve old report if available.")
    print(f"🌐 Serving Evidently dashboard at http://localhost:{PORT}  (CTRL+C to stop)")
    app.run(host="0.0.0.0", port=PORT)
