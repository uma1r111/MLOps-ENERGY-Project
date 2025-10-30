from flask import Flask, Response
import json
import os

app = Flask(__name__)

# Path to your metrics.json (update if needed)
METRICS_FILE = ('../monitoring/metrics.json')

# Optional: human-friendly descriptions for your metrics
METRIC_DESCRIPTIONS = {
    "api_success_rate": "API call success rate (1 = success)",
    "avg_api_response_time": "Avg API response time in seconds",
    "rmse": "Root Mean Squared Error of predictions",
    "mae": "Mean Absolute Error of predictions"
}

@app.route('/metrics')
def metrics():
    if not os.path.exists(METRICS_FILE):
        return Response("# metrics.json not found\n", status=404, mimetype='text/plain')

    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)

    lines = []
    for key, value in data.items():
        # Clean metric name to be Prometheus compatible (lowercase, underscores)
        metric_name = key.lower()
        desc = METRIC_DESCRIPTIONS.get(key, "No description available")
        lines.append(f"{desc}")
        lines.append(f"{metric_name} {value}")
        lines.append("")  # blank line between metrics

    output = "\n".join(lines)
    return Response(output, mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8025)