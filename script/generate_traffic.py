"""
Generate traffic to populate monitoring dashboards
"""

import requests
import time
import random

API_URL = "http://localhost:8000"

# Sample questions
QUESTIONS = [
    "How can households reduce energy consumption?",
    "What are the benefits of solar panels?",
    "How does energy efficiency affect costs?",
    "What is the impact of insulation on energy bills?",
    "How do smart thermostats save energy?",
    "What are renewable energy sources?",
    "How can I reduce my carbon footprint?",
    "What is net metering?",
    "How efficient are LED bulbs?",
    "What are peak electricity hours?",
    "How does weather affect energy consumption?",
    "What is an energy audit?",
    "How do heat pumps work?",
    "What are the best energy saving tips?",
    "How to calculate energy savings?",
]


def send_query(question: str) -> dict:
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": 3, "include_sources": True},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def main():
    print("=" * 60)
    print("Generating Traffic for Monitoring Dashboards")
    print("=" * 60)

    # Check health
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"✓ API Status: {health['status']}")
    except Exception:
        print("❌ API not available. Start it with: python src/app.py")
        return

    print(f"\n🚀 Sending {len(QUESTIONS)} queries...")
    print("This will generate metrics for Prometheus & Grafana\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question[:50]}...")

        result = send_query(question)

        if result:
            print(f"  ✓ Latency: {result['latency']:.2f}s")
            if result.get("tokens_used"):
                tokens = result["tokens_used"]
                print(
                    f"  ✓ Tokens: {tokens['total']} (${result['estimated_cost']:.6f})"
                )

        # Random delay between queries
        time.sleep(random.uniform(1, 3))

    print("\n" + "=" * 60)
    print("✅ Traffic generation complete!")
    print("=" * 60)
    print("\nCheck your dashboards:")
    print("  🔹 Prometheus: http://localhost:9090")
    print("  🔹 Grafana:    http://localhost:3000")
    print("  🔹 Metrics:    http://localhost:8000/metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
