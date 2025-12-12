from google.cloud import aiplatform
import os
import requests
import json

# Load endpoint info
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")

with open("vertex_endpoint_info.txt") as f:
    for line in f:
        if line.startswith("ENDPOINT_ID="):
            ENDPOINT_ID = line.strip().split("=")[1]

print(f"Testing endpoint: {ENDPOINT_ID}")

# Initialize
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

# Get the prediction URL
predict_url = f"https://{REGION}-aiplatform.googleapis.com/v1/{ENDPOINT_ID}:rawPredict"

print(f"Sending test query to: {predict_url}")

# Get access token
import subprocess
token_result = subprocess.run(
    ["gcloud", "auth", "print-access-token"],
    capture_output=True,
    text=True
)
access_token = token_result.stdout.strip()

# Make direct HTTP request
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Send request directly to FastAPI format
data = {
    "question": "What is renewable energy?",
    "top_k": 3,
    "include_sources": True
}

response = requests.post(predict_url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Success!")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Latency: {result['latency']}s")
    print(f"Variant: {result['variant_name']}")
    print(f"Tokens: {result['tokens_used']}")
    print(f"Cost: ")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
