from google.cloud import aiplatform
import os

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

# Test query
print("\nSending test query...")
response = endpoint.predict(instances=[{
    "question": "how to reduce carbon footprints?",
    "top_k": 3,
    "include_sources": True
}])

result = response.predictions[0]
print(f"\n✓ Success!")
print(f"Answer: {result['answer'][:200]}...")
print(f"Latency: {result['latency']}s")
print(f"Variant: {result['variant_name']}")
