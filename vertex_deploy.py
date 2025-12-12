from google.cloud import aiplatform
import os
import sys
from datetime import datetime

# Get configuration
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

if not PROJECT_ID or not GOOGLE_API_KEY:
    print("ERROR: PROJECT_ID or GOOGLE_API_KEY not set!")
    sys.exit(1)

# Build image URI
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/mlops-energy-repo/energy-forecast-rag:v1.2"
BUCKET = f"gs://mlops-energy-{PROJECT_ID}"

print(f"""
{'='*70}
Starting Vertex AI Deployment
{'='*70}
Project: {PROJECT_ID}
Region: {REGION}
Image: {IMAGE_URI}
{'='*70}
""")

# Initialize
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

# Upload model
print("\n[1/3] Uploading model...")
model = aiplatform.Model.upload(
    display_name=f"energy-rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    serving_container_image_uri=IMAGE_URI,
    serving_container_predict_route="/query",
    serving_container_health_route="/health",
    serving_container_ports=[8080],
    serving_container_environment_variables={
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "LANGSMITH_API_KEY": LANGSMITH_API_KEY,
        "PORT": "8080"
    }
)
print(f"? Model uploaded: {model.name}")

# Create endpoint
print("\n[2/3] Creating endpoint...")
endpoints = aiplatform.Endpoint.list(filter='display_name="energy-forecast-endpoint"')
if endpoints:
    endpoint = endpoints[0]
    print(f"? Using existing endpoint: {endpoint.name}")
else:
    endpoint = aiplatform.Endpoint.create(display_name="energy-forecast-endpoint")
    print(f"? Created endpoint: {endpoint.name}")

# Deploy
print("\n[3/3] Deploying model (this takes 10-15 minutes)...")
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="energy-rag-deployment",
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    sync=True
)

# Save info
with open("vertex_endpoint_info.txt", "w") as f:
    f.write(f"ENDPOINT_ID={endpoint.name}\n")
    f.write(f"PROJECT_ID={PROJECT_ID}\n")
    f.write(f"REGION={REGION}\n")

print(f"""
{'='*70}
? DEPLOYMENT COMPLETE!
{'='*70}
Endpoint ID: {endpoint.name}
Saved to: vertex_endpoint_info.txt
{'='*70}
""")
