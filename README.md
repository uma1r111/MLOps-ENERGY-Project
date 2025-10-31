# MLOps Energy Forecasting System - Milestone 1🔋

UK energy demand forecasting system with real-time monitoring and automated ML pipeline deployment.

## Architecture Diagram

*MLOps Pipeline Architecture*

<img src="images/architecture.png" alt="MLOps Pipeline Architecture" width="750" />

The pipeline consists of six main stages:

1. **Data Collection**: Using OpenMateo API with GitHub Actions for automation
2. **Feature Engineering**: Utilizing Feature Tools and Data Quality checks
3. **Feature Selection**: XGBoost regressor, Random Forest, PCA, and ACF/PACF analysis
4. **Model Training**: Multiple models including LSTM/GRU, Temporal CNN, SARIMAX, and Transformer
5. **Model Save**: BentoML archiving and S3 storage with MLflow tracking
6. **Model Forecasting**: Final predictions

Each stage is integrated with AWS S3 for artifact storage and version control, while DVC manages data versioning.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/uma1r111/MLOps-ENERGY-Project.git
cd MLOps-ENERGY-Project

# Setup development environment
make dev
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make dev` | Set up development environment, install dependencies |
| `make test` | Run unit tests with pytest |
| `make docker` | Build BentoML service Docker image |
| `make serve` | Serve BentoML model locally |
| `make bentoml-build` | Build model as deployable Bento |
| `make monitor` | Start monitoring stack (Prometheus + Grafana) |
| `make lint` | Run code quality checks (ruff, black) |
| `make clean` | Clean build artifacts and cache |

## ML Workflow Monitoring

### MLflow Tracking
MLflow tracking server: `http://localhost:8000`
- Latest model version: `v1.0.0` (registered in MLflow)
- Model Registry URI: `models:/energy_forecast/production`

![MLflow Experiments](images/mlflow_dashboard.png)

### Data Drift Monitoring
Evidently AI Dashboard for data drift monitoring: `http://localhost:7000`

![Data Drift Dashboard](images/evidently_dashboard.jpg)

## Cloud Deployment

### AWS Services Integration

Our ML pipeline utilizes several core AWS services to store artifacts, run model serving, and handle event-driven tasks:

1. **AWS S3 for Data and Model Storage**
   ![S3 Bucket Structure](images/S3_storage.png)
   - Stores training data and model artifacts
   - Enables versioned storage for reproducibility
   - Facilitates team collaboration

2. **AWS EC2 for Model Serving**
   ![EC2 Instances](images/EC2_instance.png)
   - Hosts the inference API (BentoML) for high-throughput, persistent serving
   - Auto-scaling group for handling load variations
   - Continuous monitoring via CloudWatch

3. **AWS Lambda for Serverless Tasks**
   ![AWS Lambda](images/Lambda.png)
   - Handles event-driven jobs (e.g., scheduled data pulls, lightweight feature transforms, asynchronous post-processing)
   - Integrates with S3 events and CloudWatch Events (EventBridge)
   - Useful for cost-efficient, short-running tasks and glue logic between services

If you need me to add sample deployment snippets (CloudFormation/Terraform) or the actual Lambda function code into the repo, tell me where you'd like those files placed (e.g., `infra/` or `scripts/`).
### Service Architecture

```mermaid
graph TB
    subgraph VPC
        subgraph Public Subnet
            ALB[Application Load Balancer]
            EC2[EC2 Instance with BentoML Service]
        end
        
        subgraph Private Subnet
            MLflow[MLflow Server]
        end
    end
    
    subgraph Storage
        S3[(S3 Buckets)]
        S3_Data[(Training Data)]
        S3_Models[(Model Artifacts)]
    end
    
    subgraph Monitoring
        CW[CloudWatch]
        Prom[Prometheus]
        Graf[Grafana]
    end
    
    Client-->ALB
    ALB-->EC2
    EC2-->S3
    EC2-->MLflow
    EC2-->CW
    MLflow-->S3_Models
    EC2-->Prom
    Prom-->Graf
```

## Model Serving & API Documentation

BentoML Service endpoint available at: `http://localhost:3000`

![BentoML Service UI](images/bentoml_service.png)

Service includes:
- Automatic OpenAPI documentation
- Built-in model monitoring
- Request logging and metrics
- Docker/Kubernetes deployment ready

Example usage:
```bash
# Using bentoml CLI
bentoml serve service:svc --production

# API request
curl -X POST "http://localhost:3000/predict" \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2025-10-31T12:00:00Z"}'
```

### Model Deployment
```bash
# Build model as deployable Bento
bentoml build

# Containerize the Bento
bentoml containerize energy_forecast:latest
```

The service is automatically packaged with:
- Model artifacts
- Python dependencies
- API configuration
- Environment settings

## Bonus Features Implemented

### Data Version Control (DVC)
- DVC integrated for dataset versioning
- Tracking multiple data files:
  - `uk_energy_data.csv`
  - `engineered_features.csv`
  - `selected_features.csv`
  - `predictions.csv`
- Enables reproducible data pipeline
- Efficient large file handling with S3 backend

### Docker Compose

Our project leverages Docker Compose with environment profiles for efficient microservices orchestration:

#### Microservices Architecture

| Service | Port | Description |
|---------|------|-------------|
| Data Preprocessing | 8004 | Cleans, transforms, and prepares raw data for training |
| Model Training | 8000 | Trains and evaluates ML models on processed data |
| Prediction Client | 8003 | Provides a prediction API for serving trained models |
| Monitoring Service | 8002 | Displays system and model health/status information |

Each service has its own Dockerfile and includes a health endpoint for monitoring.

#### Docker Compose Profiles

The project uses Docker Compose v3.9 with profiles for managing multiple environments:

| Profile | Includes | Purpose |
|---------|----------|----------|
| dev | All services | For local development and debugging |
| test | All services | For integration and test automation |
| prod | All services | For deployment-ready, production mode |
| app | Data Preprocessing, Model Training, Prediction Client | For app-specific tasks |
| monitoring | Monitoring only | For isolated health/status checks |

#### Usage Commands

```bash
# Build and run all services
docker compose --profile dev up --build

# Run only application services
docker compose --profile app up --build

# Run monitoring service only
docker compose --profile monitoring up --build

# Run tests
docker compose --profile test up

# Run production setup
docker compose --profile prod up -d
```

#### Service Access Points

After running the containers, access services at:
- Data Preprocessing: `http://localhost:8004`
- Model Training: `http://localhost:8000`
- Prediction Client: `http://localhost:8003`
- Monitoring: `http://localhost:8002`

Each service exposes a `/health` endpoint returning:
```json
{
  "status": "healthy",
  "service": "service_name"
}
```

#### Cleanup

```bash
# Stop and remove all containers
docker compose down

# Remove unused images and networks
docker system prune -f
```

## FAQ

### Common Build Issues

1. **Docker build fails**
   - Increase Docker memory limit (recommended: 4GB)
   - Run `docker system prune` to clear space
   - Check Docker daemon is running

2. **Package installation errors**
   - Update pip: `python -m pip install --upgrade pip`
   - Install build tools: `apt-get install build-essential` (Linux)
   - Use Python 3.11 as specified in Dockerfile

### Platform Setup

#### Windows
1. Install WSL2: `wsl --install`
2. Install Docker Desktop with WSL2 backend
3. Install Make: `choco install make`
4. Install Python 3.11

#### MacOS
1. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install dependencies: `brew install make docker python@3.11`
3. Start Docker Desktop
4. Run: `brew link python@3.11`

