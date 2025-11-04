# ============================
# 1️⃣ Builder Stage - Install Dependencies
# ============================
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy requirements and install inside a temporary directory to reduce layers
COPY requirements.txt .

RUN pip install --no-cache-dir --target=/install -r requirements.txt


# ============================
# 2️⃣ Final Stage - Runtime Image
# ============================
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy the monitoring service source code
COPY . .

# Add a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port used by the Flask app
EXPOSE 7000

# Healthcheck for Docker status monitoring
HEALTHCHECK CMD curl --fail http://localhost:7000/ || exit 1

# Default command — run the Evidently monitoring dashboard
CMD ["python", "data_drift_monitor.py"]
