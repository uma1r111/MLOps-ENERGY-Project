# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency file
COPY requirements.txt .

# Install dependencies into /install directory
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --target=/install -r requirements.txt


# -------- Stage 2: Runtime --------
FROM python:3.11-slim

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/opt/python-deps"

# Create a non-root user
RUN useradd -m appuser

# Create and set working directory
WORKDIR /app

# Install minimal runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /install /opt/python-deps

# Copy your project files
COPY . .
RUN chmod +x /app/healthcheck.sh

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Expose app port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["/bin/bash", "/app/healthcheck.sh"]

# Default command (if using Flask or FastAPI)
CMD ["python", "app.py"]
