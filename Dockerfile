FROM python:3.11-slim

WORKDIR /app

# Add cache busting ARG to force rebuild when needed
ARG CACHEBUST=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & setuptools to fix vulnerabilities
RUN python -m pip install --upgrade pip setuptools

# Copy requirements and install Python dependencies
COPY requirements-rag.txt .
RUN pip install --no-cache-dir -r requirements-rag.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY experiments/ ./experiments/

# Expose port
EXPOSE 8000

# Health check (increase retries & timeout for canary)
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=10 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
