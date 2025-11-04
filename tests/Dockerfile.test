# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command to run tests with coverage ≥80%
CMD ["pytest", "--cov=.", "--cov-report=term-missing", "--cov-fail-under=80"]
