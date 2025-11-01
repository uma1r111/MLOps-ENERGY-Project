# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --target=/install -r requirements.txt

# -------- Stage 2: Runtime --------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/opt/python-deps"

RUN useradd -m appuser
WORKDIR /app

COPY --from=builder /install /opt/python-deps
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8004
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "app.py", "health"]

CMD ["python", "app.py"]
