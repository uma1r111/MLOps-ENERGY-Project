FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8002
HEALTHCHECK CMD curl --fail http://localhost:8002/health || exit 1
CMD ["python", "app.py"]
