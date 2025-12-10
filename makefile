.PHONY: help setup ingest run-api run-ui monitoring test clean rag all

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
VENV := .venv
DOCKER_COMPOSE := docker-compose -f docker-compose.monitoring.yml

help:  ## Show this help message
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  MLOps Energy RAG Assistant - Make Commands"
	@echo "═══════════════════════════════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo "═══════════════════════════════════════════════════════════"

setup:  ## Install dependencies and setup environment
	@echo "Setting up environment..."
	$(PYTHON) -m venv $(VENV) || true
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete!"

ingest:  ## Run document ingestion pipeline (D2)
	@echo "Starting document ingestion..."
	$(PYTHON) src/rag/ingest.py
	@echo "Ingestion complete! FAISS index created."

run-api:  ## Start FastAPI server
	@echo "Starting FastAPI server..."
	$(PYTHON) src/app.py

run-ui:  ## Start Gradio UI
	@echo "Starting Gradio UI..."
	$(PYTHON) src/ui.py

monitoring:  ## Start monitoring stack (Prometheus + Grafana)
	@echo "starting monitoring stack..."
	$(DOCKER_COMPOSE) up -d
	@echo "Monitoring started!"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

monitoring-stop:  ## Stop monitoring stack
	@echo "Stopping monitoring stack..."
	$(DOCKER_COMPOSE) down
	@echo "Monitoring stopped!"

test:  ## Run complete test suite
	@echo "Running test suite..."
	$(PYTHON) scripts/test_rag.py
	@echo "Tests complete!"

test-quick:  ## Run quick test
	@echo "Running quick test..."
	$(PYTHON) scripts/test_rag.py --quick

test-simple:  ## Run simple test
	@echo "Running simple test..."
	$(PYTHON) scripts/simple_test.py

generate-traffic:  ## Generate A/B test traffic
	@echo "Generating A/B test traffic..."
	$(PYTHON) scripts/generate_ab_traffic.py
	@echo "Traffic generation complete!"

analyze-ab:  ## Run A/B test statistical analysis
	@echo "Running A/B test analysis..."
	$(PYTHON) src/monitoring/ab_test_analysis.py
	@echo "Analysis complete! Check monitoring/ab_test_plots.png"

evidently:  ## Generate Evidently data drift report
	@echo "Generating Evidently drift report..."
	$(PYTHON) src/monitoring/evidently_monitor.py
	@echo "Report generated! Open monitoring/evidently_report.html"

rag: setup ingest  ## Complete RAG pipeline (D2 requirement)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "RAG Pipeline Complete!"
	@echo "═══════════════════════════════════════════════════════════"
	@echo "Next steps:"
	@echo "  1. make run-api    # Start API server"
	@echo "  2. make monitoring # Start monitoring"
	@echo "  3. make test       # Run tests"
	@echo "═══════════════════════════════════════════════════════════"

all: setup ingest monitoring run-api  ## Setup everything and start services
	@echo "═══════════════════════════════════════════════════════════"
	@echo "All services started!"
	@echo "═══════════════════════════════════════════════════════════"
	@echo "Access points:"
	@echo "  🔹 API:        http://localhost:8000"
	@echo "  🔹 API Docs:   http://localhost:8000/docs"
	@echo "  🔹 Prometheus: http://localhost:9090"
	@echo "  🔹 Grafana:    http://localhost:3000"
	@echo "═══════════════════════════════════════════════════════════"

check-health:  ## Check if services are running
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo " API not responding"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus not responding"
	@curl -s http://localhost:3000/api/health || echo "Grafana not responding"

logs-api:  ## Show API logs (requires screen/tmux or separate process)
	@tail -f logs/app.log 2>/dev/null || echo "No logs found. API may not be running."

clean:  ## Clean generated files
	@echo "Cleaning generated files..."
	rm -rf data/faiss_index/*
	rm -rf monitoring/*.html
	rm -rf monitoring/*.csv
	rm -rf monitoring/*.jsonl
	rm -rf monitoring/*.png
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/**/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

clean-all: clean monitoring-stop  ## Clean everything including Docker volumes
	@echo "Deep cleaning..."
	$(DOCKER_COMPOSE) down -v
	rm -rf $(VENV)
	@echo "Deep cleanup complete!"

requirements:  ## Generate requirements.txt from environment
	$(PIP) freeze > requirements.txt
	@echo "requirements.txt updated!"

lint:  ## Run linters
	@echo "Running linters..."
	$(PYTHON) -m flake8 src/ --max-line-length=100 --ignore=E501,W503
	$(PYTHON) -m black src/ --check
	@echo "Linting complete!"

format:  ## Format code with black
	@echo "Formatting code..."
	$(PYTHON) -m black src/
	@echo " Formatting complete!"

docs:  ## Generate API documentation
	@echo "Generating documentation..."
	@echo "Open http://localhost:8000/docs after starting the API"

demo:  ## Run full demo (ingest + API + generate traffic + analyze)
	@echo "Running full demo..."
	@make ingest
	@make monitoring
	@echo "Waiting for monitoring to start..."
	@sleep 10
	@make run-api &
	@echo "Waiting for API to start..."
	@sleep 5
	@make generate-traffic
	@make analyze-ab
	@make evidently
	@echo "Demo complete! Check dashboards at http://localhost:3000"

docker-build:  ## Build Docker image
	@echo "Building Docker image..."
	docker build -t rag-api:latest .
	@echo "Docker image built!"

docker-run:  ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8000:8000 \
		-e GOOGLE_API_KEY=$(GOOGLE_API_KEY) \
		-e LANGSMITH_API_KEY=$(LANGSMITH_API_KEY) \
		rag-api:latest

status:  ## Show status of all components
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  System Status"
	@echo "═══════════════════════════════════════════════════════════"
	@echo "FAISS Index:"
	@if [ -d "data/faiss_index" ] && [ -f "data/faiss_index/index.faiss" ]; then \
		echo " Index exists"; \
		ls -lh data/faiss_index/index.faiss; \
	else \
		echo " Index not found (run: make ingest)"; \
	fi
	@echo ""
	@echo "Docker Containers:"
	@docker ps --filter "name=rag-" --format "table {{.Names}}\t{{.Status}}" || echo " Docker not running"
	@echo ""
	@echo "API Health:"
	@curl -s http://localhost:8000/health > /dev/null 2>&1 && echo " API running" || echo " API not running (run: make run-api)"
	@echo "═══════════════════════════════════════════════════════════"