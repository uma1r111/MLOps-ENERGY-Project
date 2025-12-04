# Makefile for RAG Pipeline
# Energy RAG System - FastEmbed + Google Gemini

.PHONY: help install setup ingest api test rag clean lint format docker

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
VENV := venv
SRC_DIR := src
DATA_DIR := data
DOCS_DIR := data/documents
INDEX_DIR := data/faiss_index

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo "$(BLUE)RAG Pipeline - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: ## Complete setup (create dirs, check env)
	@echo "$(BLUE)Setting up project...$(NC)"
	@mkdir -p $(DOCS_DIR)
	@mkdir -p $(INDEX_DIR)
	@mkdir -p logs
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠ Creating .env from .env.example$(NC)"; \
		cp .env.example .env; \
		echo "$(RED)⚠ Please update GOOGLE_API_KEY in .env$(NC)"; \
	fi
	@echo "$(GREEN)✓ Project setup complete$(NC)"
	@echo "$(YELLOW)📁 Data directories created:$(NC)"
	@echo "  - $(DOCS_DIR) (add your PDFs/TXT files here)"
	@echo "  - $(INDEX_DIR) (FAISS index will be stored here)"

env-check: ## Check if .env file exists and has required variables
	@echo "$(BLUE)Checking environment variables...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(RED)✗ .env file not found$(NC)"; \
		exit 1; \
	fi
	@if ! grep -q "GOOGLE_API_KEY=" .env || grep -q "GOOGLE_API_KEY=your" .env; then \
		echo "$(RED)✗ GOOGLE_API_KEY not set in .env$(NC)"; \
		echo "$(YELLOW)Get your API key from: https://makersuite.google.com/app/apikey$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Environment variables OK$(NC)"

##@ RAG Pipeline

ingest: env-check ## Run document ingestion pipeline
	@echo "$(BLUE)Running ingestion pipeline...$(NC)"
	@if [ -z "$$(ls -A $(DOCS_DIR) 2>/dev/null)" ]; then \
		echo "$(RED)✗ No documents found in $(DOCS_DIR)$(NC)"; \
		echo "$(YELLOW)Please add PDF or TXT files to $(DOCS_DIR)$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SRC_DIR)/rag/ingest.py
	@echo "$(GREEN)✓ Ingestion complete$(NC)"

api: env-check ## Start the FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	@if [ ! -f $(INDEX_DIR)/index.faiss ]; then \
		echo "$(RED)✗ FAISS index not found$(NC)"; \
		echo "$(YELLOW)Run 'make ingest' first$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) $(SRC_DIR)/app.py

test-api: ## Test the API with sample queries
	@echo "$(BLUE)Testing RAG API...$(NC)"
	@sleep 2  # Wait for server to start
	$(PYTHON) scripts/test_rag.py

rag: setup install ingest api ## 🚀 Run complete RAG pipeline end-to-end
	@echo "$(GREEN)✓ RAG pipeline completed successfully!$(NC)"

##@ Testing

test: ## Run pytest tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-quick: ## Run quick API test (server must be running)
	@echo "$(BLUE)Testing API endpoints...$(NC)"
	curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"question": "What is energy efficiency?"}' | python -m json.tool

##@ Code Quality

lint: ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 $(SRC_DIR) --max-line-length=120
	pylint $(SRC_DIR) --max-line-length=120
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(SRC_DIR)
	isort $(SRC_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

##@ Monitoring

metrics: ## View Prometheus metrics
	@echo "$(BLUE)Fetching metrics...$(NC)"
	curl -s http://localhost:8000/metrics

logs: ## Tail application logs
	@echo "$(BLUE)Tailing logs...$(NC)"
	tail -f logs/app.log

##@ Cleanup

clean: ## Clean generated files and cache
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf $(INDEX_DIR)/*
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(SRC_DIR)/rag/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Clean everything including venv
	@echo "$(YELLOW)⚠ This will delete the virtual environment$(NC)"
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t energy-rag:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p 8000:8000 --env-file .env -v $(PWD)/data:/app/data energy-rag:latest

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop $$(docker ps -q --filter ancestor=energy-rag:latest)

##@ Documentation

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Architecture diagram: docs/architecture.md$(NC)"
	@echo "$(YELLOW)API docs: http://localhost:8000/docs$(NC)"

info: ## Show project information
	@echo "$(BLUE)RAG System Information$(NC)"
	@echo "$(GREEN)Project Structure:$(NC)"
	@tree -L 2 -I 'venv|__pycache__|*.pyc'
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@grep -v "^#" .env 2>/dev/null || echo "No .env file"
	@echo ""
	@echo "$(GREEN)Index Status:$(NC)"
	@if [ -f $(INDEX_DIR)/index.faiss ]; then \
		echo "  ✓ FAISS index exists"; \
		$(PYTHON) -c "import faiss; idx=faiss.read_index('$(INDEX_DIR)/index.faiss'); print(f'  Vectors: {idx.ntotal}')"; \
	else \
		echo "  ✗ FAISS index not found"; \
	fi