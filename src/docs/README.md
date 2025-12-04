# Energy RAG System 🚀

**Retrieval-Augmented Generation pipeline using FastEmbed (local embeddings) and Google Gemini 2.0 Flash**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


\
## ✨ Features

- 🚀 **Fast & Local**: FastEmbed for embeddings (no API key, runs locally)
- 🤖 **Powerful LLM**: Google Gemini 2.0 Flash (free tier available)
- 📚 **Multi-format**: Support for PDF and TXT documents
- 🔍 **FAISS Search**: Ultra-fast vector similarity search
- 🌐 **REST API**: Production-ready FastAPI server
- 📊 **Monitoring**: Built-in Prometheus metrics
- 🔄 **Reproducible**: Complete Makefile for end-to-end automation
- 📦 **Modular**: Clean, maintainable code structure

---

## 🏗️ Architecture

### System Architecture

See detailed architecture diagrams in [`docs/architecture.md`](docs/rag-architecture.md)

**Key Components:**
1. **Ingestion Pipeline**: Document loading → Chunking → Embedding → Indexing
2. **Vector Store**: FAISS for fast similarity search
3. **RAG Pipeline**: Query embedding → Retrieval → Context generation
4. **API Layer**: FastAPI with health checks and metrics
5. **LLM**: Google Gemini for answer generation

**Technology Stack:**
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5) - Local, 384 dimensions
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **LLM**: Google Gemini 2.0 Flash - Free tier (15 req/min)
- **API**: FastAPI + Pydantic v2
- **Monitoring**: Prometheus + Structured Logging

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google AI Studio API key ([Get it here](https://makersuite.google.com/app/apikey))

### One-Command Setup
```bash
# Complete setup and run
make rag
```

This single command will:
1. ✅ Create necessary directories
2. ✅ Install dependencies
3. ✅ Check environment variables
4. ✅ Run document ingestion
5. ✅ Start the API server

### Manual Setup
```bash
# 1. Clone repository
git clone <your-repo-url>
cd energy-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
make install

# 4. Setup project
make setup

# 5. Configure environment
# Edit .env and add your GOOGLE_API_KEY
nano .env

# 6. Add documents
# Place PDF/TXT files in data/documents/

# 7. Run ingestion
make ingest

# 8. Start API
make api
```

---

## 📖 Usage

### 1. Add Documents
```bash
# Add your documents to the data directory
cp your_documents/*.pdf data/documents/
cp your_documents/*.txt data/documents/
```

### 2. Ingest Documents
```bash
make ingest
```

**Output:**
```
============================================================
Starting Document Ingestion Pipeline
FastEmbed (local) + Google Gemini
============================================================
Found 5 files in data/documents
✓ Loaded 123 document chunks
✓ Generated embeddings: shape=(123, 384)
✓ FAISS index created: 123 vectors, 384 dimensions
✓ Index and documents saved to data/faiss_index
============================================================
Ingestion Pipeline Completed Successfully!
============================================================
```

### 3. Start API Server
```bash
make api
```

Server will start at: `http://localhost:8000`

### 4. Test the System
```bash
# In another terminal
make test-quick

# Or use the test script
python scripts/test_rag.py
```

### 5. Query via API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is solar energy efficiency?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "answer": "Solar energy efficiency refers to...",
  "sources": [
    {
      "content": "Solar panels convert...",
      "source": "solar_energy.pdf",
      "page": 5
    }
  ],
  "latency": 1.23,
  "tokens_used": 450,
  "model": "gemini-2.0-flash",
  "embedding_model": "BAAI/bge-small-en-v1.5 (local)"
}
```

---

## 🔌 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with system info |
| `/health` | GET | Health check |
| `/query` | POST | RAG query endpoint |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API docs (Swagger) |

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## ⚙️ Configuration

### Environment Variables

Edit `.env` file:
```bash
# Google AI Studio API Key
GOOGLE_API_KEY=your_key_here

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
TEMPERATURE=0.7
MAX_TOKENS=2048
```

### Available Models

**FastEmbed Models** (Local, No API Key):
- `BAAI/bge-small-en-v1.5` (fast, 384 dim) ⭐ Recommended
- `BAAI/bge-base-en-v1.5` (balanced, 768 dim)
- `sentence-transformers/all-MiniLM-L6-v2` (fast, 384 dim)

**Gemini Models**:
- `gemini-2.0-flash-exp` (newest, experimental)
- `gemini-2.0-flash` (stable, fast) ⭐ Recommended
- `gemini-1.5-flash` (older, stable)
- `gemini-1.5-pro` (more capable, slower)

---

## 📊 Monitoring

### View Metrics
```bash
# View Prometheus metrics
make metrics

# Or visit
curl http://localhost:8000/metrics
```

### Available Metrics

- `rag_queries_total` - Total queries by status
- `rag_query_latency_seconds` - Query latency histogram
- `rag_tokens_used_total` - Total tokens used
- `rag_retrievals_total` - Total document retrievals


## 🛠️ Development

### Makefile Commands
```bash
make help          # Show all available commands
make install       # Install dependencies
make setup         # Setup project structure
make ingest        # Run document ingestion
make api           # Start API server
make rag           # Run complete pipeline
make test          # Run tests
make lint          # Run linters
make format        # Format code
make clean         # Clean generated files
make docs          # Generate documentation
```

