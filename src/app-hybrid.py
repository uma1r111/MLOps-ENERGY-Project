"""
Hybrid FastAPI RAG API
- LangChain + FastEmbed + Google Gemini
- Dual token tracking: Prometheus + LangSmith
- Guardrails enabled
"""
import sys
import time
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import (
    INDEX_PATH, GOOGLE_API_KEY, GEMINI_MODEL, FASTEMBED_MODEL,
    TOP_K, TEMPERATURE, MAX_TOKENS, LANGSMITH_API_KEY, LANGSMITH_PROJECT
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Globals
rag_chain = None
guardrail_engine = None

# Prometheus metrics
def clear_metrics():
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

clear_metrics()

query_counter = Counter('rag_queries_total', 'Total RAG queries', ['status'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency')
retrieval_counter = Counter('rag_retrievals_total', 'Total retrievals')
guardrail_violations = Counter('guardrail_violations_total', 'Guardrail violations', ['violation_type', 'stage'])
token_usage = Counter('llm_tokens_total', 'Total tokens', ['type'])
estimated_cost = Counter('llm_cost_dollars_total', 'Estimated cost in USD')
active_queries = Gauge('rag_active_queries', 'Active queries')
retrieval_quality = Histogram('retrieval_quality_score', 'Retrieval scores')

# Gemini 2.0 Flash pricing per 1M tokens
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

def estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough estimate: 4 chars per token

def load_rag_system():
    """Load retriever, RAG chain, and guardrails"""
    global rag_chain, guardrail_engine
    logger.info("Loading RAG system...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    retriever = create_retriever(INDEX_PATH, FASTEMBED_MODEL, TOP_K)
    rag_chain = create_rag_chain(
        retriever=retriever,
        llm_model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        conversational=False
    )

    try:
        guardrail_engine = GuardrailEngine("config/guardrails/guardrails_config.json")
        logger.info("✓ Guardrails initialized")
    except Exception:
        guardrail_engine = None

    logger.info("✓ RAG system ready")
    logger.info(f"LangSmith: {'✓ Active' if LANGSMITH_API_KEY else '✗ Disabled'}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_rag_system()
    yield
    logger.info("Shutting down...")

# FastAPI init
app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "How can households reduce energy consumption?",
            "top_k": 3,
            "include_sources": True
        }
    })
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10)
    include_sources: bool = True

class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    retrieval_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None
    latency: float
    model: str
    embedding_model: str
    tokens_used: Optional[dict] = None
    estimated_cost: Optional[float] = None
    langsmith_trace: Optional[str] = None

# Routes
@app.get("/")
async def root():
    return {
        "service": "Hybrid RAG API",
        "version": "1.0.0",
        "langsmith_enabled": LANGSMITH_API_KEY is not None,
        "guardrails_enabled": guardrail_engine is not None,
        "embedding_model": FASTEMBED_MODEL,
        "llm_model": GEMINI_MODEL
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "guardrails_enabled": guardrail_engine is not None,
        "langsmith_enabled": LANGSMITH_API_KEY is not None,
        "embedding_model": FASTEMBED_MODEL,
        "llm_model": GEMINI_MODEL
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest):
    if rag_chain is None:
        query_counter.labels(status='error').inc()
        raise HTTPException(503, "RAG system not initialized")

    active_queries.inc()
    start_time = time.time()
    sanitized_query = query.question

    try:
        # Input validation
        if guardrail_engine:
            validation = guardrail_engine.validate_input(query.question)
            if not validation['passed']:
                violations = validation.get('violations', [])
                for v in violations:
                    guardrail_violations.labels(
                        violation_type=v.get('type', 'unknown'),
                        stage='input'
                    ).inc()
                query_counter.labels(status='guardrail_rejected').inc()
                raise HTTPException(400, f"Query rejected by guardrails: {violations}")
            sanitized_query = validation.get('sanitized_input', query.question)

        # Invoke RAG chain (LangSmith callback tracks automatically if enabled)
        result = rag_chain.invoke(sanitized_query)
        retrieval_counter.inc()

        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        # Output moderation
        if guardrail_engine:
            moderation = guardrail_engine.moderate_output(answer)
            if not moderation['passed']:
                violations = moderation.get('violations', [])
                blocking = [v for v in violations if v.get('severity') != 'WARNING']
                if blocking:
                    answer = "Response blocked due to content safety."

        # Track retrieval quality
        for doc in source_docs:
            score = doc.metadata.get('retrieval_score', 0)
            retrieval_quality.observe(score)

        # Dual token tracking
        input_tokens = estimate_tokens(sanitized_query)
        output_tokens = estimate_tokens(answer)

        # Prometheus metrics
        token_usage.labels(type='input').inc(input_tokens)
        token_usage.labels(type='output').inc(output_tokens)
        cost = (input_tokens * COST_PER_INPUT_TOKEN +
                output_tokens * COST_PER_OUTPUT_TOKEN)
        estimated_cost.inc(cost)

        latency = time.time() - start_time
        query_counter.labels(status='success').inc()
        query_latency.observe(latency)

        # Format sources
        sources = None
        if query.include_sources and source_docs:
            sources = [
                SourceDocument(
                    content=doc.page_content[:300],
                    source=doc.metadata.get('source', 'unknown'),
                    page=doc.metadata.get('page'),
                    chunk_id=doc.metadata.get('chunk_id'),
                    retrieval_score=doc.metadata.get('retrieval_score')
                )
                for doc in source_docs
            ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(latency, 3),
            model=GEMINI_MODEL,
            embedding_model=FASTEMBED_MODEL,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            estimated_cost=round(cost, 6),
            langsmith_trace=f"Check LangSmith dashboard: {LANGSMITH_PROJECT}" if LANGSMITH_API_KEY else None
        )

    except HTTPException:
        raise
    except Exception as e:
        query_counter.labels(status='error').inc()
        raise HTTPException(500, str(e))
    finally:
        active_queries.dec()

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
