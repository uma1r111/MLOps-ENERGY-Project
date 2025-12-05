"""
Enhanced FastAPI RAG with Token & Cost Tracking
"""
import sys
import time
import logging
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import *
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_chain = None
guardrail_engine = None

def clear_metrics():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except:
            pass

clear_metrics()

# Enhanced Prometheus Metrics
query_counter = Counter('rag_queries_total', 'Total queries', ['status'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency')
retrieval_counter = Counter('rag_retrievals_total', 'Total retrievals')
guardrail_violations = Counter('guardrail_violations_total', 'Violations', ['violation_type', 'stage'])

# NEW: Token & Cost Metrics
token_usage = Counter('llm_tokens_total', 'Total tokens', ['type'])
estimated_cost = Counter('llm_cost_dollars_total', 'Estimated cost in USD')
active_queries = Gauge('rag_active_queries', 'Active queries')
retrieval_quality = Histogram('retrieval_quality_score', 'Retrieval scores')

# Gemini 2.0 Flash pricing (per 1M tokens)
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def load_rag_system():
    global rag_chain, guardrail_engine
    
    logger.info("Loading RAG system...")
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")
    
    retriever = create_retriever(INDEX_PATH, FASTEMBED_MODEL, TOP_K)
    rag_chain = create_rag_chain(
        retriever, GEMINI_MODEL, GOOGLE_API_KEY, 
        TEMPERATURE, MAX_TOKENS, False
    )
    
    try:
        guardrail_engine = GuardrailEngine("config/guardrails/guardrails_config.json")
        logger.info("✓ Guardrails initialized")
    except:
        guardrail_engine = None
    
    logger.info("✓ RAG system ready")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_rag_system()
    yield

app = FastAPI(
    title="RAG API with Monitoring",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10)
    include_sources: bool = True

class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    retrieval_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None
    latency: float
    model: str
    tokens_used: Optional[dict] = None
    estimated_cost: Optional[float] = None

@app.get("/")
async def root():
    return {
        "service": "RAG API with Monitoring",
        "version": "4.0.0",
        "monitoring": {
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "guardrails": guardrail_engine is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest):
    if rag_chain is None:
        raise HTTPException(503, "RAG not initialized")
    
    active_queries.inc()
    start_time = time.time()
    
    try:
        sanitized_query = query.question
        
        # Input validation
        if guardrail_engine:
            validation = guardrail_engine.validate_input(query.question)
            if not validation['passed']:
                for v in validation.get('violations', []):
                    guardrail_violations.labels(
                        violation_type=v.get('type', 'unknown'),
                        stage='input'
                    ).inc()
                query_counter.labels(status='rejected').inc()
                raise HTTPException(400, "Query rejected by guardrails")
            sanitized_query = validation.get('sanitized_input', query.question)
        
        # Query RAG
        result = rag_chain.invoke(sanitized_query)
        retrieval_counter.inc()
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # Track retrieval quality
        for doc in source_docs:
            score = doc.metadata.get('retrieval_score', 0)
            retrieval_quality.observe(score)
        
        # Output moderation
        if guardrail_engine:
            moderation = guardrail_engine.moderate_output(answer)
            if not moderation['passed']:
                violations = moderation.get('violations', [])
                blocking = [v for v in violations if v.get('severity') != 'WARNING']
                if blocking:
                    answer = "Response blocked by content safety."
        
        # Token tracking
        input_tokens = estimate_tokens(sanitized_query)
        output_tokens = estimate_tokens(answer)
        
        token_usage.labels(type='input').inc(input_tokens)
        token_usage.labels(type='output').inc(output_tokens)
        
        cost = (input_tokens * COST_PER_INPUT_TOKEN + 
                output_tokens * COST_PER_OUTPUT_TOKEN)
        estimated_cost.inc(cost)
        
        # Format response
        sources = None
        if query.include_sources and source_docs:
            sources = [
                SourceDocument(
                    content=doc.page_content[:300],
                    source=doc.metadata.get('source', 'unknown'),
                    page=doc.metadata.get('page'),
                    retrieval_score=doc.metadata.get('retrieval_score')
                )
                for doc in source_docs
            ]
        
        latency = time.time() - start_time
        query_counter.labels(status='success').inc()
        query_latency.observe(latency)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(latency, 3),
            model=GEMINI_MODEL,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            estimated_cost=round(cost, 6)
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