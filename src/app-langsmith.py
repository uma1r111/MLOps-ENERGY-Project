"""
FastAPI RAG Application - LangChain + FastEmbed + Google Gemini + LangSmith
Enhanced with LangChain chains and LangSmith token monitoring
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
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import (
    INDEX_PATH,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    FASTEMBED_MODEL,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
rag_chain = None
guardrail_engine = None


def load_rag_system():
    """Load RAG system with LangChain"""
    global rag_chain, guardrail_engine
    
    logger.info("Loading RAG system with LangChain + LangSmith...")
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Please run 'python src/rag/ingest.py' first."
        )
    
    try:
        # Create custom retriever
        logger.info("Creating custom retriever...")
        retriever = create_retriever(
            index_path=INDEX_PATH,
            embedding_model=FASTEMBED_MODEL,
            k=TOP_K
        )
        logger.info("✓ Retriever created")
        
        # Create RAG chain
        logger.info("Creating RAG chain...")
        rag_chain = create_rag_chain(
            retriever=retriever,
            llm_model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            conversational=False
        )
        logger.info("✓ RAG chain created")
        
        # Initialize guardrails
        logger.info("Initializing guardrails...")
        try:
            guardrail_engine = GuardrailEngine(
                config_path="config/guardrails/guardrails_config.json"
            )
            logger.info("✓ Guardrails initialized")
        except Exception as e:
            logger.warning(f"⚠️  Guardrails initialization failed: {str(e)}")
            guardrail_engine = None
        
        logger.info("=" * 60)
        logger.info("🚀 RAG System Ready!")
        logger.info(f"  Embeddings: {FASTEMBED_MODEL} (local)")
        logger.info(f"  LLM: {GEMINI_MODEL}")
        logger.info(f"  LangSmith: {'✓ Active' if LANGSMITH_API_KEY else '✗ Disabled'}")
        logger.info(f"  Guardrails: {'✓ Active' if guardrail_engine else '✗ Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load RAG system: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    try:
        load_rag_system()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
    
    yield
    
    logger.info("Shutting down...")


# Initialize FastAPI
app = FastAPI(
    title="Energy RAG API - LangChain Edition",
    description="RAG API with LangChain, FastEmbed, Google Gemini, and LangSmith monitoring",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Clear existing metrics
def clear_metrics():
    """Clear existing Prometheus metrics"""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


clear_metrics()

# Prometheus Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries', ['status'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency')
retrieval_counter = Counter('rag_retrievals_total', 'Total retrievals')
guardrail_violations = Counter('guardrail_violations_total', 'Guardrail violations', ['violation_type', 'stage'])
guardrail_checks = Counter('guardrail_checks_total', 'Guardrail checks', ['check_type', 'result'])


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What factors affect solar panel efficiency?",
                "top_k": 3,
                "include_sources": True
            }
        }
    )
    
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10)
    include_sources: bool = Field(True, description="Include source documents")


class SourceDocument(BaseModel):
    """Source document model"""
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    retrieval_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model"""
    answer: str
    sources: Optional[List[SourceDocument]] = None
    latency: float
    model: str
    embedding_model: str
    langsmith_trace: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Energy RAG API - LangChain Edition",
        "version": "3.0.0",
        "features": {
            "langchain": "✓ Enabled",
            "langsmith": "✓ Enabled" if LANGSMITH_API_KEY else "✗ Disabled",
            "guardrails": "✓ Enabled" if guardrail_engine else "✗ Disabled",
            "embedding_model": FASTEMBED_MODEL + " (local)",
            "llm_model": GEMINI_MODEL
        },
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return {
        "status": "healthy",
        "rag_initialized": True,
        "guardrails_enabled": guardrail_engine is not None,
        "langsmith_enabled": LANGSMITH_API_KEY is not None,
        "embedding_model": FASTEMBED_MODEL,
        "llm_model": GEMINI_MODEL
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest):
    """
    Query the RAG system with LangChain.
    
    Flow:
    1. 🛡️ Input validation (guardrails)
    2. 🤖 RAG chain invocation (LangSmith tracked)
    3. 🛡️ Output moderation (guardrails)
    4. ✅ Return response
    """
    if rag_chain is None:
        query_counter.labels(status='error').inc()
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    sanitized_query = query.question
    
    try:
        logger.info(f"Processing query: {query.question[:50]}...")
        
        # Step 1: Input validation
        if guardrail_engine:
            logger.info("🛡️ Input validation...")
            input_validation = guardrail_engine.validate_input(query.question)
            
            if not input_validation['passed']:
                violations = input_validation.get('violations', [])
                for v in violations:
                    guardrail_violations.labels(
                        violation_type=v.get('type', 'unknown'),
                        stage='input'
                    ).inc()
                
                query_counter.labels(status='guardrail_rejected').inc()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Query rejected by guardrails",
                        "violations": violations
                    }
                )
            
            sanitized_query = input_validation.get('sanitized_input', query.question)
            logger.info("✓ Input validation passed")
        
        # Step 2: Invoke RAG chain (LangSmith tracks this automatically)
        logger.info("🤖 Invoking RAG chain...")
        result = rag_chain.invoke(sanitized_query)
        retrieval_counter.inc()
        logger.info("✓ RAG chain completed")
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # Step 3: Output moderation
        if guardrail_engine:
            logger.info("🛡️ Output moderation...")
            output_moderation = guardrail_engine.moderate_output(answer)
            
            if not output_moderation['passed']:
                violations = output_moderation.get('violations', [])
                blocking = [v for v in violations if v.get('severity') != 'WARNING']
                
                if blocking:
                    answer = (
                        "I apologize, but I cannot provide this response as it "
                        "violates content safety guidelines."
                    )
                    logger.warning("🚫 Output blocked")
            
            logger.info("✓ Output moderation passed")
        
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
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        query_counter.labels(status='success').inc()
        query_latency.observe(latency)
        
        logger.info(f"✅ Query completed in {latency:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(latency, 3),
            model=GEMINI_MODEL,
            embedding_model=FASTEMBED_MODEL,
            langsmith_trace=f"Check LangSmith dashboard: {LANGSMITH_PROJECT}" if LANGSMITH_API_KEY else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        query_counter.labels(status='error').inc()
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )