"""
<<<<<<< HEAD
FastAPI RAG Application - LangChain + FastEmbed + Google Gemini + LangSmith
Enhanced with LangChain chains and LangSmith token monitoring
=======
Enhanced FastAPI RAG with A/B Testing
>>>>>>> feat/ab-testing
"""
import sys
import time
import logging
import os
<<<<<<< HEAD
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

=======
>>>>>>> feat/ab-testing
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

<<<<<<< HEAD
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
=======
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import *
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine
from src.monitoring.ab_testing import (
    create_ab_testing_engine,
    ABTestResult,
    PromptVariants
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
rag_chains = {}  # Dict of variant_id -> rag_chain
ab_engine = None
guardrail_engine = None
retriever = None

# Pricing
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def load_rag_system():
    global rag_chains, ab_engine, guardrail_engine, retriever
    
    logger.info("Loading RAG system with A/B testing...")
    
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")
    
    # Create retriever (shared)
    retriever = create_retriever(INDEX_PATH, FASTEMBED_MODEL, TOP_K)
    
    # Initialize A/B testing with traffic split
    ab_engine = create_ab_testing_engine(
        enabled_variants=['control', 'concise', 'detailed', 'conversational'],
        traffic_split={
            'control': 0.40,        # 40% control
            'concise': 0.20,        # 20% concise
            'detailed': 0.20,       # 20% detailed
            'conversational': 0.20  # 20% conversational
        }
    )
    
    # Create RAG chain for each variant
    for variant in ab_engine.variants:
        logger.info(f"Creating chain for variant: {variant.name}")
        chain = create_rag_chain(
            retriever=retriever,
            llm_model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=variant.temperature,
            max_tokens=variant.max_tokens,
            conversational=False,
            system_prompt=variant.system_prompt
        )
        rag_chains[variant.id] = chain
    
    # Load guardrails
    try:
        guardrail_engine = GuardrailEngine("config/guardrails/guardrails_config.json")
        logger.info("✓ Guardrails initialized")
    except:
        guardrail_engine = None
    
    logger.info(f"✓ RAG system ready with {len(rag_chains)} variants")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_rag_system()
    yield

app = FastAPI(
    title="RAG API with A/B Testing",
    version="5.0.0",
    lifespan=lifespan
)

>>>>>>> feat/ab-testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD

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
=======
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10)
    include_sources: bool = True
    variant_id: Optional[str] = None  # Optional: specify variant
    user_id: Optional[str] = None     # Optional: for consistent assignment

class FeedbackRequest(BaseModel):
    query: str
    variant_id: str
    satisfaction_score: float = Field(..., ge=0, le=5)
    comment: Optional[str] = None

class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    retrieval_score: Optional[float] = None

class QueryResponse(BaseModel):
>>>>>>> feat/ab-testing
    answer: str
    sources: Optional[List[SourceDocument]] = None
    latency: float
    model: str
<<<<<<< HEAD
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
=======
    variant_id: str
    variant_name: str
    tokens_used: Optional[dict] = None
    estimated_cost: Optional[float] = None

@app.get("/")
async def root():
    return {
        "service": "RAG API with A/B Testing",
        "version": "5.0.0",
        "features": {
            "ab_testing": "✓ Enabled",
            "variants": len(rag_chains),
            "monitoring": "✓ Prometheus + Grafana"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "variants_loaded": len(rag_chains),
        "ab_testing": ab_engine is not None
    }

@app.get("/variants")
async def list_variants():
    """List all available prompt variants"""
    return {
        "variants": [
            {
                "id": v.id,
                "name": v.name,
                "description": v.description,
                "temperature": v.temperature,
                "max_tokens": v.max_tokens,
                "traffic_percentage": ab_engine.traffic_split.get(v.id, 0) * 100
            }
            for v in ab_engine.variants
        ]
    }

@app.get("/ab-stats")
async def ab_stats():
    """Get A/B testing statistics"""
    return ab_engine.get_comparison_report()

@app.post("/query", response_model=QueryResponse)
async def query_rag(
    query: QueryRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Query with A/B testing"""
    if not rag_chains:
        raise HTTPException(503, "RAG not initialized")
    
    start_time = time.time()
    
    try:
        # Assign variant
        if query.variant_id:
            variant = PromptVariants.get_variant(query.variant_id)
            if not variant:
                raise HTTPException(400, f"Invalid variant: {query.variant_id}")
        else:
            user_id = x_user_id or query.user_id
            variant = ab_engine.assign_variant(user_id)
        
        logger.info(f"Using variant: {variant.name} ({variant.id})")
        
        # Get appropriate chain
        chain = rag_chains[variant.id]
        
        sanitized_query = query.question
        
        # Input validation
        if guardrail_engine:
            validation = guardrail_engine.validate_input(query.question)
            if not validation['passed']:
                raise HTTPException(400, "Query rejected by guardrails")
            sanitized_query = validation.get('sanitized_input', query.question)
        
        # Query RAG
        result = chain.invoke(sanitized_query)
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
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
        cost = (input_tokens * COST_PER_INPUT_TOKEN + 
                output_tokens * COST_PER_OUTPUT_TOKEN)
        
        # Log A/B test result
        retrieval_scores = [doc.metadata.get('retrieval_score', 0) for doc in source_docs]
        
        ab_result = ABTestResult(
            variant_id=variant.id,
            variant_name=variant.name,
            query=sanitized_query,
            answer=answer,
            latency=time.time() - start_time,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost=cost,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            retrieval_scores=retrieval_scores
        )
        ab_engine.log_result(ab_result)
        
        # Format response
>>>>>>> feat/ab-testing
        sources = None
        if query.include_sources and source_docs:
            sources = [
                SourceDocument(
                    content=doc.page_content[:300],
                    source=doc.metadata.get('source', 'unknown'),
                    page=doc.metadata.get('page'),
<<<<<<< HEAD
                    chunk_id=doc.metadata.get('chunk_id'),
=======
>>>>>>> feat/ab-testing
                    retrieval_score=doc.metadata.get('retrieval_score')
                )
                for doc in source_docs
            ]
        
<<<<<<< HEAD
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
=======
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(time.time() - start_time, 3),
            model=GEMINI_MODEL,
            variant_id=variant.id,
            variant_name=variant.name,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            estimated_cost=round(cost, 6)
>>>>>>> feat/ab-testing
        )
    
    except HTTPException:
        raise
    except Exception as e:
<<<<<<< HEAD
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
=======
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for A/B testing"""
    results = ab_engine.load_results()
    
    # Find matching result
    for result in reversed(results):
        if (result['query'] == feedback.query and 
            result['variant_id'] == feedback.variant_id):
            # Update satisfaction score
            result['satisfaction_score'] = feedback.satisfaction_score
            if feedback.comment:
                result['comment'] = feedback.comment
            
            # Re-log result
            ab_result = ABTestResult(**result)
            ab_engine.log_result(ab_result)
            
            return {"status": "success", "message": "Feedback recorded"}
    
    raise HTTPException(404, "Query result not found")

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
>>>>>>> feat/ab-testing
