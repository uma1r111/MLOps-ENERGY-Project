"""
Enhanced FastAPI RAG with A/B Testing and LangSmith Monitoring
(Compliant with LangSmith setup in config.py)
"""

import sys
import time
import logging
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Set initial thread environment variables for stability
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# FIXED: Import specific variables instead of using wildcard import
from src.rag.config import (
    INDEX_PATH,
    FASTEMBED_MODEL,
    TOP_K,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine
from src.monitoring.ab_testing import (
    create_ab_testing_engine,
    ABTestResult,
    PromptVariants,
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
    """A very rough estimate of tokens based on character count."""
    return len(text) // 4


def load_rag_system():
    global rag_chains, ab_engine, guardrail_engine, retriever

    logger.info("Loading RAG system with A/B testing...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}")

    # Create retriever (shared)
    logger.info("Creating custom retriever...")
    retriever = create_retriever(INDEX_PATH, FASTEMBED_MODEL, TOP_K)
    logger.info("✓ Retriever created")

    # Initialize A/B testing with traffic split
    ab_engine = create_ab_testing_engine(
        enabled_variants=["control", "concise", "detailed", "conversational"],
        traffic_split={
            "control": 0.40,
            "concise": 0.20,
            "detailed": 0.20,
            "conversational": 0.20,
        },
    )
    logger.info(f"✓ A/B Engine ready with {len(ab_engine.variants)} variants")

    # Create RAG chain for each variant
    for variant in ab_engine.variants:
        logger.info(f"Creating chain for variant: {variant.name}")
        # The rag_chain creation implicitly picks up LangSmith tracing
        # from the environment variables set in config.py
        chain = create_rag_chain(
            retriever=retriever,
            llm_model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=variant.temperature,
            max_tokens=variant.max_tokens,
            conversational=False,
            system_prompt=variant.system_prompt,
        )
        rag_chains[variant.id] = chain

    # Load guardrails
    try:
        guardrail_engine = GuardrailEngine("config/guardrails/guardrails_config.json")
        logger.info("✓ Guardrails initialized")
    except Exception as e:
        logger.warning(f"⚠️  Guardrails initialization failed: {str(e)}")
        guardrail_engine = None

    logger.info(f"✓ RAG system ready with {len(rag_chains)} variants")


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
    title="RAG API with A/B Testing and LangSmith",
    description="RAG API with A/B Testing, LangSmith monitoring, and Guardrails",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models (Updated for LangSmith) ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10)
    include_sources: bool = True
    variant_id: Optional[str] = None
    user_id: Optional[str] = None


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
    answer: str
    sources: Optional[List[SourceDocument]] = None
    latency: float
    model: str
    variant_id: str
    variant_name: str
    tokens_used: Optional[dict] = None
    estimated_cost: Optional[float] = None
    # ADDED: LangSmith trace information
    langsmith_trace: Optional[str] = None


# --- Endpoints ---


@app.get("/")
async def root():
    return {
        "service": "RAG API with A/B Testing",
        "version": "5.0.0",
        "features": {
            "ab_testing": "✓ Enabled",
            "variants": len(rag_chains),
            "monitoring": "✓ Prometheus + Grafana",
            # Check LangSmith status via the config variable
            "langsmith": "✓ Active" if LANGSMITH_API_KEY else "✗ Disabled",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "variants_loaded": len(rag_chains),
        "ab_testing": ab_engine is not None,
        "langsmith_enabled": LANGSMITH_API_KEY is not None,
    }


@app.get("/variants")
async def list_variants():
    """List all available prompt variants"""
    if ab_engine is None:
        raise HTTPException(503, "A/B Testing Engine not initialized")
    return {
        "variants": [
            {
                "id": v.id,
                "name": v.name,
                "description": v.description,
                "temperature": v.temperature,
                "max_tokens": v.max_tokens,
                "traffic_percentage": ab_engine.traffic_split.get(v.id, 0) * 100,
            }
            for v in ab_engine.variants
        ]
    }


@app.get("/ab-stats")
async def ab_stats():
    """Get A/B testing statistics"""
    if ab_engine is None:
        raise HTTPException(503, "A/B Testing Engine not initialized")
    return ab_engine.get_comparison_report()


@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest, x_user_id: Optional[str] = Header(None)):
    """Query with A/B testing and LangSmith tracing"""
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
            if not validation["passed"]:
                raise HTTPException(400, "Query rejected by guardrails")
            sanitized_query = validation.get("sanitized_input", query.question)

        # Query RAG - Traced by LangSmith due to config.py setup
        result = chain.invoke(sanitized_query)
        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        # Output moderation
        if guardrail_engine:
            moderation = guardrail_engine.moderate_output(answer)
            if not moderation["passed"]:
                violations = moderation.get("violations", [])
                blocking = [v for v in violations if v.get("severity") != "WARNING"]
                if blocking:
                    answer = "Response blocked by content safety."

        # Token tracking
        input_tokens = estimate_tokens(sanitized_query)
        output_tokens = estimate_tokens(answer)
        cost = (
            input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN
        )

        # Log A/B test result (existing functionality preserved)
        retrieval_scores = [
            doc.metadata.get("retrieval_score", 0) for doc in source_docs
        ]

        ab_result = ABTestResult(
            variant_id=variant.id,
            variant_name=variant.name,
            query=sanitized_query,
            answer=answer,
            latency=time.time() - start_time,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost=cost,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            retrieval_scores=retrieval_scores,
        )
        ab_engine.log_result(ab_result)

        # Format sources
        sources = None
        if query.include_sources and source_docs:
            sources = [
                SourceDocument(
                    content=doc.page_content[:300],
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    retrieval_score=doc.metadata.get("retrieval_score"),
                )
                for doc in source_docs
            ]

        # Determine LangSmith trace link
        langsmith_url = None
        if LANGSMITH_API_KEY:
            # Rely on the LangSmith project name set in config.py
            langsmith_url = f"Trace logged to LangSmith project: {LANGSMITH_PROJECT}"

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
                "total": input_tokens + output_tokens,
            },
            estimated_cost=round(cost, 6),
            langsmith_trace=langsmith_url,  # New field returned
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for A/B testing"""
    if ab_engine is None:
        raise HTTPException(503, "A/B Testing Engine not initialized")

    results = ab_engine.load_results()

    # Find matching result
    for result in reversed(results):
        if (
            result.get("query") == feedback.query
            and result.get("variant_id") == feedback.variant_id
        ):

            # Update satisfaction score
            result["satisfaction_score"] = feedback.satisfaction_score
            if feedback.comment:
                result["comment"] = feedback.comment

            # Re-log result
            ab_result = ABTestResult(**result)
            ab_engine.log_result(ab_result)

            return {"status": "success", "message": "Feedback recorded"}

    raise HTTPException(404, "Query result not found")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
