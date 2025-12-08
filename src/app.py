"""
FastAPI RAG Application
Merged: LangChain + FastEmbed + Gemini + LangSmith + A/B Testing + Guardrails + Metrics
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Environment fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Pydantic
from pydantic import BaseModel, Field

# Prometheus
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# Add src/ to path
sys.path.append(str(Path(__file__).parent.parent))

# Internal imports
from src.rag.config import (
    INDEX_PATH,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    FASTEMBED_MODEL,
    TOP_K,
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain
from src.guardrails import GuardrailEngine
from src.monitoring.ab_testing import (
    create_ab_testing_engine,
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# GLOBAL COMPONENTS
# -----------------------------
rag_chains = {}  # variant → chain
retriever = None  # shared retriever
guardrail_engine = None
ab_engine = None

# -----------------------------
# TOKEN COSTS
# -----------------------------
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000


def estimate_tokens(text: str) -> int:
    return len(text) // 4


# -----------------------------
# METRICS
# -----------------------------
def clear_metrics():
    collectors = list(REGISTRY._collector_to_names.keys())
    for c in collectors:
        try:
            REGISTRY.unregister(c)
        except Exception:
            pass


clear_metrics()

query_counter = Counter("rag_queries_total", "Total RAG queries", ["status"])
query_latency = Histogram("rag_query_latency_seconds", "Query latency")
retrieval_counter = Counter("rag_retrievals_total", "Total retrievals")
guardrail_violations = Counter(
    "guardrail_violations_total", "Guardrail violations", ["violation_type", "stage"]
)
guardrail_checks = Counter(
    "guardrail_checks_total", "Guardrail checks", ["check_type", "result"]
)


# -----------------------------
# Pydantic Models
# -----------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: Optional[int] = TOP_K
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
    embedding_model: str
    variant_id: Optional[str] = None
    langsmith_trace: Optional[str] = None


# -----------------------------
# SYSTEM INITIALIZATION
# -----------------------------
def load_rag_system():
    global rag_chains, retriever, ab_engine, guardrail_engine

    logger.info("Initializing RAG system with A/B testing...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at: {INDEX_PATH}. Run ingest.py first."
        )

    # Shared retriever
    retriever = create_retriever(INDEX_PATH, FASTEMBED_MODEL, TOP_K)

    # A/B testing variants
    ab_engine = create_ab_testing_engine(
        enabled_variants=["control", "concise", "detailed", "conversational"],
        traffic_split={
            "control": 0.40,
            "concise": 0.20,
            "detailed": 0.20,
            "conversational": 0.20,
        },
    )

    # Build RAG chain for each variant
    for variant in ab_engine.variants:
        logger.info(f"Creating chain for variant: {variant.name}")
        rag_chains[variant.id] = create_rag_chain(
            retriever=retriever,
            llm_model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=variant.temperature,
            max_tokens=variant.max_tokens,
            conversational=False,
            system_prompt=variant.system_prompt,
        )

    # Guardrails
    try:
        guardrail_engine = GuardrailEngine("config/guardrails/guardrails_config.json")
        logger.info("✓ Guardrails ready")
    except Exception as e:
        logger.warning(f"Guardrails failed: {e}")
        guardrail_engine = None

    logger.info(f"✓ RAG System Loaded ({len(rag_chains)} variants)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_rag_system()
    yield


# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Energy RAG API (Merged)", version="6.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# ROOT & HEALTH
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "RAG API Running",
        "variants": list(rag_chains.keys()),
        "guardrails": guardrail_engine is not None,
        "metrics": "/metrics",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "variants_loaded": len(rag_chains)}


# -----------------------------
# QUERY ENDPOINT
# -----------------------------
@app.post("/query", response_model=QueryResponse)
async def query_rag(body: QueryRequest):
    start = time.time()

    # 1 — Guardrail INPUT check
    sanitized_query = body.question
    if guardrail_engine:
        result = guardrail_engine.validate_input(body.question)
        guardrail_checks.labels(check_type="input", result="ok").inc()

        if not result["passed"]:
            query_counter.labels(status="guardrail_reject").inc()
            for v in result["violations"]:
                guardrail_violations.labels(v["type"], "input").inc()

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Rejected by guardrails",
                    "violations": result["violations"],
                },
            )

        sanitized_query = result.get("sanitized_input", body.question)

    # 2 — Determine variant
    variant = ab_engine.assign_variant(
        requested_variant=body.variant_id, user_id=body.user_id
    )
    chain = rag_chains[variant.id]

    # 3 — Run RAG
    try:
        with query_latency.time():
            response = chain.invoke({"question": sanitized_query})
    except Exception as e:
        query_counter.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

    query_counter.labels(status="success").inc()

    # 4 — Guardrail OUTPUT check
    if guardrail_engine:
        ocheck = guardrail_engine.validate_output(response["answer"])
        guardrail_checks.labels(check_type="output", result="ok").inc()

        if not ocheck["passed"]:
            for v in ocheck["violations"]:
                guardrail_violations.labels(v["type"], "output").inc()

    # 5 — Build response
    latency = time.time() - start

    return QueryResponse(
        answer=response["answer"],
        sources=(
            [
                SourceDocument(
                    content=d.page_content,
                    source=d.metadata.get("source", "unknown"),
                    page=d.metadata.get("page"),
                    retrieval_score=d.metadata.get("score"),
                )
                for d in response.get("source_documents", [])
            ]
            if body.include_sources
            else None
        ),
        latency=latency,
        model=GEMINI_MODEL,
        embedding_model=FASTEMBED_MODEL,
        variant_id=variant.id,
        langsmith_trace=response.get("langsmith_trace"),
    )


# -----------------------------
# FEEDBACK
# -----------------------------
@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    ab_engine.record_feedback(
        variant_id=body.variant_id, score=body.satisfaction_score, comment=body.comment
    )
    return {"status": "recorded"}


# -----------------------------
# METRICS
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
