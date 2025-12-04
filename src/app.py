"""
FastAPI RAG Application - FastEmbed + Google Gemini
Provides REST API for querying the RAG system.
"""
import os
import sys
import time
import logging
import pickle
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from fastembed import TextEmbedding
import numpy as np
import faiss

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import (
    INDEX_PATH,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    FASTEMBED_MODEL,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS
)

from src.guardrails import GuardrailEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG Components (loaded once at startup)
faiss_index = None
documents = None
embedding_model = None
llm = None
guardrail_engine = None

def load_rag_system():
    """Load FAISS index and initialize RAG system"""
    global faiss_index, documents, embedding_model, llm, guardrail_engine
    
    logger.info("Loading RAG system (FastEmbed + Google Gemini)...")
    
    # Check if index exists
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Please run 'python src/rag/ingest.py' first."
        )
    
    try:
        # Load FastEmbed (local embeddings)
        logger.info(f"Loading FastEmbed: {FASTEMBED_MODEL}")
        embedding_model = TextEmbedding(model_name=FASTEMBED_MODEL)
        logger.info("✓ FastEmbed loaded (running locally)")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        faiss_index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
        logger.info(f"✓ FAISS index loaded: {faiss_index.ntotal} vectors")
        
        # Load documents
        with open(os.path.join(INDEX_PATH, "documents.pkl"), "rb") as f:
            documents = pickle.load(f)
        logger.info(f"✓ Loaded {len(documents)} document chunks")
        
        # Initialize Google Gemini LLM
        logger.info(f"Initializing Google Gemini: {GEMINI_MODEL}")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            convert_system_message_to_human=True
        )
        logger.info("✓ Gemini model initialized")

        logger.info("Initializing Guardrails...")
        try:
            guardrail_engine = GuardrailEngine(
                config_path="config/guardrails/guardrails_config.json"
            )
            logger.info("✓ Guardrails initialized")
            logger.info(f"  - Input Validators: Active")
            logger.info(f"  - Output Moderators: Active")
        except Exception as e:
            logger.warning(f"⚠️  Guardrails initialization failed: {str(e)}")
            logger.warning("Continuing without guardrails...")
            guardrail_engine = None
        
        logger.info("=" * 60)
        logger.info("RAG System Ready! 🚀")
        logger.info(f"  Embeddings: {FASTEMBED_MODEL} (local)")
        logger.info(f"  LLM: {GEMINI_MODEL}")
        logger.info(f"  Guardrails: {'✓ Active' if guardrail_engine else '✗ Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load RAG system: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown"""
    # Startup
    try:
        load_rag_system()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Energy RAG API - FastEmbed + Gemini",
    description="RAG API with FastEmbed (local embeddings) and Google Gemini 2.0 Flash",
    version="2.0.0",
    lifespan=lifespan
)


# Clear any existing metrics (prevents duplication error)
def clear_metrics():
    """Clear existing Prometheus metrics to avoid duplication"""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


# Clear metrics before creating new ones
clear_metrics()

# Prometheus Metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['status']
)
query_latency = Histogram(
    'rag_query_latency_seconds',
    'RAG query latency in seconds'
)
token_counter = Counter(
    'rag_tokens_used_total',
    'Total tokens used by LLM'
)
retrieval_counter = Counter(
    'rag_retrievals_total',
    'Total number of document retrievals'
)

guardrail_violations = Counter(
    'guardrail_violations_total',
    'Total guardrail violations by type',
    ['violation_type', 'stage']  # stage: input or output
)

guardrail_checks = Counter(
    'guardrail_checks_total',
    'Total guardrail checks performed',
    ['check_type', 'result']  # result: passed or failed
)

guardrail_latency = Histogram(
    'guardrail_check_latency_seconds',
    'Time taken for guardrail checks',
    ['stage']  # input or output
)


# Pydantic Models (Fixed for Pydantic v2)
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What factors affect solar panel efficiency?",
                "top_k": 3
            }
        }
    )
    
    question: str = Field(..., min_length=5, max_length=500, description="Question to ask")
    top_k: Optional[int] = Field(TOP_K, ge=1, le=10, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    """Model for source document information"""
    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Source file path")
    page: Optional[int] = Field(None, description="Page number (for PDFs)")


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Solar panel efficiency is affected by...",
                "sources": [
                    {
                        "content": "Temperature affects solar panel...",
                        "source": "solar_energy.pdf",
                        "page": 5
                    }
                ],
                "latency": 1.23,
                "tokens_used": 450,
                "model": "gemini-2.0-flash-exp",
                "embedding_model": "BAAI/bge-small-en-v1.5"
            }
        }
    )
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    latency: float = Field(..., description="Query latency in seconds")
    tokens_used: Optional[int] = Field(None, description="Tokens used by LLM")
    model: str = Field(..., description="Model used for generation")
    embedding_model: str = Field(..., description="Embedding model used")


def retrieve_documents(query: str, top_k: int = TOP_K) -> List[Document]:
    """
    Retrieve relevant documents for a query using FastEmbed + FAISS.
    
    Args:
        query: Query string
        top_k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    # Create query embedding with FastEmbed
    query_embedding = np.array(list(embedding_model.embed([query])), dtype=np.float32)
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Get documents
    retrieved_docs = [documents[idx] for idx in indices[0]]
    
    return retrieved_docs


def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    Generate answer using Google Gemini with retrieved context.
    
    Args:
        query: User question
        context_docs: Retrieved context documents
        
    Returns:
        Generated answer string
    """
    # Create context from documents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Provide a clear, detailed answer based on the context.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    prompt = prompt_template.format(context=context, question=query)
    
    # Generate answer with Gemini
    response = llm.invoke(prompt)
    
    return response.content


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Energy RAG API - FastEmbed + Google Gemini",
        "version": "2.0.0",
        "embedding_model": FASTEMBED_MODEL + " (local)",
        "llm_model": GEMINI_MODEL,
        "status": "running",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "metrics": "/metrics",
            "guardrails_stats": "/guardrails/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if faiss_index is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please run ingestion first."
        )
    
    return {
        "status": "healthy",
        "rag_initialized": True,
        "guardrails_enabled": guardrail_engine is not None,
        "embedding_model": FASTEMBED_MODEL + " (local)",
        "llm_model": GEMINI_MODEL,
        "num_documents": len(documents) if documents else 0,
        "num_vectors": faiss_index.ntotal if faiss_index else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest):
    """
    Query the RAG system with guardrails protection.
    
    Flow:
    1. 🛡️ INPUT VALIDATION (Guardrails)
    2. 📚 Document Retrieval
    3. 🤖 Answer Generation
    4. 🛡️ OUTPUT MODERATION (Guardrails)
    5. ✅ Return Response
    """
    # Check if RAG system is loaded
    if faiss_index is None or llm is None:
        query_counter.labels(status='error').inc()
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    guardrail_flags = {}
    sanitized_query = query.question  # Will be updated if PII detected
    
    try:
        logger.info(f"Processing query: {query.question[:50]}...")
        
        # ============================================================
        # STEP 1: INPUT VALIDATION (GUARDRAILS) - FIXED
        # ============================================================
        if guardrail_engine:
            logger.info("🛡️ Step 1: Input Validation & Guardrails")
            input_start = time.time()
            
            try:
                # CRITICAL FIX: Use correct field name 'passed' not 'is_safe'
                input_validation = guardrail_engine.validate_input(query.question)
                guardrail_flags['input_validation'] = {
                    'passed': input_validation['passed'],
                    'pii_detected': len(input_validation.get('pii_detected', [])) > 0,
                    'injection_detected': input_validation.get('injection_detected', False)
                }
                
                # Track metrics
                guardrail_checks.labels(
                    check_type='input',
                    result='passed' if input_validation['passed'] else 'failed'
                ).inc()
                
                guardrail_latency.labels(stage='input').observe(time.time() - input_start)
                
                # CRITICAL FIX: Check 'passed' field, not 'is_safe'
                if not input_validation['passed']:
                    violations = input_validation.get('violations', [])
                    
                    # Log each violation
                    for violation in violations:
                        violation_type = violation.get('type', 'unknown')
                        guardrail_violations.labels(
                            violation_type=violation_type,
                            stage='input'
                        ).inc()
                        logger.warning(f"🚫 Input violation: {violation_type} - {violation.get('message', '')}")
                    
                    query_counter.labels(status='guardrail_rejected').inc()
                    
                    # BLOCK THE REQUEST
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Query rejected by input guardrails",
                            "violations": violations,
                            "message": "Your query violates content safety policies. Please rephrase and try again."
                        }
                    )
                
                # Use sanitized input (PII anonymized)
                sanitized_query = input_validation.get('sanitized_input', query.question)
                
                # Log PII detection
                if input_validation.get('pii_detected'):
                    pii_types = [pii['type'] for pii in input_validation['pii_detected']]
                    logger.info(f"ℹ️  PII detected and anonymized: {pii_types}")
                
                logger.info(f"✓ Input validation passed ({time.time() - input_start:.3f}s)")
                
            except HTTPException:
                raise  # Re-raise HTTP exceptions (blocks)
            except Exception as e:
                logger.error(f"Guardrail check failed: {str(e)}", exc_info=True)
                # Continue without guardrails if they fail
                guardrail_flags['input_validation_error'] = str(e)
        
        # ============================================================
        # STEP 2: DOCUMENT RETRIEVAL (Use sanitized query)
        # ============================================================
        logger.info("📚 Step 2: Document Retrieval")
        retrieved_docs = retrieve_documents(sanitized_query, query.top_k)
        retrieval_counter.inc()
        logger.info(f"✓ Retrieved {len(retrieved_docs)} documents")
        
        # ============================================================
        # STEP 3: ANSWER GENERATION (Use sanitized query)
        # ============================================================
        logger.info("🤖 Step 3: Answer Generation")
        answer = generate_answer(sanitized_query, retrieved_docs)
        logger.info(f"✓ Answer generated ({len(answer)} chars)")
        
        # ============================================================
        # STEP 4: OUTPUT MODERATION (GUARDRAILS) - FIXED
        # ============================================================
        if guardrail_engine:
            logger.info("🛡️ Step 4: Output Moderation")
            output_start = time.time()
            
            try:
                # CRITICAL FIX: Use correct field name 'passed' not 'is_safe'
                output_moderation = guardrail_engine.moderate_output(answer)
                guardrail_flags['output_moderation'] = {
                    'passed': output_moderation['passed'],
                    'toxicity_detected': bool(output_moderation.get('toxicity_scores')),
                    'hallucination_detected': output_moderation.get('hallucination_detected', False)
                }
                
                # Track metrics
                guardrail_checks.labels(
                    check_type='output',
                    result='passed' if output_moderation['passed'] else 'failed'
                ).inc()
                
                guardrail_latency.labels(stage='output').observe(time.time() - output_start)
                
                # CRITICAL FIX: Check 'passed' field, not 'is_safe'
                if not output_moderation['passed']:
                    violations = output_moderation.get('violations', [])
                    
                    # Log violations
                    for violation in violations:
                        violation_type = violation.get('type', 'unknown')
                        guardrail_violations.labels(
                            violation_type=violation_type,
                            stage='output'
                        ).inc()
                        logger.warning(f"🚫 Output violation: {violation_type} - {violation.get('message', '')}")
                    
                    # Check if any blocking violations (not just warnings)
                    blocking_violations = [
                        v for v in violations 
                        if v.get('severity') != 'WARNING'
                    ]
                    
                    if blocking_violations:
                        # BLOCK: Return sanitized response
                        answer = (
                            "I apologize, but I cannot provide this response as it "
                            "violates content safety guidelines. The generated content "
                            "contained inappropriate material. Please rephrase your "
                            "question or ask something else."
                        )
                        guardrail_flags['output_blocked'] = True
                        logger.warning("🚫 Output blocked due to safety violations")
                    else:
                        # Just warnings, use sanitized output
                        answer = output_moderation.get('sanitized_output', answer)
                        guardrail_flags['output_sanitized'] = True
                
                logger.info(f"✓ Output moderation passed ({time.time() - output_start:.3f}s)")
                
            except Exception as e:
                logger.error(f"Output moderation failed: {str(e)}", exc_info=True)
                guardrail_flags['output_moderation_error'] = str(e)
        
        # ============================================================
        # STEP 5: FORMAT AND RETURN RESPONSE
        # ============================================================
        # Format sources
        sources = []
        for doc in retrieved_docs:
            sources.append(SourceDocument(
                content=doc.page_content[:300],
                source=doc.metadata.get('source', 'unknown'),
                page=doc.metadata.get('page', None)
            ))
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        query_counter.labels(status='success').inc()
        query_latency.observe(latency)
        
        # Estimate tokens
        tokens_used = len(answer.split()) * 1.3
        token_counter.inc(tokens_used)
        
        logger.info(f"✅ Query completed successfully in {latency:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(latency, 3),
            tokens_used=int(tokens_used),
            model=GEMINI_MODEL,
            embedding_model=FASTEMBED_MODEL + " (local)"
        )
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions (already logged)
    except Exception as e:
        query_counter.labels(status='error').inc()
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/guardrails/stats")
async def guardrails_stats():
    """Get guardrails statistics"""
    if not guardrail_engine:
        return {
            "enabled": False,
            "message": "Guardrails not initialized"
        }
    
    try:
        stats = guardrail_engine.get_stats()
        return {
            "enabled": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get guardrail stats: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )