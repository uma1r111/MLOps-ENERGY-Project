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


def load_rag_system():
    """Load FAISS index and initialize RAG system"""
    global faiss_index, documents, embedding_model, llm
    
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
        
        logger.info("=" * 60)
        logger.info("RAG System Ready! 🚀")
        logger.info(f"  Embeddings: {FASTEMBED_MODEL} (local)")
        logger.info(f"  LLM: {GEMINI_MODEL}")
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
            "metrics": "/metrics"
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
        "embedding_model": FASTEMBED_MODEL + " (local)",
        "llm_model": GEMINI_MODEL,
        "num_documents": len(documents) if documents else 0,
        "num_vectors": faiss_index.ntotal if faiss_index else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(query: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        query: QueryRequest with question and optional top_k
        
    Returns:
        QueryResponse with answer, sources, and metadata
    """
    # Check if RAG system is loaded
    if faiss_index is None or llm is None:
        query_counter.labels(status='error').inc()
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {query.question[:50]}...")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = retrieve_documents(query.question, query.top_k)
        retrieval_counter.inc()
        
        # Step 2: Generate answer with Gemini
        answer = generate_answer(query.question, retrieved_docs)
        
        # Format sources
        sources = []
        for doc in retrieved_docs:
            sources.append(SourceDocument(
                content=doc.page_content[:300],  # Truncate for response
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
        
        logger.info(f"✓ Query completed in {latency:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            latency=round(latency, 3),
            tokens_used=int(tokens_used),
            model=GEMINI_MODEL,
            embedding_model=FASTEMBED_MODEL + " (local)"
        )
    
    except Exception as e:
        query_counter.labels(status='error').inc()
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
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
        app,  # Changed from "app:app" to just app
        host="0.0.0.0",
        port=8000,
        reload=False,  # Changed to False to avoid metric duplication
        log_level="info"
    )