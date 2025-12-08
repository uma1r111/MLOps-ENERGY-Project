"""
Custom FAISS Retriever with FastEmbed - LangChain Integration
Implements BaseRetriever for LangChain compatibility
"""

import os
import pickle
import logging
from typing import List, Optional

import numpy as np
import faiss
from fastembed import TextEmbedding

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class FAISSFastEmbedRetriever(BaseRetriever):
    """
    Custom LangChain Retriever using FAISS + FastEmbed.

    Integrates with LangChain's retrieval system while using
    FastEmbed for local embeddings (no API calls).
    """

    # Pydantic fields
    index_path: str = Field(description="Path to FAISS index directory")
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    k: int = Field(default=3, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum similarity score"
    )

    # Internal state (not Pydantic fields)
    faiss_index: Optional[faiss.Index] = Field(default=None, exclude=True)
    documents: Optional[List[Document]] = Field(default=None, exclude=True)
    embedding_model: Optional[TextEmbedding] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize the retriever and load index"""
        super().__init__(**data)
        self._load_index()

    def _load_index(self):
        """Load FAISS index and documents"""
        logger.info(f"Loading FAISS index from {self.index_path}")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        # Load FAISS index
        index_file = os.path.join(self.index_path, "index.faiss")
        self.faiss_index = faiss.read_index(index_file)
        logger.info(f"✓ Loaded FAISS index: {self.faiss_index.ntotal} vectors")

        # Load documents
        docs_file = os.path.join(self.index_path, "documents.pkl")
        with open(docs_file, "rb") as f:
            self.documents = pickle.load(f)
        logger.info(f"✓ Loaded {len(self.documents)} documents")

        # Initialize FastEmbed
        logger.info(f"Initializing FastEmbed: {self.embedding_model_name}")
        self.embedding_model = TextEmbedding(model_name=self.embedding_model_name)
        logger.info("✓ FastEmbed initialized")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        This method is called by LangChain's retrieval system.
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")

        # Create query embedding
        query_embedding = np.array(
            list(self.embedding_model.embed([query])), dtype=np.float32
        )

        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, self.k)

        # Get documents
        retrieved_docs = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Convert L2 distance to similarity score (lower is better)
            similarity_score = 1 / (1 + distance)

            # Apply score threshold if set
            if self.score_threshold and similarity_score < self.score_threshold:
                logger.info(f"  Document {i+1} below threshold: {similarity_score:.3f}")
                continue

            doc = self.documents[idx]

            # Add retrieval metadata
            doc.metadata["retrieval_score"] = float(similarity_score)
            doc.metadata["retrieval_rank"] = i + 1

            retrieved_docs.append(doc)
            logger.info(f"  Retrieved doc {i+1}: score={similarity_score:.3f}")

        logger.info(f"✓ Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense (FAISS) and keyword-based retrieval.

    Can be extended to include BM25 or other sparse retrieval methods.
    """

    dense_retriever: FAISSFastEmbedRetriever = Field(
        description="Dense vector retriever"
    )
    k: int = Field(default=3, description="Number of documents to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Hybrid retrieval: dense + keyword (extensible).

        Currently uses dense retrieval, but can be extended
        to combine with BM25 or other methods.
        """
        logger.info(f"Hybrid retrieval for: {query[:50]}...")

        # Dense retrieval
        dense_docs = self.dense_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )

        # TODO: Add BM25 or keyword-based retrieval
        # sparse_docs = self._keyword_search(query)
        # combined_docs = self._rerank(dense_docs, sparse_docs)

        return dense_docs[: self.k]


def create_retriever(
    index_path: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    k: int = 3,
    score_threshold: Optional[float] = None,
) -> FAISSFastEmbedRetriever:
    """
    Factory function to create a retriever.

    Args:
        index_path: Path to FAISS index
        embedding_model: FastEmbed model name
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score

    Returns:
        Configured retriever instance
    """
    return FAISSFastEmbedRetriever(
        index_path=index_path,
        embedding_model_name=embedding_model,
        k=k,
        score_threshold=score_threshold,
    )
