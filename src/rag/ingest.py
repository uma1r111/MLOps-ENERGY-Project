"""
Document Ingestion Pipeline - FastEmbed + Google Gemini
Loads documents, chunks them, creates embeddings with FastEmbed (local), and builds FAISS index.
"""
import os
import sys
from pathlib import Path
from typing import List
import logging
import pickle

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from fastembed import TextEmbedding
import numpy as np
import faiss

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rag.config import (
    DOCUMENTS_PATH,
    INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FASTEMBED_MODEL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handles document loading, chunking, and indexing with FastEmbed"""
    
    def __init__(
        self,
        docs_path: str = DOCUMENTS_PATH,
        index_path: str = INDEX_PATH,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            docs_path: Path to documents folder
            index_path: Path to save FAISS index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.docs_path = docs_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize FastEmbed (local, no API key needed)
        logger.info(f"Initializing FastEmbed: {FASTEMBED_MODEL}")
        self.embedding_model = TextEmbedding(model_name=FASTEMBED_MODEL)
        logger.info("✓ FastEmbed initialized (running locally)")
        
        logger.info(f"Initialized DocumentIngestion with docs_path={docs_path}")
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the documents folder.
        Supports PDF and TXT files.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents path not found: {self.docs_path}")
        
        files = os.listdir(self.docs_path)
        if not files:
            raise ValueError(f"No documents found in {self.docs_path}")
        
        logger.info(f"Found {len(files)} files in {self.docs_path}")
        
        for filename in files:
            filepath = os.path.join(self.docs_path, filename)
            
            try:
                if filename.endswith('.pdf'):
                    logger.info(f"Loading PDF: {filename}")
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded {len(docs)} pages from {filename}")
                
                elif filename.endswith('.txt'):
                    logger.info(f"Loading TXT: {filename}")
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded {filename}")
                
                else:
                    logger.warning(f"  ✗ Skipping unsupported file: {filename}")
            
            except Exception as e:
                logger.error(f"  ✗ Error loading {filename}: {str(e)}")
                continue
        
        if not documents:
            raise ValueError("No documents were successfully loaded!")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings using FastEmbed (local, fast).
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with FastEmbed...")
        
        # FastEmbed returns a generator, convert to list of arrays
        embeddings = list(self.embedding_model.embed(texts))
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        logger.info(f"✓ Generated embeddings: shape={embeddings_array.shape}")
        return embeddings_array
    
    def create_index(self) -> tuple:
        """
        Create FAISS vector store index from documents using FastEmbed.
        
        Returns:
            Tuple of (FAISS index, document chunks)
        """
        logger.info("=" * 60)
        logger.info("Starting Document Ingestion Pipeline")
        logger.info("FastEmbed (local) + Google Gemini")
        logger.info("=" * 60)
        
        # Step 1: Load documents
        documents = self.load_documents()
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Extract texts for embedding
        texts = [doc.page_content for doc in chunks]
        
        # Step 4: Create embeddings with FastEmbed
        embeddings = self.create_embeddings(texts)
        
        # Step 5: Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for exact search
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info(f"✓ FAISS index created: {index.ntotal} vectors, {dimension} dimensions")
        
        # Step 6: Save index and documents
        logger.info(f"Saving index to {self.index_path}...")
        os.makedirs(self.index_path, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(index, os.path.join(self.index_path, "index.faiss"))
            
            # Save documents metadata
            with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
                pickle.dump(chunks, f)
            
            logger.info(f"✓ Index and documents saved to {self.index_path}")
        except Exception as e:
            logger.error(f"✗ Error saving index: {str(e)}")
            raise
        
        logger.info("=" * 60)
        logger.info("Ingestion Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        return index, chunks
    
    def test_retrieval(self, index, chunks: List[Document], query: str = "What is energy efficiency?"):
        """
        Test the retrieval system with a sample query.
        
        Args:
            index: FAISS index
            chunks: List of document chunks
            query: Test query string
        """
        logger.info(f"\nTesting retrieval with query: '{query}'")
        
        # Create query embedding
        query_embedding = np.array(list(self.embedding_model.embed([query])), dtype=np.float32)
        
        # Search
        k = 3
        distances, indices = index.search(query_embedding, k)
        
        logger.info(f"Retrieved {len(indices[0])} documents:")
        for i, idx in enumerate(indices[0], 1):
            doc = chunks[idx]
            logger.info(f"\n--- Result {i} (distance: {distances[0][i-1]:.4f}) ---")
            logger.info(f"Source: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"Content preview: {doc.page_content[:200]}...")


def main():
    """Main function to run ingestion pipeline"""
    try:
        # Initialize ingestion
        ingestion = DocumentIngestion()
        
        # Create index
        index, chunks = ingestion.create_index()
        
        # Test retrieval
        ingestion.test_retrieval(index, chunks)
        
        print("\n✅ SUCCESS! Your RAG ingestion pipeline is ready!")
        print(f"📁 FAISS index saved to: {INDEX_PATH}")
        print("🚀 Next step: Run the API with 'python src/app.py'")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()