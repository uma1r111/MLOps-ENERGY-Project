"""
Document Ingestion Pipeline - LangChain + FastEmbed + Google Gemini
Loads documents, chunks them with LangChain, creates embeddings with FastEmbed
"""
import os
import sys
from pathlib import Path
from typing import List
import logging
import pickle

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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
    """
    Enhanced document ingestion with LangChain loaders.
    
    Features:
    - Multiple document formats (PDF, TXT, MD, DOCX)
    - Smart chunking with RecursiveCharacterTextSplitter
    - FastEmbed for local embeddings
    - FAISS for vector storage
    - LangChain Document format
    """
    
    def __init__(
        self,
        docs_path: str = DOCUMENTS_PATH,
        index_path: str = INDEX_PATH,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """Initialize the ingestion pipeline"""
        self.docs_path = docs_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize FastEmbed
        logger.info(f"Initializing FastEmbed: {FASTEMBED_MODEL}")
        self.embedding_model = TextEmbedding(model_name=FASTEMBED_MODEL)
        logger.info("✓ FastEmbed initialized (running locally)")
        
        logger.info(f"Initialized DocumentIngestion with docs_path={docs_path}")
    
    def load_documents(self) -> List[Document]:
        """
        Load documents using LangChain loaders.
        
        Supports: PDF, TXT, MD, DOCX
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents path not found: {self.docs_path}")
        
        files = [f for f in os.listdir(self.docs_path) if not f.startswith('.')]
        if not files:
            raise ValueError(f"No documents found in {self.docs_path}")
        
        logger.info(f"Found {len(files)} files in {self.docs_path}")
        logger.info("=" * 60)
        
        for filename in files:
            filepath = os.path.join(self.docs_path, filename)
            
            try:
                if filename.endswith('.pdf'):
                    logger.info(f"📄 Loading PDF: {filename}")
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded {len(docs)} pages")
                
                elif filename.endswith('.txt'):
                    logger.info(f"📝 Loading TXT: {filename}")
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded {len(docs)} documents")
                
                elif filename.endswith('.md'):
                    logger.info(f"📋 Loading Markdown: {filename}")
                    loader = UnstructuredMarkdownLoader(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded {len(docs)} documents")
                
                else:
                    logger.warning(f"  ⚠️  Skipping unsupported file: {filename}")
            
            except Exception as e:
                logger.error(f"  ✗ Error loading {filename}: {str(e)}")
                continue
        
        if not documents:
            raise ValueError("No documents were successfully loaded!")
        
        logger.info("=" * 60)
        logger.info(f"✓ Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        # Create text splitter with optimal settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(f"  Avg chunk size: {np.mean([c.metadata['chunk_size'] for c in chunks]):.0f} chars")
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings using FastEmbed (local).
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with FastEmbed...")
        
        # FastEmbed returns a generator
        embeddings = list(self.embedding_model.embed(texts))
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        logger.info(f"✓ Generated embeddings: shape={embeddings_array.shape}")
        return embeddings_array
    
    def create_index(self) -> tuple:
        """
        Create FAISS index with LangChain documents.
        
        Returns:
            Tuple of (FAISS index, document chunks)
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Document Ingestion Pipeline")
        logger.info("LangChain + FastEmbed + Google Gemini")
        logger.info("=" * 60)
        
        # Step 1: Load documents
        documents = self.load_documents()
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Extract texts
        texts = [doc.page_content for doc in chunks]
        
        # Step 4: Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Step 5: Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for exact search
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logger.info(f"✓ FAISS index created:")
        logger.info(f"  - Vectors: {index.ntotal}")
        logger.info(f"  - Dimensions: {dimension}")
        
        # Step 6: Save index and documents
        logger.info(f"Saving index to {self.index_path}...")
        os.makedirs(self.index_path, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(index, os.path.join(self.index_path, "index.faiss"))
            
            # Save LangChain documents
            with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
                pickle.dump(chunks, f)
            
            # Save metadata
            metadata = {
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "embedding_model": FASTEMBED_MODEL,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "dimension": dimension
            }
            
            with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✓ Index saved to {self.index_path}")
        
        except Exception as e:
            logger.error(f"✗ Error saving index: {str(e)}")
            raise
        
        logger.info("=" * 60)
        logger.info("✅ Ingestion Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        return index, chunks
    
    def test_retrieval(
        self,
        index,
        chunks: List[Document],
        query: str = "What is energy efficiency?"
    ):
        """
        Test retrieval with a sample query.
        
        Args:
            index: FAISS index
            chunks: List of document chunks
            query: Test query string
        """
        logger.info(f"\n🔍 Testing retrieval with query: '{query}'")
        
        # Create query embedding
        query_embedding = np.array(
            list(self.embedding_model.embed([query])),
            dtype=np.float32
        )
        
        # Search
        k = 3
        distances, indices = index.search(query_embedding, k)
        
        logger.info(f"Retrieved {len(indices[0])} documents:")
        logger.info("-" * 60)
        
        for i, idx in enumerate(indices[0], 1):
            doc = chunks[idx]
            score = 1 / (1 + distances[0][i-1])  # Convert L2 to similarity
            
            logger.info(f"\n📄 Result {i} (score: {score:.3f})")
            logger.info(f"Source: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"Preview: {doc.page_content[:200]}...")
        
        logger.info("-" * 60)


def main():
    """Main function to run ingestion pipeline"""
    try:
        # Initialize ingestion
        ingestion = DocumentIngestion()
        
        # Create index
        index, chunks = ingestion.create_index()
        
        # Test retrieval
        ingestion.test_retrieval(index, chunks)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Your RAG ingestion pipeline is ready!")
        print("=" * 60)
        print(f"📁 FAISS index saved to: {INDEX_PATH}")
        print(f"📊 Total chunks: {len(chunks)}")
        print(f"🔧 Embedding model: {FASTEMBED_MODEL}")
        print("\n🚀 Next steps:")
        print("  1. Run the API: python src/app.py")
        print("  2. Or run the UI: python src/ui.py")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()