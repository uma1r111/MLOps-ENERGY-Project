# RAG System Architecture

## System Architecture Diagram
```mermaid
graph TB
    subgraph "Data Sources"
        A[PDF Documents]
        B[Text Files]
    end
    
    subgraph "Ingestion Pipeline"
        C[Document Loader]
        D[Text Chunker]
        E[FastEmbed<br/>Local Embeddings]
        F[FAISS Index]
    end
    
    subgraph "Storage"
        G[(FAISS Vector DB)]
        H[(Document Store<br/>Pickle)]
    end
    
    subgraph "API Layer"
        I[FastAPI Server]
        J[Query Endpoint]
        K[Health Check]
        L[Metrics]
    end
    
    subgraph "RAG Pipeline"
        M[Query Embedding<br/>FastEmbed]
        N[Vector Search<br/>FAISS]
        O[Context Retrieval]
        P[LLM Generation<br/>Google Gemini]
    end
    
    subgraph "Monitoring"
        Q[Prometheus Metrics]
        R[Logging]
    end
    
    subgraph "Client"
        S[User Query]
        T[Response]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    D --> H
    
    S --> I
    I --> J
    J --> M
    M --> N
    N --> O
    O --> P
    P --> T
    
    N --> G
    O --> H
    
    J --> Q
    J --> R
    I --> K
    I --> L
    
    style E fill:#90EE90
    style P fill:#87CEEB
    style G fill:#FFD700
    style I fill:#FF6B6B
```

## Data Flow Diagram
```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI Server
    participant FE as FastEmbed
    participant FAISS as FAISS Index
    participant Docs as Document Store
    participant Gemini as Google Gemini LLM
    
    Note over User,Gemini: Ingestion Phase (One-time)
    User->>API: Upload Documents
    API->>FE: Generate Embeddings
    FE-->>API: Vector Embeddings
    API->>FAISS: Store Vectors
    API->>Docs: Store Documents
    
    Note over User,Gemini: Query Phase (Runtime)
    User->>API: POST /query {"question": "..."}
    API->>FE: Embed Query
    FE-->>API: Query Vector
    API->>FAISS: Search Similar Vectors
    FAISS-->>API: Top-K Indices
    API->>Docs: Retrieve Documents
    Docs-->>API: Context Documents
    API->>Gemini: Generate Answer (Context + Query)
    Gemini-->>API: Generated Answer
    API-->>User: Response + Sources + Metadata
```

## Component Details

### 1. **Ingestion Pipeline**
- **Input**: PDF/TXT files from `data/documents/`
- **Processing**:
  - Load documents using LangChain loaders
  - Chunk text (1000 chars, 200 overlap)
  - Generate embeddings using FastEmbed (local)
  - Build FAISS index
  - Save index and documents
- **Output**: `data/faiss_index/` (index.faiss, documents.pkl)

### 2. **Inference API**
- **Endpoints**:
  - `POST /query`: Main RAG query endpoint
  - `GET /health`: Health check
  - `GET /metrics`: Prometheus metrics
- **Process**:
  1. Receive user query
  2. Embed query using FastEmbed
  3. Search FAISS for top-K similar documents
  4. Retrieve context documents
  5. Generate answer using Google Gemini
  6. Return answer with sources

### 3. **Technology Stack**
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5) - Local, no API key
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **LLM**: Google Gemini 2.0 Flash - Free tier
- **API**: FastAPI with Pydantic v2
- **Monitoring**: Prometheus metrics, structured logging

### 4. **Storage Architecture**
```
data/
├── documents/          # Source documents (input)
│   ├── *.pdf
│   └── *.txt
└── faiss_index/       # Generated index (output)
    ├── index.faiss    # Vector embeddings
    └── documents.pkl  # Document metadata
```