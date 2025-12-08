"""
LangChain RAG Pipeline with Custom Retriever
Implements full RAG chain with LangSmith monitoring
"""
import logging
from typing import Dict, List, Optional, Any
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)


class EnhancedRAGChain:
    """
    Enhanced RAG Chain with LangChain + LangSmith monitoring.
    
    Features:
    - Custom FAISS + FastEmbed retriever
    - Structured prompts with chat history support
    - Automatic token tracking via LangSmith
    - Source document preservation
    """
    
    def __init__(
        self,
        retriever,
        llm_model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        google_api_key: str = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: LangChain retriever instance
            llm_model: Gemini model name
            temperature: LLM temperature
            max_tokens: Max output tokens
            google_api_key: Google API key
            system_prompt: Custom system prompt
        """
        self.retriever = retriever
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model}")
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        
        # Create prompts
        self.qa_prompt = self._create_qa_prompt(system_prompt)
        
        # Build chain
        self.chain = self._build_chain()
        
        logger.info("✓ RAG Chain initialized with LangSmith monitoring")
    
    def _create_qa_prompt(self, system_prompt: Optional[str] = None) -> ChatPromptTemplate:
        """Create QA prompt template"""
        
        default_system = """You are an expert energy and sustainability assistant. 
Use the following pieces of context to answer the question accurately and comprehensively.

Guidelines:
- Base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide detailed, well-structured answers
- Cite specific information from the context when relevant
- If asked about numerical data, be precise

Context:
{context}"""
        
        system_msg = system_prompt or default_system
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}")
        ])
        
        return prompt
    
    def _build_chain(self):
        """Build the RAG chain with LangChain LCEL"""
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.qa_prompt
        )
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=document_chain
        )
        
        return retrieval_chain
    
    def invoke(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the RAG chain.
        
        Args:
            query: User question
            **kwargs: Additional parameters
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Invoking RAG chain for query: {query[:50]}...")
        
        # Run chain (LangSmith automatically tracks this)
        result = self.chain.invoke({"input": query})
        
        # Extract and format response
        response = {
            "answer": result["answer"],
            "source_documents": result.get("context", []),
            "query": query
        }
        
        logger.info(f"✓ Chain completed: {len(response['answer'])} chars")
        
        return response
    
    async def ainvoke(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of invoke"""
        logger.info(f"Async invoking RAG chain: {query[:50]}...")
        
        result = await self.chain.ainvoke({"input": query})
        
        response = {
            "answer": result["answer"],
            "source_documents": result.get("context", []),
            "query": query
        }
        
        return response
    
    def batch(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch process multiple queries.
        
        Args:
            queries: List of questions
            
        Returns:
            List of responses
        """
        logger.info(f"Batch processing {len(queries)} queries...")
        
        inputs = [{"input": q} for q in queries]
        results = self.chain.batch(inputs)
        
        responses = [
            {
                "answer": r["answer"],
                "source_documents": r.get("context", []),
                "query": q
            }
            for r, q in zip(results, queries)
        ]
        
        return responses


class ConversationalRAGChain(EnhancedRAGChain):
    """
    RAG Chain with conversation history support.
    
    Maintains context across multiple turns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: List[Dict[str, str]] = []
    
    def _create_qa_prompt(self, system_prompt: Optional[str] = None) -> ChatPromptTemplate:
        """Create conversational prompt with history"""
        
        default_system = """You are an expert energy and sustainability assistant with conversation history.

Use the provided context and conversation history to answer questions accurately.

Context:
{context}"""
        
        system_msg = system_prompt or default_system
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        return prompt
    
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """Invoke with conversation history"""
        
        # Convert history to message format
        from langchain_core.messages import HumanMessage, AIMessage
        
        chat_history = []
        for msg in self.history:
            if msg["role"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        # Invoke chain with history
        result = self.chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # Extract response
        response = {
            "answer": result["answer"],
            "source_documents": result.get("context", []),
            "query": query
        }
        
        # Update history
        self.history.append({"role": "human", "content": query})
        self.history.append({"role": "assistant", "content": response["answer"]})
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")


def create_rag_chain(
    retriever,
    llm_model: str,
    google_api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    conversational: bool = False,
    system_prompt: Optional[str] = None
):
    """
    Factory function to create a RAG chain.
    
    Args:
        retriever: LangChain retriever
        llm_model: Gemini model name
        google_api_key: Google API key
        temperature: LLM temperature
        max_tokens: Max tokens
        conversational: Use conversational chain
        system_prompt: Custom system prompt
        
    Returns:
        RAG chain instance
    """
    chain_class = ConversationalRAGChain if conversational else EnhancedRAGChain
    
    return chain_class(
        retriever=retriever,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        google_api_key=google_api_key,
        system_prompt=system_prompt
    )