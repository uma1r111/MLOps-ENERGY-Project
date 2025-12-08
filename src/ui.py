"""
Gradio UI for RAG System - LangChain + LangSmith Edition
Beautiful, modern interface with conversation history and source viewing
"""

import sys
import time
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

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
    LANGSMITH_PROJECT,
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import create_rag_chain, ConversationalRAGChain
from src.guardrails import GuardrailEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global components
rag_chain: Optional[ConversationalRAGChain] = None
guardrail_engine: Optional[GuardrailEngine] = None


def load_rag_system():
    """Load RAG system"""
    global rag_chain, guardrail_engine

    logger.info("Loading RAG system...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please run 'python src/rag/ingest.py' first."
        )

    # Create retriever
    retriever = create_retriever(
        index_path=INDEX_PATH, embedding_model=FASTEMBED_MODEL, k=TOP_K
    )

    # Create conversational RAG chain
    rag_chain = create_rag_chain(
        retriever=retriever,
        llm_model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        conversational=True,
    )

    # Load guardrails
    try:
        guardrail_engine = GuardrailEngine(
            config_path="config/guardrails/guardrails_config.json"
        )
        logger.info("✓ Guardrails initialized")
    except Exception as e:
        logger.warning(f"⚠️  Guardrails disabled: {str(e)}")
        guardrail_engine = None

    logger.info("✓ RAG system loaded")
    return True


def format_source_display(sources: List) -> str:
    """Format source documents for display"""
    if not sources:
        return "No sources available"

    formatted = "### 📚 Source Documents\n\n"

    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("retrieval_score", 0)
        page = doc.metadata.get("page")

        formatted += f"**Source {i}** "
        if page:
            formatted += f"({source}, Page {page}) "
        else:
            formatted += f"({source}) "

        formatted += f"| Score: {score:.3f}\n\n"
        formatted += f"```\n{doc.page_content[:400]}...\n```\n\n"
        formatted += "---\n\n"

    return formatted


def query_rag_system(
    question: str,
    history: List[Tuple[str, str]],
    show_sources: bool,
    enable_guardrails: bool,
) -> Tuple[List[Tuple[str, str]], str, str]:
    """
    Query the RAG system.

    Returns:
        (updated_history, sources_display, stats)
    """
    if not question.strip():
        return history, "", "Please enter a question."

    if rag_chain is None:
        return history, "", "❌ RAG system not initialized!"

    start_time = time.time()

    try:
        sanitized_query = question

        # Input validation
        if enable_guardrails and guardrail_engine:
            logger.info("🛡️ Running input validation...")
            input_validation = guardrail_engine.validate_input(question)

            if not input_validation["passed"]:
                violations = input_validation.get("violations", [])
                error_msg = "❌ **Query Rejected by Guardrails**\n\n"
                for v in violations:
                    error_msg += (
                        f"- {v.get('type', 'Unknown')}: {v.get('message', '')}\n"
                    )

                history.append((question, error_msg))
                return history, "", error_msg

            sanitized_query = input_validation.get("sanitized_input", question)

        # Query RAG chain
        logger.info(f"Querying: {sanitized_query[:50]}...")
        result = rag_chain.invoke(sanitized_query)

        answer = result["answer"]
        sources = result.get("source_documents", [])

        # Output moderation
        if enable_guardrails and guardrail_engine:
            logger.info("🛡️ Running output moderation...")
            output_mod = guardrail_engine.moderate_output(answer)

            if not output_mod["passed"]:
                violations = output_mod.get("violations", [])
                blocking = [v for v in violations if v.get("severity") != "WARNING"]

                if blocking:
                    answer = (
                        "⚠️ I apologize, but I cannot provide this response as it "
                        "violates content safety guidelines. Please rephrase your question."
                    )

        # Format response
        latency = time.time() - start_time

        # Update history
        history.append((question, answer))

        # Format sources
        sources_display = format_source_display(sources) if show_sources else ""

        # Stats
        stats = f"""
### 📊 Query Statistics

- **Latency:** {latency:.2f}s
- **Sources Retrieved:** {len(sources)}
- **Model:** {GEMINI_MODEL}
- **Embeddings:** {FASTEMBED_MODEL}
- **LangSmith:** {'✓ Active' if LANGSMITH_API_KEY else '✗ Disabled'}
- **Guardrails:** {'✓ Enabled' if (enable_guardrails and guardrail_engine) else '✗ Disabled'}
"""

        if LANGSMITH_API_KEY:
            stats += f"\n**LangSmith Project:** [{LANGSMITH_PROJECT}](https://smith.langchain.com)\n"

        return history, sources_display, stats

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(f"Query failed: {e}", exc_info=True)
        history.append((question, error_msg))
        return history, "", error_msg


def clear_conversation():
    """Clear conversation history"""
    if rag_chain:
        rag_chain.clear_history()
    return [], "", "Conversation cleared."


def create_ui():
    """Create Gradio UI"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .chat-message {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .source-box {
        background-color: #f7f7f7;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .stats-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(), css=custom_css, title="Energy RAG Assistant"
    ) as demo:

        gr.Markdown(
            """
        # 🌟 Energy RAG Assistant
        ### Powered by LangChain + FastEmbed + Google Gemini + LangSmith
        
        Ask questions about energy, sustainability, and related topics. The system retrieves relevant 
        documents and generates informed answers with full conversation memory.
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    show_label=True,
                    avatar_images=("👤", "🤖"),
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ask a question about energy, solar panels, sustainability...",
                        label="Your Question",
                        lines=2,
                        scale=4,
                    )
                    submit_btn = gr.Button("🚀 Ask", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    show_sources_checkbox = gr.Checkbox(
                        label="Show Sources", value=True
                    )
                    guardrails_checkbox = gr.Checkbox(
                        label="Enable Guardrails", value=True
                    )

            with gr.Column(scale=1):
                # Stats and sources
                stats_display = gr.Markdown(
                    label="📊 Statistics", value="Submit a query to see statistics."
                )

                sources_display = gr.Markdown(
                    label="📚 Sources", value="Sources will appear here after querying."
                )

        # System info footer
        gr.Markdown(
            f"""
        ---
        ### ℹ️ System Information
        
        - **LLM:** {GEMINI_MODEL}
        - **Embeddings:** {FASTEMBED_MODEL} (local)
        - **LangSmith Monitoring:** {'✓ Enabled' if LANGSMITH_API_KEY else '✗ Disabled'}
        - **Guardrails:** {'✓ Available' if guardrail_engine else '✗ Not Available'}
        
        **Features:**
        - 🧠 Conversational memory
        - 📚 Source document retrieval
        - 🛡️ Input/output guardrails
        - 📊 LangSmith token tracking
        - ⚡ Local embeddings (FastEmbed)
        """
        )

        # Event handlers
        submit_btn.click(
            fn=query_rag_system,
            inputs=[
                question_input,
                chatbot,
                show_sources_checkbox,
                guardrails_checkbox,
            ],
            outputs=[chatbot, sources_display, stats_display],
        ).then(fn=lambda: "", outputs=[question_input])

        question_input.submit(
            fn=query_rag_system,
            inputs=[
                question_input,
                chatbot,
                show_sources_checkbox,
                guardrails_checkbox,
            ],
            outputs=[chatbot, sources_display, stats_display],
        ).then(fn=lambda: "", outputs=[question_input])

        clear_btn.click(
            fn=clear_conversation, outputs=[chatbot, sources_display, stats_display]
        )

    return demo


def main():
    """Main function"""
    try:
        # Load system
        logger.info("=" * 60)
        logger.info("Starting Energy RAG Assistant UI")
        logger.info("=" * 60)

        load_rag_system()

        logger.info("=" * 60)
        logger.info("✅ System Ready!")
        logger.info("=" * 60)

        # Create and launch UI
        demo = create_ui()

        demo.launch(
            server_name="0.0.0.0", server_port=7860, share=False, show_error=True
        )

    except Exception as e:
        logger.error(f"Failed to start UI: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
