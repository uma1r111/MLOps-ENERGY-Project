"""
Streamlit UI Client for FastAPI RAG System - Professional Chatbot Interface
Enhanced with modern, clean design and conversational UX
"""
<<<<<<< Updated upstream

import sys
import time
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
=======
import streamlit as st
import requests
import json
import os
import sys
from typing import List, Dict
from datetime import datetime
>>>>>>> Stashed changes

# Adjust path to ensure config is imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.config import LANGSMITH_API_KEY, LANGSMITH_PROJECT, GEMINI_MODEL, FASTEMBED_MODEL

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000"
GRAFANA_BASE_URL = "http://localhost:3000"
PROMETHEUS_BASE_URL = "http://localhost:9090"
LANGSMITH_BASE_URL = "https://smith.langchain.com"

<<<<<<< Updated upstream
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
=======
# --- CUSTOM CSS FOR MODERN CHATBOT LOOK ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin-bottom: 24px;
        max-height: 500px;
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
    
    /* User message bubble */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 12px 0;
        margin-left: 20%;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    /* Assistant message bubble */
    .assistant-message {
        background: #f7f9fc;
        color: #2d3748;
        padding: 14px 20px;
        border-radius: 20px 20px 20px 4px;
        margin: 12px 0;
        margin-right: 20%;
        max-width: 75%;
        float: left;
        clear: both;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        animation: slideInLeft 0.3s ease-out;
        word-wrap: break-word;
        line-height: 1.7;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Input area styling */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 32px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,0.1) !important;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin: 10px 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Source card styling */
    .source-card {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        background: #48bb78;
        color: white;
        box-shadow: 0 2px 8px rgba(72, 187, 120, 0.3);
    }
    
    /* Results container */
    .results-container {
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-top: 24px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid rgba(255,255,255,0.1);
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clear floats */
    .chat-container::after {
        content: "";
        display: table;
        clear: both;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Warning/Error boxes */
    .stAlert {
        border-radius: 12px !important;
        border-left-width: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_initial_data():
    """Fetches variants and health status on startup."""
    try:
        variants_response = requests.get(f"{API_BASE_URL}/variants", timeout=5)
        variants_response.raise_for_status()
        variants_list = variants_response.json().get('variants', [])
>>>>>>> Stashed changes

        variant_details = {v['id']: v for v in variants_list}
        variant_ids = ["Auto Assign"] + [v['id'] for v in variants_list]

        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        api_ready = health_response.status_code == 200

        return variant_ids, api_ready, variant_details

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        return ["Auto Assign"], False, {}

def format_source_display(sources: List) -> str:
    """Formats source documents in a clean card layout."""
    if not sources:
<<<<<<< Updated upstream
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

=======
        return '<div style="text-align: center; padding: 30px; color: #718096;">📄 No source documents found</div>'

    formatted = '<div style="margin-top: 16px;">'
    
    for i, doc in enumerate(sources, 1):
        source = doc.get('source', 'Unknown')
        score = doc.get('retrieval_score', 0)
        page = doc.get('page')
        content = doc.get('content', '').strip()
        preview = content[:250] + "..." if len(content) > 250 else content

        page_info = f" • Page {page}" if page else ""
        
        formatted += f'''
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #2d3748; font-size: 14px;">📄 Source {i}</span>
                <span style="background: #edf2f7; padding: 5px 12px; border-radius: 10px; font-size: 12px; color: #4a5568; font-weight: 600;">
                    Score: {score:.3f}
                </span>
            </div>
            <div style="font-size: 12px; color: #718096; margin-bottom: 10px; font-weight: 500;">
                {source}{page_info}
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; font-size: 13px; color: #4a5568; line-height: 1.7; border: 1px solid #e2e8f0;">
                {preview}
            </div>
        </div>
        '''
    
    formatted += '</div>'
>>>>>>> Stashed changes
    return formatted

def submit_feedback(query, variant_id, score, comment):
    """Submits user feedback to the API."""
    if not variant_id or variant_id == "N/A":
        st.error("⚠️ Cannot submit feedback without a query result.")
        return

<<<<<<< Updated upstream
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
=======
    feedback_payload = {
        "query": query,
        "variant_id": variant_id,
        "satisfaction_score": float(score),
        "comment": comment if comment else None
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback", 
            json=feedback_payload,
            timeout=10
        )
        response.raise_for_status()
        st.success("✅ Thank you for your feedback!")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Feedback Error: {e}")

# --- MAIN UI ---

def main_ui():
    """Main Streamlit UI with modern chatbot design."""
    st.set_page_config(
        layout="wide", 
        page_title="Energy RAG Assistant",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_data" not in st.session_state:
        st.session_state.last_query_data = {"query": "N/A", "variant_id": "N/A", "variant_name": "N/A"}
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    variant_ids, api_ready, variant_details = fetch_initial_data()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("# ⚡ Energy RAG")
        st.markdown("### AI-Powered Energy Assistant")
        
        status_color = "#48bb78" if api_ready else "#f56565"
        status_text = "Connected" if api_ready else "Disconnected"
        st.markdown(f'<div class="status-badge" style="background: {status_color};">{status_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings Section
        with st.expander("⚙️ Query Settings", expanded=False):
            user_id_input = st.text_input("👤 User ID (Optional)", placeholder="user123", help="For consistent variant assignment")
            variant_selection = st.selectbox("🎯 Force Variant", variant_ids, help="Override automatic variant selection")
            include_sources = st.checkbox("📚 Show Sources", value=True, help="Display source documents")
        
        st.markdown("---")
        
        # A/B Variants Info
        with st.expander("🧪 A/B Test Variants", expanded=False):
            if variant_details:
                for v in variant_details.values():
                    st.markdown(f"""
                    **{v['name']}**  
                    Traffic: {v['traffic_percentage']}%  
                    Temp: {v['temperature']} | Tokens: {v['max_tokens']}
                    """)
        
        st.markdown("---")
        
        # System Info
        st.markdown("### 📊 System Info")
        st.caption(f"**Model:** {GEMINI_MODEL}")
        st.caption(f"**Embeddings:** {FASTEMBED_MODEL}")
        
        st.markdown("---")
        
        # Monitoring Links
        st.markdown("### 🔍 Monitoring")
        if st.button("📈 Grafana Dashboard", use_container_width=True):
            st.write(f"[Open Dashboard]({GRAFANA_BASE_URL})")
        if st.button("📉 Prometheus", use_container_width=True):
            st.write(f"[View Metrics]({PROMETHEUS_BASE_URL})")
        if LANGSMITH_API_KEY:
            if st.button("🔗 LangSmith Traces", use_container_width=True):
                st.write(f"[View Project]({LANGSMITH_BASE_URL})")
        
        st.markdown("---")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.show_results = False
            st.session_state.last_result = None
            st.session_state.input_key += 1  # Also clear input field
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by Google Gemini & LangChain")

    # --- MAIN CONTENT ---
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <h1 style="color: white; font-size: 48px; font-weight: 700; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">⚡ Energy RAG Assistant</h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 18px; margin-top: 12px; font-weight: 500;">Ask me anything about energy systems, sustainability, and power generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #718096;">
            <div style="font-size: 64px; margin-bottom: 20px;">💬</div>
            <div style="font-size: 22px; font-weight: 600; color: #2d3748; margin-bottom: 10px;">Start a Conversation</div>
            <div style="font-size: 15px; color: #718096; line-height: 1.6;">Ask me about renewable energy, power systems, or energy efficiency!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Query Input Area
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query_input = st.text_area(
            "Your Question",
            placeholder="💭 Type your question here... (e.g., How can I reduce my energy consumption?)",
            height=100,
            key=f"query_input_field_{st.session_state.input_key}",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
        submit_button = st.button("🚀 Send", use_container_width=True, type="primary")

    # --- HANDLE QUERY SUBMISSION ---
    if submit_button and query_input and api_ready:
        if not query_input.strip():
            st.warning("⚠️ Please enter a question.")
            st.stop()
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query_input})

        # Get settings from sidebar
        variant_id_to_send = None if variant_selection == "Auto Assign" else variant_selection

        with st.spinner("🤔 Thinking..."):
            try:
                # API Request
                payload = {
                    "question": query_input,
                    "include_sources": include_sources,
                    "variant_id": variant_id_to_send,
                    "user_id": user_id_input if user_id_input else None
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()

                # Process response
                answer = result.get("answer", "I apologize, but I couldn't generate an answer.")
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Store query data and result
                st.session_state.last_query_data = {
                    "query": query_input,
                    "variant_id": result.get("variant_id", "N/A"),
                    "variant_name": result.get("variant_name", "N/A"),
                }
                st.session_state.last_result = result
                st.session_state.show_results = True
                
                # Clear the input by incrementing the key (forces widget recreation)
                st.session_state.input_key += 1
                
                # Rerun to update UI
                st.rerun()
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                st.error(error_msg)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                st.error(error_msg)

    elif submit_button and not api_ready:
        st.error("❌ Cannot send message. API is not connected.")

    # --- DISPLAY RESULTS SECTION ---
    if st.session_state.show_results and st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.markdown("### 📊 Query Details")
            
            # Metrics in cards
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #718096; margin-bottom: 6px; font-weight: 500;">Variant Used</div>
                <div style="font-size: 20px; font-weight: 600; color: #2d3748;">{result.get('variant_name', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #718096; margin-bottom: 6px; font-weight: 500;">Response Time</div>
                <div style="font-size: 20px; font-weight: 600; color: #2d3748;">{result.get('latency', 0):.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #718096; margin-bottom: 6px; font-weight: 500;">Tokens Used</div>
                <div style="font-size: 20px; font-weight: 600; color: #2d3748;">
                    {result.get('tokens_used', {}).get('input', 0)} in / {result.get('tokens_used', {}).get('output', 0)} out
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #718096; margin-bottom: 6px; font-weight: 500;">Estimated Cost</div>
                <div style="font-size: 20px; font-weight: 600; color: #2d3748;">${result.get('estimated_cost', 0):.6f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Feedback Section
            st.markdown("---")
            st.markdown("### 💬 Rate This Response")
            
            feedback_score = st.slider("How satisfied are you?", 1, 5, 5, key="feedback_score_slider")
            feedback_comment = st.text_area("Additional Comments (Optional)", height=80, key="feedback_comment_input", placeholder="Tell us what you think...")

            if st.button("📤 Submit Feedback", type="secondary", use_container_width=True):
                submit_feedback(
                    st.session_state.last_query_data["query"],
                    st.session_state.last_query_data["variant_id"],
                    feedback_score,
                    feedback_comment
                )

        with col_b:
            if include_sources and result.get("sources"):
                st.markdown("### 📚 Source Documents")
                st.markdown(format_source_display(result.get("sources", [])), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 60px 20px; color: #718096;">
                    <div style="font-size: 56px; margin-bottom: 16px;">📄</div>
                    <div style="font-size: 16px; font-weight: 500;">No sources to display</div>
                    <div style="font-size: 13px; color: #a0aec0; margin-top: 8px;">Enable "Show Sources" in settings</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main_ui()
>>>>>>> Stashed changes
