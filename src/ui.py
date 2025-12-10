"""
Streamlit UI Client for FastAPI RAG System - Professional Chatbot Interface
Enhanced with modern, clean design and conversational UX
"""

import streamlit as st
import requests
import os
import sys
from typing import List

# Adjust path to ensure config is imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.config import LANGSMITH_API_KEY, GEMINI_MODEL, FASTEMBED_MODEL

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000"
GRAFANA_BASE_URL = "http://localhost:3000"
PROMETHEUS_BASE_URL = "http://localhost:9090"
LANGSMITH_BASE_URL = "https://smith.langchain.com"


# --- CUSTOM CSS FOR MODERN CHATBOT LOOK ---
def inject_custom_css():
    st.markdown(
        """
    <style>

    * { font-family: 'Inter', sans-serif; }

    /* === BACKGROUND IMAGE + DARK OVERLAY === */
    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.94)),
            url("background.jpg")
            center/cover no-repeat fixed !important;
    }


    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }

    /* === TITLE === */
    h1, h2, h3 {
        font-weight: 700 !important;
        color: #e8f6ff !important;
        text-shadow: 0 0 24px rgba(0, 200, 255, 0.55);
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: rgba(5, 8, 12, 0.92) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,150,255,0.25);
    }
    section[data-testid="stSidebar"] * {
        color: #e5f4ff !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #00aaff, #00e0ff) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        filter: brightness(1.15);
    }

    /* === CHAT === */
    .chat-container {
        background: rgba(20,25,32,0.75);
        border-radius: 18px;
        padding: 22px;
        border: 1px solid rgba(0,150,255,0.3);
        backdrop-filter: blur(8px);
        box-shadow: 0 0 24px rgba(0,150,255,0.18);
        max-height: 500px;
        overflow-y: auto;
    }

    /* User */
    .user-message {
        background: rgba(0,150,255,0.9) !important;
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 20px;
        margin: 10px 0;
        float: right;
        max-width: 70%;
        box-shadow: 0 0 18px rgba(0,150,255,0.55);
    }

    /* Assistant */
    .assistant-message {
        background: rgba(12,18,24,0.85) !important;
        color: #e8f6ff;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 20px;
        margin: 10px 0;
        float: left;
        max-width: 70%;
        border-left: 4px solid #00caff;
        box-shadow: 0 0 12px rgba(0,150,255,0.35);
    }

    /* Input */
    .stTextArea textarea {
        background: rgba(10,14,20,0.85) !important;
        color: #eaf7ff !important;
        border: 1px solid rgba(0,150,255,0.25) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        backdrop-filter: blur(6px);
    }

    .stTextArea textarea:focus {
        border: 1px solid #00caff !important;
        box-shadow: 0 0 16px rgba(0,150,255,0.45);
    }

    /* Hide default UI items */
    #MainMenu, header, footer {visibility: hidden;}

    </style>
    """,
        unsafe_allow_html=True,
    )


# --- HELPER FUNCTIONS ---


@st.cache_data(ttl=3600)
def fetch_initial_data():
    """Fetches variants and health status on startup."""
    try:
        variants_response = requests.get(f"{API_BASE_URL}/variants", timeout=5)
        variants_response.raise_for_status()
        variants_list = variants_response.json().get("variants", [])

        variant_details = {v["id"]: v for v in variants_list}
        variant_ids = ["Auto Assign"] + [v["id"] for v in variants_list]

        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        api_ready = health_response.status_code == 200

        return variant_ids, api_ready, variant_details

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
        return ["Auto Assign"], False, {}


def format_source_display(sources: List) -> str:
    """Formats source documents in a clean card layout."""
    if not sources:
        return '<div style="text-align: center; padding: 30px; color: #a0aec0;">📄 No source documents found</div>'

    formatted = '<div style="margin-top: 16px;">'

    for i, doc in enumerate(sources, 1):
        source = doc.get("source", "Unknown")
        score = doc.get("retrieval_score", 0)
        page = doc.get("page")
        content = doc.get("content", "").strip()
        preview = content[:250] + "..." if len(content) > 250 else content

        page_info = f" • Page {page}" if page else ""

        formatted += f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #e2e8f0; font-size: 14px;">📄 Source {i}</span>
                <span style="background: rgba(102, 126, 234, 0.2); padding: 5px 12px; border-radius: 10px; font-size: 12px; color: #a0d0f4; font-weight: 600;">
                    Score: {score:.3f}
                </span>
            </div>
            <div style="font-size: 12px; color: #a0aec0; margin-bottom: 10px; font-weight: 500;">
                {source}{page_info}
            </div>
            <div style="background: rgba(26, 32, 44, 0.6); padding: 12px; border-radius: 8px; font-size: 13px; color: #cbd5e0; line-height: 1.7; border: 1px solid rgba(102, 126, 234, 0.2);">
                {preview}
            </div>
        </div>
        """

    formatted += "</div>"
    return formatted


def submit_feedback(query, variant_id, score, comment):
    """Submits user feedback to the API."""
    if not variant_id or variant_id == "N/A":
        st.error("⚠️ Cannot submit feedback without a query result.")
        return

    feedback_payload = {
        "query": query,
        "variant_id": variant_id,
        "satisfaction_score": float(score),
        "comment": comment if comment else None,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback", json=feedback_payload, timeout=10
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
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    inject_custom_css()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_data" not in st.session_state:
        st.session_state.last_query_data = {
            "query": "N/A",
            "variant_id": "N/A",
            "variant_name": "N/A",
        }
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
        st.markdown(
            f'<div class="status-badge" style="background: {status_color};">{status_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Settings Section
        with st.expander("⚙️ Query Settings", expanded=False):
            user_id_input = st.text_input(
                "👤 User ID (Optional)",
                placeholder="user123",
                help="For consistent variant assignment",
            )
            variant_selection = st.selectbox(
                "🎯 Force Variant",
                variant_ids,
                help="Override automatic variant selection",
            )
            include_sources = st.checkbox(
                "📚 Show Sources", value=True, help="Display source documents"
            )

        st.markdown("---")

        # A/B Variants Info
        with st.expander("🧪 A/B Test Variants", expanded=False):
            if variant_details:
                for v in variant_details.values():
                    st.markdown(
                        f"""
                    **{v['name']}**  
                    Traffic: {v['traffic_percentage']}%  
                    Temp: {v['temperature']} | Tokens: {v['max_tokens']}
                    """
                    )

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
    st.markdown(
        """
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <h1 style="color: white; font-size: 48px; font-weight: 700; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">⚡ Energy RAG Assistant</h1>
        <p style="color: rgba(255,255,255,0.95); font-size: 18px; margin-top: 12px; font-weight: 500;">Ask me anything about energy systems, sustainability, and power generation</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            """
        <div style="text-align: center; padding: 60px 20px; color: #a0aec0;">
            <div style="font-size: 64px; margin-bottom: 20px;">💬</div>
            <div style="font-size: 22px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px;">Start a Conversation</div>
            <div style="font-size: 15px; color: #a0aec0; line-height: 1.6;">Ask me about renewable energy, power systems, or energy efficiency!</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-message">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Query Input Area
    col1, col2 = st.columns([5, 1])

    with col1:
        query_input = st.text_area(
            "Your Question",
            placeholder="💭 Type your question here... (e.g., How can I reduce my energy consumption?)",
            height=100,
            key=f"query_input_field_{st.session_state.input_key}",
            label_visibility="collapsed",
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
        variant_id_to_send = (
            None if variant_selection == "Auto Assign" else variant_selection
        )

        with st.spinner("🤔 Thinking..."):
            try:
                # API Request
                payload = {
                    "question": query_input,
                    "include_sources": include_sources,
                    "variant_id": variant_id_to_send,
                    "user_id": user_id_input if user_id_input else None,
                }

                response = requests.post(
                    f"{API_BASE_URL}/query", json=payload, timeout=60
                )
                response.raise_for_status()
                result = response.json()

                # Process response
                answer = result.get(
                    "answer", "I apologize, but I couldn't generate an answer."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

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
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                )
                st.error(error_msg)
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {error_msg}"}
                )
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
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #a0aec0; margin-bottom: 6px; font-weight: 500;">Variant Used</div>
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0;">{result.get('variant_name', 'N/A')}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #a0aec0; margin-bottom: 6px; font-weight: 500;">Response Time</div>
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0;">{result.get('latency', 0):.2f}s</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #a0aec0; margin-bottom: 6px; font-weight: 500;">Tokens Used</div>
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0;">
                    {result.get('tokens_used', {}).get('input', 0)} in / {result.get('tokens_used', {}).get('output', 0)} out
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div style="font-size: 13px; color: #a0aec0; margin-bottom: 6px; font-weight: 500;">Estimated Cost</div>
                <div style="font-size: 20px; font-weight: 600; color: #e2e8f0;">${result.get('estimated_cost', 0):.6f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Feedback Section
            st.markdown("---")
            st.markdown("### 💬 Rate This Response")

            feedback_score = st.slider(
                "How satisfied are you?", 1, 5, 5, key="feedback_score_slider"
            )
            feedback_comment = st.text_area(
                "Additional Comments (Optional)",
                height=80,
                key="feedback_comment_input",
                placeholder="Tell us what you think...",
            )

            if st.button(
                "📤 Submit Feedback", type="secondary", use_container_width=True
            ):
                submit_feedback(
                    st.session_state.last_query_data["query"],
                    st.session_state.last_query_data["variant_id"],
                    feedback_score,
                    feedback_comment,
                )

        with col_b:
            if include_sources and result.get("sources"):
                st.markdown("### 📚 Source Documents")
                st.markdown(
                    format_source_display(result.get("sources", [])),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                <div style="text-align: center; padding: 60px 20px; color: #a0aec0;">
                    <div style="font-size: 56px; margin-bottom: 16px;">📄</div>
                    <div style="font-size: 16px; font-weight: 500; color: #cbd5e0;">No sources to display</div>
                    <div style="font-size: 13px; color: #718096; margin-top: 8px;">Enable "Show Sources" in settings</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main_ui()
