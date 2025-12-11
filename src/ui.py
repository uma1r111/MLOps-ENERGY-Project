"""
Streamlit UI Client for FastAPI RAG System - Professional Chatbot Interface
Enhanced with Hourly Predictions Tab
"""

import streamlit as st
import requests
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# Adjust path to ensure config is imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.config import LANGSMITH_API_KEY, GEMINI_MODEL, FASTEMBED_MODEL

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000"
GRAFANA_BASE_URL = "http://localhost:3000"
PROMETHEUS_BASE_URL = "http://localhost:9090"
LANGSMITH_BASE_URL = "https://smith.langchain.com"

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "bentoml_forecast_output.csv"


# --- DATA LOADING ---

def load_predictions() -> Optional[pd.DataFrame]:
    """Load predictions from local BentoML forecast CSV"""
    try:
        if PREDICTIONS_FILE.exists():
            df = pd.read_csv(PREDICTIONS_FILE)
            
            # Parse datetime column
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                df['hour'] = df['datetime'].dt.hour
                df['time'] = df['datetime'].dt.strftime('%H:%M')
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None


def get_price_category(price):
    """Categorize price into Low/Medium/High"""
    if price < 0.3:
        return "Low", "#00ff88"
    elif price < 0.7:
        return "Medium", "#ffd700"
    else:
        return "High", "#ff4444"


def get_time_icon(hour):
    """Return emoji based on time of day"""
    if 0 <= hour < 6:
        return "🌙"
    elif 6 <= hour < 12:
        return "🌅"
    elif 12 <= hour < 18:
        return "☀️"
    else:
        return "🌆"


# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
    * { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.94)), url("background.jpg") center/cover no-repeat fixed !important;
    }

    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        color: #e8f6ff !important;
        text-shadow: 0 0 24px rgba(0, 200, 255, 0.55);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 14, 20, 0.6);
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 30, 40, 0.8);
        color: #a0d0f4;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid rgba(0, 150, 255, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00aaff, #00e0ff) !important;
        color: white !important;
    }

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

    /* Query Details and Rate Response section styling */
    .stMarkdown h3 {
        color: #ffffff !important;
    }

    /* Metrics in Query Details */
    [data-testid="stMetricValue"] {
        color: #00e0ff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* Slider label */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Text area labels */
    label[data-testid="stWidgetLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* All paragraph text */
    p {
        color: #e8f6ff !important;
    }

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

    .hour-card {
        background: rgba(30,45,65,0.85);
        border: 2px solid rgba(0,200,255,0.4);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,150,255,0.2);
    }

    .hour-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0,200,255,0.7);
        box-shadow: 0 8px 25px rgba(0,150,255,0.4);
    }

    .hour-time {
        font-size: 18px;
        font-weight: 700;
        color: #e8f6ff;
        margin-bottom: 12px;
    }

    .hour-icon {
        font-size: 48px;
        margin: 15px 0;
    }

    .hour-price {
        font-size: 32px;
        font-weight: 800;
        margin: 15px 0;
        text-shadow: 0 0 20px currentColor;
    }

    .hour-category {
        font-size: 14px;
        font-weight: 600;
        color: #c0d5e8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .summary-card {
        background: rgba(30,40,50,0.9);
        border: 2px solid rgba(0,200,255,0.5);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,150,255,0.25);
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        color: #00e0ff;
        text-shadow: 0 0 25px rgba(0,224,255,0.8);
    }

    .metric-label {
        font-size: 15px;
        color: #e0f0ff;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    #MainMenu, header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


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
        return ["Auto Assign"], False, {}


def format_source_display(sources: List) -> str:
    """Formats source documents"""
    if not sources:
        return '<div style="text-align: center; padding: 30px; color: #ffffff;">📄 No source documents found</div>'

    formatted = '<div style="margin-top: 16px;">'
    for i, doc in enumerate(sources, 1):
        source = doc.get("source", "Unknown")
        score = doc.get("retrieval_score", 0)
        page = doc.get("page")
        content = doc.get("content", "").strip()
        preview = content[:250] + "..." if len(content) > 250 else content
        page_info = f" • Page {page}" if page else ""

        formatted += f"""
        <div style="background: rgba(40, 50, 65, 0.85); padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid rgba(102, 126, 234, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #ffffff; font-size: 14px;">📄 Source {i}</span>
                <span style="background: rgba(0, 200, 255, 0.25); padding: 5px 12px; border-radius: 10px; font-size: 12px; color: #00e0ff; font-weight: 600;">
                    Score: {score:.3f}
                </span>
            </div>
            <div style="font-size: 12px; color: #e0e0e0; margin-bottom: 10px; font-weight: 500;">{source}{page_info}</div>
            <div style="font-size: 13px; color: #ffffff; line-height: 1.7;">{preview}</div>
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
        response = requests.post(f"{API_BASE_URL}/feedback", json=feedback_payload, timeout=10)
        response.raise_for_status()
        st.success("✅ Thank you for your feedback!")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Feedback Error: {e}")


# --- PREDICTIONS TAB ---

def render_hourly_day_tab(day_df, day_name, day_date):
    """Render hourly predictions for a single day"""
    
    if day_df.empty:
        st.warning(f"No data available for {day_name}")
        return
    
    price_col = 'predicted_retail_price_£_per_kWh'
    
    # Header with date
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: rgba(20,30,40,0.8); border-radius: 12px; margin-bottom: 30px; border: 2px solid rgba(0,200,255,0.5);">
        <h2 style="margin: 0; color: #00e0ff;">{day_name}</h2>
        <p style="font-size: 20px; color: #e8f6ff; margin: 10px 0;">Get a 24-hour forecast of Retail Energy Prices (£/kWh)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value">£{day_df[price_col].mean():.3f}</div>
            <div style="font-size: 13px; color: #e0f0ff; margin-top: 8px;">per kWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        peak_hour = day_df.loc[day_df[price_col].idxmax()]
        st.markdown(f"""
        <div class="summary-card">
            <div class="metric-label">Peak Price</div>
            <div class="metric-value">£{day_df[price_col].max():.3f}</div>
            <div style="font-size: 13px; color: #e0f0ff; margin-top: 8px;">at {peak_hour['time']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lowest_hour = day_df.loc[day_df[price_col].idxmin()]
        st.markdown(f"""
        <div class="summary-card">
            <div class="metric-label">Lowest Price</div>
            <div class="metric-value">£{day_df[price_col].min():.3f}</div>
            <div style="font-size: 13px; color: #e0f0ff; margin-top: 8px;">at {lowest_hour['time']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        daily_total = day_df[price_col].sum()
        st.markdown(f"""
        <div class="summary-card">
            <div class="metric-label">Daily Total</div>
            <div class="metric-value">£{daily_total:.2f}</div>
            <div style="font-size: 13px; color: #e0f0ff; margin-top: 8px;">24h sum</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hourly forecast cards in 6 columns (4 rows of 6)
    hours_data = []
    for _, row in day_df.iterrows():
        hours_data.append({
            'hour': row['hour'],
            'time': row['time'],
            'price': row[price_col],
            'icon': get_time_icon(row['hour'])
        })
    
    # Display in rows of 6
    for row_start in range(0, 24, 6):
        cols = st.columns(6)
        for i, col in enumerate(cols):
            hour_idx = row_start + i
            if hour_idx < len(hours_data):
                hour_data = hours_data[hour_idx]
                category, color = get_price_category(hour_data['price'])
                
                with col:
                    st.markdown(f"""
                    <div class="hour-card">
                        <div class="hour-time">{hour_data['time']}</div>
                        <div class="hour-icon">{hour_data['icon']}</div>
                        <div class="hour-price" style="color: {color};">£{hour_data['price']:.3f}</div>
                        <div class="hour-category">{category}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart
    st.markdown("### 📊 24-Hour Price Trend")
    
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=day_df['time'],
        y=day_df[price_col],
        mode='lines+markers',
        name='Price',
        line=dict(color='#00caff', width=3),
        marker=dict(size=8, color='#00caff'),
        fill='tozeroy',
        fillcolor='rgba(0, 202, 255, 0.15)'
    ))
    
    # Highlight peak
    fig.add_trace(go.Scatter(
        x=[peak_hour['time']],
        y=[peak_hour[price_col]],
        mode='markers',
        name='Peak',
        marker=dict(size=18, color='#ff4444', symbol='star', line=dict(width=2, color='white'))
    ))
    
    # Highlight lowest
    fig.add_trace(go.Scatter(
        x=[lowest_hour['time']],
        y=[lowest_hour[price_col]],
        mode='markers',
        name='Lowest',
        marker=dict(size=18, color='#00ff88', symbol='star', line=dict(width=2, color='white'))
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,30,40,0.6)',
        font=dict(color='#e8f6ff', size=12),
        xaxis_title="Time",
        yaxis_title="Price (£/kWh)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_predictions_dashboard():
    """Render the predictions dashboard with 3 day tabs"""
    st.markdown("## ⚡ Retail Energy Price Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📂 **Data Source:** `{PREDICTIONS_FILE.name}`")
    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    predictions_df = load_predictions()
    
    if predictions_df is None or predictions_df.empty:
        st.warning(f"⚠️ No predictions data available at: `{PREDICTIONS_FILE}`")
        st.info("""
        **How to get predictions:**
        1. Ensure BentoML forecast has been run
        2. Check that `bentoml_forecast_output.csv` exists in project root
        3. Click 'Reload' button above
        """)
        return
    
    # Get unique dates and create tabs
    unique_dates = sorted(predictions_df['date'].unique())
    
    if len(unique_dates) < 3:
        st.error("Not enough data for 3 days. Need at least 72 hours of predictions.")
        return
    
    # Create tabs for the 3 days
    day1_date = unique_dates[0]
    day2_date = unique_dates[1]
    day3_date = unique_dates[2]
    
    day1_name = pd.to_datetime(day1_date).strftime('%A, %B %d, %Y')
    day2_name = pd.to_datetime(day2_date).strftime('%A, %B %d, %Y')
    day3_name = pd.to_datetime(day3_date).strftime('%A, %B %d, %Y')
    
    tab1, tab2, tab3 = st.tabs([
        f"📅 {pd.to_datetime(day1_date).strftime('%a, %b %d')}",
        f"📅 {pd.to_datetime(day2_date).strftime('%a, %b %d')}",
        f"📅 {pd.to_datetime(day3_date).strftime('%a, %b %d')}"
    ])
    
    with tab1:
        day1_df = predictions_df[predictions_df['date'] == day1_date].reset_index(drop=True)
        render_hourly_day_tab(day1_df, day1_name, day1_date)
    
    with tab2:
        day2_df = predictions_df[predictions_df['date'] == day2_date].reset_index(drop=True)
        render_hourly_day_tab(day2_df, day2_name, day2_date)
    
    with tab3:
        day3_df = predictions_df[predictions_df['date'] == day3_date].reset_index(drop=True)
        render_hourly_day_tab(day3_df, day3_name, day3_date)


# --- CHAT TAB ---

def render_chat_interface():
    """Render chat interface"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #e0f0ff;">
            <div style="font-size: 64px; margin-bottom: 20px;">💬</div>
            <div style="font-size: 22px; font-weight: 600; color: #e8f6ff; margin-bottom: 10px;">Start a Conversation</div>
            <div style="font-size: 15px; color: #c0d5f0;">Ask about renewable energy, power systems, or efficiency!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        query_input = st.text_area(
            "Your Question",
            placeholder="💭 Type your question...",
            height=100,
            key=f"query_input_{st.session_state.input_key}",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
        submit_button = st.button("🚀 Send", use_container_width=True, type="primary")

    # Handle submission
    if submit_button and query_input and st.session_state.api_ready:
        st.session_state.messages.append({"role": "user", "content": query_input})
        
        variant_id = None if st.session_state.variant_selection == "Auto Assign" else st.session_state.variant_selection
        
        with st.spinner("🤔 Thinking..."):
            try:
                payload = {
                    "question": query_input,
                    "include_sources": st.session_state.include_sources,
                    "variant_id": variant_id,
                    "user_id": st.session_state.user_id_input or None
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                answer = result.get("answer", "I couldn't generate an answer.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                st.session_state.last_query_data = {
                    "query": query_input,
                    "variant_id": result.get("variant_id", "N/A"),
                    "variant_name": result.get("variant_name", "N/A")
                }
                st.session_state.last_result = result
                st.session_state.show_results = True
                st.session_state.input_key += 1
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {str(e)}"})

    # Results display
    if st.session_state.show_results and st.session_state.last_result:
        result = st.session_state.last_result
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.markdown("### 📊 Query Details")
            st.metric("Variant", result.get('variant_name', 'N/A'))
            st.metric("Response Time", f"{result.get('latency', 0):.2f}s")
            tokens = result.get('tokens_used', {})
            st.metric("Tokens", f"{tokens.get('input', 0)} in / {tokens.get('output', 0)} out")
            st.metric("Cost", f"${result.get('estimated_cost', 0):.6f}")
            
            st.markdown("### 💬 Rate Response")
            score = st.slider("Satisfaction", 1, 5, 5, key="feedback_slider")
            comment = st.text_area("Comments", height=60, key="feedback_comment")
            if st.button("📤 Submit Feedback", use_container_width=True):
                submit_feedback(
                    st.session_state.last_query_data["query"],
                    st.session_state.last_query_data["variant_id"],
                    score,
                    comment
                )
        
        with col_b:
            if st.session_state.include_sources and result.get("sources"):
                st.markdown("### 📚 Sources")
                st.markdown(format_source_display(result.get("sources", [])), unsafe_allow_html=True)


# --- MAIN UI ---

def main_ui():
    """Main UI with tabs"""
    st.set_page_config(
        layout="wide",
        page_title="Energy RAG Assistant",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )

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
    st.session_state.api_ready = api_ready

    # Sidebar
    with st.sidebar:
        st.markdown("# ⚡ Energy RAG")
        st.markdown("### AI-Powered Assistant")
        
        # API Status
        status_color = "#48bb78" if api_ready else "#f56565"
        status_text = "Connected" if api_ready else "Disconnected"
        st.markdown(f'<div style="background: {status_color}; padding: 8px; border-radius: 8px; text-align: center; color: white; font-weight: 600;">API: {status_text}</div>', unsafe_allow_html=True)
        
        # Data File Status
        data_exists = PREDICTIONS_FILE.exists()
        data_color = "#48bb78" if data_exists else "#f56565"
        data_text = "Available" if data_exists else "Not Found"
        st.markdown(f'<div style="background: {data_color}; padding: 8px; border-radius: 8px; text-align: center; color: white; font-weight: 600; margin-top: 8px;">Data: {data_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.user_id_input = st.text_input("👤 User ID", placeholder="user123")
            st.session_state.variant_selection = st.selectbox("🎯 Variant", variant_ids)
            st.session_state.include_sources = st.checkbox("📚 Show Sources", value=True)
        
        st.markdown("---")
        
        with st.expander("🧪 A/B Variants", expanded=False):
            for v in variant_details.values():
                st.markdown(f"**{v['name']}** - {v['traffic_percentage']}%")
        
        st.markdown("---")
        st.markdown("### 📊 System")
        st.caption(f"**Model:** {GEMINI_MODEL}")
        st.caption(f"**Embeddings:** {FASTEMBED_MODEL}")
        
        st.markdown("---")
        st.markdown("### 🔍 Monitoring")
        if st.button("📈 Grafana", use_container_width=True):
            st.write(f"[Dashboard]({GRAFANA_BASE_URL})")
        if st.button("📉 Prometheus", use_container_width=True):
            st.write(f"[Metrics]({PROMETHEUS_BASE_URL})")
        if LANGSMITH_API_KEY:
            if st.button("🔗 LangSmith Traces", use_container_width=True):
                st.write(f"[View Project]({LANGSMITH_BASE_URL})")
                
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_results = False
            st.session_state.last_result = None
            st.session_state.input_key += 1
            st.rerun()

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1>⚡ Energy RAG Assistant</h1>
        <p style="color: #e8f6ff; font-size: 18px;">AI-powered Q&A with retail energy price predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📈 Price Forecast"])

    with tab1:
        render_chat_interface()

    with tab2:
        render_predictions_dashboard()


if __name__ == "__main__":
    main_ui()