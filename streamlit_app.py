"""
TapTap Analytics Chatbot — Streamlit Frontend
Run with: streamlit run streamlit_app.py
Requires: pip install streamlit requests pandas
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TapTap Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables */
:root {
    --bg: #0d0f14;
    --surface: #151820;
    --surface2: #1c2030;
    --border: #252a3a;
    --accent: #4f8ef7;
    --accent2: #7c3aed;
    --success: #22c55e;
    --warning: #f59e0b;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --user-bg: #1e2a4a;
    --bot-bg: #151820;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: var(--user-bg);
    border: 1px solid #2d3a5e;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin-left: 2rem;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin-right: 2rem;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 142, 247, 0.15) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}

/* Buttons */
.stButton button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface2);
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.status-online { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.status-offline { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* Header */
.taptap-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.taptap-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.taptap-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}

.taptap-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: 0;
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    margin: 4px;
    transition: all 0.2s;
}
.suggestion-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* Row count badge */
.row-badge {
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.3);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

SUGGESTIONS = [
    "Who solved today's POD?",
    "Which students failed the POD?",
    "Top 10 by employability score",
    "Who solved POD in the IT domain?",
    "Which hackathon had most participants?",
    "Students with POD streak above 5",
]

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_state" not in st.session_state:
    st.session_state.api_state = {}
if "suggested" not in st.session_state:
    st.session_state.suggested = None

# ── Helper functions ────────────────────────────────────────────────────────

def check_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def send_query(query: str, state: dict) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"query": query, "state": state},
            timeout=60,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": "❌ Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.",
            "data": [],
            "state": state,
            "success": False,
        }
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "data": [],
            "state": state,
            "success": False,
        }


def render_data_table(data: list):
    if not data:
        return
    df = pd.DataFrame(data)
    # Convert datetime columns to readable strings
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 TapTap Analytics")
    st.markdown("---")

    # Health check
    health = check_health()
    if health:
        db_status = health.get("database", "unknown")
        llm_status = health.get("llm", "unknown")
        badge_db = "online" if db_status == "connected" else "offline"
        badge_llm = "online" if llm_status == "initialised" else "offline"
        st.markdown(f"""
        <div style='margin-bottom:8px'>
            <span style='color:#64748b;font-size:12px'>DATABASE</span><br>
            <span class='status-badge status-{badge_db}'>{db_status.upper()}</span>
        </div>
        <div style='margin-bottom:16px'>
            <span style='color:#64748b;font-size:12px'>LLM AGENT</span><br>
            <span class='status-badge status-{badge_llm}'>{llm_status.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        v = health.get("version", "")
        if v:
            st.caption(f"v{v}")
    else:
        st.markdown("<span class='status-badge status-offline'>SERVER OFFLINE</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state.suggested = s

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_state = {}
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Azure OpenAI + LangGraph")

# ── Main area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='taptap-header'>
    <div class='taptap-logo'>🎯</div>
    <div>
        <div class='taptap-title'>TapTap Analytics</div>
        <div class='taptap-subtitle'>Faculty Intelligence Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome message if no history
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;'>
        <div style='font-size:2.5rem;margin-bottom:1rem'>🎯</div>
        <div style='font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.5rem'>
            Ask anything about student performance
        </div>
        <div style='font-size:0.85rem;color:#64748b;max-width:400px;margin:0 auto'>
            Query POD submissions, employability scores, hackathon results,
            and more — in plain English.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🏫" if msg["role"] == "user" else "🎯"):
        st.markdown(msg["content"])
        if msg.get("data"):
            with st.expander(f"📊 View data  —  {len(msg['data'])} rows", expanded=False):
                render_data_table(msg["data"])

# ── Handle suggestion click ─────────────────────────────────────────────────
prompt = st.session_state.suggested
st.session_state.suggested = None

# ── Chat input ──────────────────────────────────────────────────────────────
typed = st.chat_input("Ask about students, POD, hackathons, employability...")
if typed:
    prompt = typed

# ── Process query ───────────────────────────────────────────────────────────
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🏫"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant", avatar="🎯"):
        with st.spinner("Querying database..."):
            result = send_query(prompt, st.session_state.api_state)

        answer = result.get("answer", "No response received.")
        data = result.get("data", [])

        # Update conversation state for multi-turn
        if result.get("state"):
            st.session_state.api_state = result["state"]

        st.markdown(answer)

        if data:
            with st.expander(f"📊 View data  —  {len(data)} rows", expanded=True):
                render_data_table(data)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "data": data,
    })

    st.rerun()