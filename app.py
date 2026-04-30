import streamlit as st
import os
import re
from dotenv import load_dotenv
from utils import (
    create_llms,
    create_orchestrator,
    create_math_science_agent,
    create_code_agent,
    create_websearch_agent,
    build_workflow
)

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

:root {
    --bg:      #0d0f14;
    --surface: #13161e;
    --border:  #1f2433;
    --muted:   #4a5070;
    --text:    #d4d8e8;
    --accent:  #4f8ef7;
    --green:   #3ecf8e;
    --amber:   #f5a623;
    --red:     #e05c5c;
    --mono:    'IBM Plex Mono', monospace;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 980px; }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.mas-title {
    font-family: var(--mono);
    font-size: 1.65rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.mas-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 2rem;
    font-weight: 300;
}

textarea {
    font-family: var(--mono) !important;
    font-size: 0.875rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    resize: vertical !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.12) !important;
}

div[data-testid="stButton"] > button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 0.45rem 1.6rem;
    font-family: var(--mono);
    font-size: 0.83rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: background 0.2s, transform 0.1s;
}
div[data-testid="stButton"] > button:hover  { background: #3a7de6; transform: translateY(-1px); }
div[data-testid="stButton"] > button:active { transform: translateY(0); }

.pipeline-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.6rem 0 1rem;
}
.badge {
    font-family: var(--mono);
    font-size: 0.73rem;
    padding: 3px 11px;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.badge-ok  { background:rgba(62,207,142,0.10); color:var(--green); border:1px solid rgba(62,207,142,0.28); }
.badge-fb  { background:rgba(224,92,92,0.08);  color:var(--red);   border:1px solid rgba(224,92,92,0.22); }
.badge-arr { color:var(--muted); font-size:0.85rem; }

.cat-chip {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 2px 9px;
    border-radius: 3px;
    background: rgba(79,142,247,0.08);
    color: var(--accent);
    border: 1px solid rgba(79,142,247,0.2);
    margin-bottom: 0.8rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.response-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1.3rem 1.5rem;
    margin-top: 0.4rem;
    line-height: 1.75;
    color: var(--text);
}

.sec-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

.status-ok { color:var(--green); font-family:var(--mono); font-size:0.76rem; margin-bottom:0.8rem; }
.status-fb { color:var(--amber); font-family:var(--mono); font-size:0.76rem; margin-bottom:0.8rem; }

.hist-q {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text);
    background: rgba(255,255,255,0.025);
    padding: 5px 11px;
    border-radius: 4px;
    border-left: 2px solid var(--muted);
    margin-bottom: 5px;
}
.hist-m {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    margin-bottom: 7px;
}
hr.div { border:none; border-top:1px solid var(--border); margin:0.9rem 0; }

.api-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: var(--muted);
    margin: 5px 0;
    font-family: var(--mono);
}
.dot-on  { width:6px; height:6px; border-radius:50%; background:var(--green); flex-shrink:0; }
.dot-off { width:6px; height:6px; border-radius:50%; background:var(--muted); flex-shrink:0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def convert_latex(text: str) -> str:
    """Convert \\( \\) and \\[ \\] delimiters to KaTeX-compatible $ and $$."""
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$',   text, flags=re.DOTALL)
    return text

def strip_agent_prefix(text: str) -> str:
    """Remove the leading 'Agent Label:\\n\\n' prefix from responses."""
    return re.sub(r'^[^\n]+:\n\n', '', text, count=1)

def render_response(response: str, category: str):
    body = strip_agent_prefix(response)
    if category == "math/science":
        st.markdown(convert_latex(body))
    else:
        st.markdown(body)

def pipeline_html(agents: list) -> str:
    parts = []
    for i, name in enumerate(agents):
        clean = re.sub(r'[^\x00-\x7F]+', '', name).strip()  # strip emojis
        cls   = "badge-fb" if "fallback" in name.lower() else "badge-ok"
        parts.append(f'<span class="badge {cls}">{clean}</span>')
        if i < len(agents) - 1:
            parts.append('<span class="badge-arr">&#8594;</span>')
    return f'<div class="pipeline-wrap">{"".join(parts)}</div>'

def api_dot(label: str, ok: bool):
    cls = "dot-on" if ok else "dot-off"
    st.markdown(
        f'<div class="api-row"><div class="{cls}"></div>{label}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Multi-Agent System")
    st.markdown("<hr style='border:none;border-top:1px solid #1f2433;margin:0.5rem 0 1.2rem'>",
                unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Agents</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.8rem;line-height:2;color:#8892b0;font-family:"IBM Plex Mono",monospace'>
Math / Science &mdash; equations, derivations<br>
Code &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&mdash; algorithms, LeetCode<br>
Web Search &nbsp;&nbsp;&nbsp;&nbsp;&mdash; news, general knowledge
</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #1f2433;margin:1rem 0'>",
                unsafe_allow_html=True)

    st.markdown('<div class="sec-label">API Status</div>', unsafe_allow_html=True)
    api_dot("Groq",          True)
    api_dot("Wolfram Alpha", "WOLFRAM_ALPHA_APPID" in os.environ)
    api_dot("SerpAPI",       "SERPAPI_API_KEY"     in os.environ)

    st.markdown("<hr style='border:none;border-top:1px solid #1f2433;margin:1rem 0'>",
                unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.last_result = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Init agents
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_agents():
    general_llm, classifier_llm, math_science_llm, code_llm = create_llms()
    orchestrator = create_orchestrator(classifier_llm)
    math_agent   = create_math_science_agent(math_science_llm)
    code_agent_  = create_code_agent(code_llm)
    search_agent = create_websearch_agent(general_llm)
    return build_workflow(orchestrator, math_agent, code_agent_, search_agent)

agent_graph = init_agents()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="mas-title">Multi-Agent Query System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mas-sub">Specialized models handle math and code directly. '
    'Web search activates only when the primary agent fails.</div>',
    unsafe_allow_html=True
)

# Conversation history
if st.session_state.conversation_history:
    n = len(st.session_state.conversation_history)
    with st.expander(f"Conversation history — {n} turn{'s' if n > 1 else ''}", expanded=False):
        for ex in reversed(st.session_state.conversation_history):
            cat   = ex.get("category", "?")
            tried = ex.get("agents_tried", [])
            trail = " → ".join(re.sub(r'[^\x00-\x7F]+', '', a).strip() for a in tried)
            st.markdown(f'<div class="hist-q">{ex["query"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hist-m">{cat.upper()} &nbsp;|&nbsp; {trail}</div>', unsafe_allow_html=True)
            body = strip_agent_prefix(ex["response"])
            preview = (convert_latex(body) if cat == "math/science" else body)[:500]
            st.markdown(preview + "…")
            st.markdown('<hr class="div">', unsafe_allow_html=True)

# Input
st.markdown('<div class="sec-label" style="margin-top:1.2rem">Question</div>', unsafe_allow_html=True)

query = st.text_area(
    label="question",
    label_visibility="collapsed",
    placeholder="Ask anything — a maths derivation, coding problem, or news query…\n\nShift + Enter for a new line, then click Submit.",
    height=115,
    key="user_query"
)

submitted = st.button("Submit")

if submitted:
    if query.strip():
        with st.spinner("Running…"):
            try:
                result = agent_graph.invoke({
                    "query":                query.strip(),
                    "category":             "",
                    "response":             "",
                    "agents_tried":         [],
                    "conversation_history": st.session_state.conversation_history,
                })
                st.session_state.last_result = result
                st.session_state.conversation_history.append({
                    "query":        query.strip(),
                    "category":     result["category"],
                    "response":     result["response"],
                    "agents_tried": result.get("agents_tried", []),
                })
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.last_result = None
    else:
        st.warning("Please enter a question.")

# Result display
if st.session_state.last_result:
    r        = st.session_state.last_result
    agents   = r.get("agents_tried", [])
    category = r.get("category", "unknown")
    used_fb  = any("fallback" in a.lower() for a in agents)

    st.markdown('<div class="sec-label" style="margin-top:1.6rem">Agent pipeline</div>',
                unsafe_allow_html=True)
    st.markdown(pipeline_html(agents), unsafe_allow_html=True)

    if used_fb:
        st.markdown('<div class="status-fb">Primary agent failed — web search used as fallback.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-ok">Resolved by primary agent.</div>',
                    unsafe_allow_html=True)

    st.markdown(f'<div class="cat-chip">{category}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Response</div>', unsafe_allow_html=True)
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    render_response(r["response"], category)
    st.markdown('</div>', unsafe_allow_html=True)