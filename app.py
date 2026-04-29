import streamlit as st
import os
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

st.set_page_config(page_title="Multi-Agent System", layout="wide")
st.title("🤖 Multi-Agent Query System")

# ── Session State ─────────────────────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    Routes queries to specialized agents:
    - **🧪 Math/Science** — equations, derivations, research
    - **💻 Code** — algorithms, implementations, LeetCode
    - **🔍 Web Search** — news, general knowledge

    Specialized agents try first. If they fail, the **web agent acts as fallback**.
    """)

    st.subheader("API Status")
    st.success("✅ Groq API Key detected")
    if "WOLFRAM_ALPHA_APPID" in os.environ:
        st.success("✅ Wolfram Alpha detected")
    else:
        st.warning("⚠️ Wolfram Alpha not configured (optional)")
    if "SERPAPI_API_KEY" in os.environ:
        st.success("✅ SerpAPI detected")
    else:
        st.info("ℹ️ SerpAPI not set — using DuckDuckGo")

    st.subheader("Context")
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Cleared!")

# ── Init agents (cached implicitly by Streamlit's top-level execution) ────────
general_llm, math_science_llm, code_llm = create_llms()
orchestrator_chain    = create_orchestrator(general_llm)
math_science_executor = create_math_science_agent(math_science_llm)
code_executor         = create_code_agent(code_llm)
websearch_executor    = create_websearch_agent(general_llm)

agent_graph = build_workflow(
    orchestrator_chain,
    math_science_executor,
    code_executor,
    websearch_executor
)

# ── Conversation History expander ─────────────────────────────────────────────
if st.session_state.conversation_history:
    with st.expander("📜 Conversation History", expanded=False):
        for i, ex in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Query {i+1}:** {ex['query']}")
            st.markdown(f"**Category:** `{ex['category']}`")

            # Show agent trail as badges
            if ex.get("agents_tried"):
                badges = "  ".join([f"`{a}`" for a in ex["agents_tried"]])
                st.markdown(f"**Agents tried:** {badges}")

            st.markdown(f"**Response:** {ex['response'][:400]}...")
            st.divider()

# ── Main Input ────────────────────────────────────────────────────────────────
st.markdown("""
Ask anything — math problems, coding questions, or general knowledge.
The system picks the best agent automatically and falls back to web search if needed.
""")

query = st.text_input("Enter your question:", key="user_query", placeholder="e.g. Derive the optical flow equation")

if st.button("🚀 Submit"):
    if query.strip():
        with st.spinner("Agents working on your query..."):
            try:
                result = agent_graph.invoke({
                    "query": query,
                    "category": "",
                    "response": "",
                    "agents_tried": [],
                    "conversation_history": st.session_state.conversation_history
                })

                # ── Agent Trail ───────────────────────────────────────────────
                st.subheader("🔎 Agent Pipeline")
                agents = result.get("agents_tried", [])
                if agents:
                    cols = st.columns(len(agents))
                    for i, agent_name in enumerate(agents):
                        with cols[i]:
                            if "fallback" in agent_name.lower():
                                st.error(f"⚠️ {agent_name}")
                            elif i == 0:
                                st.success(f"✅ {agent_name}")
                            else:
                                st.warning(f"🔄 {agent_name}")
                    if len(agents) > 1:
                        st.info(
                            f"ℹ️ Primary agent couldn't solve this — "
                            f"**{agents[-1]}** was used as fallback."
                        )
                    else:
                        st.success("✅ Solved by primary agent — no fallback needed.")

                # ── Category badge ─────────────────────────────────────────────
                cat = result.get("category", "unknown")
                cat_icon = {"math/science": "🧪", "code": "💻", "websearch": "🔍"}.get(cat, "🤖")
                st.caption(f"Classified as: {cat_icon} **{cat}**")

                # ── Response ───────────────────────────────────────────────────
                st.subheader("📋 Result")
                st.markdown(result["response"])

                # ── Save to history ────────────────────────────────────────────
                st.session_state.conversation_history.append({
                    "query": query,
                    "category": result["category"],
                    "response": result["response"],
                    "agents_tried": agents
                })

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    else:
        st.warning("Please enter a question first.")