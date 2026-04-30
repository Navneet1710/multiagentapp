import os
import re
from typing import Dict, TypedDict, Literal, List, Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langgraph.graph import StateGraph, END


# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query: str
    category: str
    response: str
    agents_tried: List[str]
    conversation_history: List[Dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Factory  —  returns 4 models
# ═══════════════════════════════════════════════════════════════════════════════

def create_llms():
    """
    general_llm      — llama-3.1-8b-instant   : web-search synthesis (cheap, fast)
    classifier_llm   — llama-3.3-70b-versatile : strong classification fallback
    math_science_llm — llama-4-scout-17b       : MoE reasoning model
    code_llm         — openai/gpt-oss-20b      : best code model on Groq
    """
    general_llm      = ChatGroq(temperature=0.3, model="llama-3.1-8b-instant")
    classifier_llm   = ChatGroq(temperature=0.0, model="llama-3.3-70b-versatile")
    math_science_llm = ChatGroq(temperature=0.1, model="meta-llama/llama-4-scout-17b-16e-instruct")
    code_llm         = ChatGroq(temperature=0.1, model="openai/gpt-oss-20b")
    return general_llm, classifier_llm, math_science_llm, code_llm


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Classifier  — Stage 1: regex  |  Stage 2: llama-3.3-70b
# ═══════════════════════════════════════════════════════════════════════════════

_CODE_RULES = [
    r'\b(?:write|give|create|provide|generate|make)\s+(?:a\s+|me\s+|an?\s+)?(?:code|program|function|script|solution|implementation)\b',
    r'\bcode\s+(?:for|in|to|that)\b',
    r'\bimplement\b',
    r'\bsolve\s+(?:this|the)\s+(?:problem|leetcode|challenge|question)\b',
    r'\b(?:c\+\+|c#|python|java(?:script)?|typescript|golang|rust|swift|kotlin|php|ruby|scala)\b',
    r'\bleetcode\b', r'\bhackerrank\b', r'\bcodeforces\b',
    r'\b(?:debug|debugging|fix\s+(?:my\s+|this\s+)?code)\b',
    r'\b(?:linked\s+list|binary\s+(?:tree|search)|bst|heap|hash\s*(?:map|table)|stack|queue|trie)\b',
    r'\b(?:time\s+complexity|space\s+complexity|big[\s-]?o)\b',
    r'\b(?:merge\s*sort|quick\s*sort|bubble\s*sort|insertion\s*sort)\b',
    r'\b(?:dfs|bfs|dijkstra|topological\s+sort)\b',
    r'\b(?:recursion|recursive)\b',
    r'\bdynamic\s+programming\b',
    r'\bprint\s+(?:hello|fibonacci|prime|factorial|pattern)\b',
    r'\b(?:array|string|integer)\s+(?:input|output|problem)\b',
]

_MATH_RULES = [
    r'\b(?:derive|derivation|proof|prove|deduce)\b',
    r'\b(?:equation|formula|theorem|lemma|corollary)\b',
    r'\b(?:integral|derivative|differentiat(?:e|ion)|calculus|gradient)\b',
    r'\b(?:physics|chemistry|biology|thermodynamics|quantum|mechanics)\b',
    r'\b(?:matrix|matrices|vector|eigenvalue|determinant|dot\s+product)\b',
    r'\b(?:probability|statistics|distribution|variance|standard\s+deviation)\b',
    r'\b(?:optical\s+flow|fourier|laplace\s+transform|euler|newton|bernoulli)\b',
    r'\b(?:sin|cos|tan|log|ln|sqrt)\s*\(',
    r'\b\d+\s*[\+\-\*\/\^]\s*\d+\b',
    r'\bwhat\s+is\s+\d',
    r'\b(?:calculate|compute|evaluate)\s+(?:the\s+)?(?:equation|integral|sum|derivative|value)\b',
]

_CODE_RE = [re.compile(p, re.IGNORECASE) for p in _CODE_RULES]
_MATH_RE  = [re.compile(p, re.IGNORECASE) for p in _MATH_RULES]


def _keyword_classify(query: str) -> Optional[str]:
    """
    Stage-1 keyword classifier.
    Returns 'code', 'math/science', or None (ambiguous → escalate to LLM).
    """
    code_hits = sum(1 for r in _CODE_RE if r.search(query))
    math_hits = sum(1 for r in _MATH_RE if r.search(query))

    if code_hits > 0 and math_hits == 0:   return "code"
    if math_hits > 0 and code_hits == 0:   return "math/science"
    if code_hits > math_hits:              return "code"
    if math_hits > code_hits:              return "math/science"
    return None   # tied or both zero — use LLM


def create_orchestrator(classifier_llm):
    """The orchestrator is the strong LLM. Return it directly."""
    return classifier_llm


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Factories  (all return plain data, no ReAct / AgentExecutor)
# ═══════════════════════════════════════════════════════════════════════════════

def create_math_science_agent(math_science_llm) -> Tuple:
    """Returns (llm, wolfram_func | None)."""
    wolfram_func = None
    if "WOLFRAM_ALPHA_APPID" in os.environ:
        try:
            wolfram_func = WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper()).run
        except Exception:
            pass
    return (math_science_llm, wolfram_func)


def create_code_agent(code_llm):
    """Returns the code LLM directly."""
    return code_llm


def create_websearch_agent(general_llm) -> Tuple:
    """
    Returns (llm, search_func | None).
    Uses direct tool-call pattern — no ReAct, no initialize_agent.
    """
    search_func = None
    if "SERPAPI_API_KEY" in os.environ:
        try:
            search_func = SerpAPIWrapper().run
        except Exception:
            pass
    if search_func is None:
        try:
            search_func = DuckDuckGoSearchRun(timeout=10).run
        except Exception:
            pass   # stays None → falls back to pure LLM knowledge
    return (general_llm, search_func)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def format_conversation_history(history: List[Dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for i, ex in enumerate(history[-3:]):
        snippet = ex["response"][:200].replace("\n", " ")
        lines += [
            f"[Turn {i+1}] User    : {ex['query']}",
            f"           Agent ({ex['category']}): {snippet}...",
        ]
    return "\n".join(lines)


def _run_web_search(websearch_agent: Tuple, query: str) -> str:
    """
    Direct tool-call pattern (no ReAct / iteration loops):
      1. LLM plans 1-3 focused search queries
      2. Execute each via search tool
      3. LLM synthesises a final answer
    """
    llm, search_func = websearch_agent

    if search_func is None:
        # No search tool — use LLM knowledge directly
        return llm.invoke([
            SystemMessage("Answer the following question using your training knowledge. Be thorough."),
            HumanMessage(query)
        ]).content

    # Step 1 — Plan search queries
    plan = llm.invoke([
        SystemMessage(
            "You are a research assistant. Given a user question, output 1-3 concise, "
            "targeted search queries that together cover the question. "
            "Write ONLY the queries, one per line. No numbering or explanation."
        ),
        HumanMessage(query)
    ])
    queries = [q.strip() for q in plan.content.strip().split("\n") if q.strip()][:3]

    # Step 2 — Execute searches
    blocks = []
    for sq in queries:
        try:
            result = search_func(sq)
            blocks.append(f"Search: {sq}\nResult:\n{result}")
        except Exception as e:
            blocks.append(f"Search: {sq}\nResult: [failed — {str(e)[:80]}]")

    combined = "\n\n---\n\n".join(blocks)

    # Step 3 — Synthesise
    return llm.invoke([
        SystemMessage(
            "You are a helpful assistant. Use the web search results to give a clear, "
            "accurate, comprehensive answer. Synthesise — do not just repeat snippets."
        ),
        HumanMessage(f"Original question: {query}\n\nSearch results:\n{combined}")
    ]).content


# ═══════════════════════════════════════════════════════════════════════════════
# Workflow
# ═══════════════════════════════════════════════════════════════════════════════

def build_workflow(orchestrator_llm, math_science_agent, code_agent, websearch_agent):

    # ── Classifier ────────────────────────────────────────────────────────────
    def classify_query(state: AgentState) -> Dict:
        query = state["query"]

        # Stage 1 — keyword rules (free, instant)
        category = _keyword_classify(query)
        if category:
            return {"category": category, "agents_tried": []}

        # Stage 2 — llama-3.3-70b for ambiguous queries
        history = format_conversation_history(state["conversation_history"])
        result = orchestrator_llm.invoke([
            SystemMessage(
                "You are a precise query classifier. Classify the user query into exactly one category.\n\n"
                "CATEGORIES:\n"
                "  code         — Any request for code, programming, implementation, algorithms,\n"
                "                 data structures, competitive/LeetCode problems, debugging,\n"
                "                 or any question that involves writing source code in any language.\n"
                "  math/science — Mathematics, physics, chemistry, biology, equations,\n"
                "                 derivations, proofs, scientific concepts, calculations.\n"
                "  websearch    — Current events, news, history, general knowledge, opinions,\n"
                "                 or anything that does not fit the above two.\n\n"
                "HARD RULES (override everything else):\n"
                "  • 'write/give/create/implement' + technical task     → code\n"
                "  • Any programming language name in the query         → code\n"
                "  • LeetCode / HackerRank / competitive problem        → code\n"
                "  • Derivation / proof / scientific formula            → math/science\n\n"
                "Respond with ONLY one word: code | math/science | websearch"
            ),
            HumanMessage(f"Recent context:\n{history}\n\nQuery to classify: {query}")
        ])

        raw = result.content.strip().lower().strip("'\".,")
        if   "math" in raw or "science" in raw: category = "math/science"
        elif "code" in raw or "program" in raw: category = "code"
        elif "web"  in raw or "search"  in raw: category = "websearch"
        else:                                   category = "websearch"

        return {"category": category, "agents_tried": []}

    # ── Router ────────────────────────────────────────────────────────────────
    def route_to_agent(state: AgentState) -> Literal["math_science", "code", "websearch"]:
        c = state["category"]
        if c == "math/science": return "math_science"
        if c == "code":         return "code"
        return "websearch"

    # ── Math / Science ────────────────────────────────────────────────────────
    def process_math_science(state: AgentState) -> Dict:
        llm, wolfram_func = math_science_agent
        agents_tried = list(state.get("agents_tried", []))
        history = format_conversation_history(state["conversation_history"])
        query   = state["query"]

        label = "🧪 Math/Science Agent" + (" + Wolfram Alpha" if wolfram_func else "")
        agents_tried.append(label)

        try:
            wolfram_context = ""
            if wolfram_func:
                try:
                    wolfram_context = f"\nWolfram Alpha result: {wolfram_func(query)}\n"
                except Exception:
                    pass   # don't fail the whole node

            response = llm.invoke([
                SystemMessage(
                    "You are an expert mathematician and scientist. "
                    "Solve problems with clear step-by-step working. "
                    "Show every algebraic step in derivations. Be precise and rigorous."
                ),
                HumanMessage(
                    f"Conversation history:\n{history}\n"
                    f"{wolfram_context}\n"
                    f"Question: {query}\n\n"
                    "Provide a complete, accurate, well-explained answer."
                )
            ]).content

            return {"response": f"{label}:\n\n{response}", "agents_tried": agents_tried}

        except Exception as e:
            agents_tried.append("🔍 Web Search Agent (fallback)")
            web = _run_web_search(websearch_agent, f"Solve this math/science problem: {query}")
            return {
                "response": (
                    f"{label}: Failed — `{str(e)[:120]}`\n\n"
                    f"🔍 Web Search Agent (fallback):\n\n{web}"
                ),
                "agents_tried": agents_tried,
            }

    # ── Code ──────────────────────────────────────────────────────────────────
    def process_code(state: AgentState) -> Dict:
        llm = code_agent
        agents_tried = list(state.get("agents_tried", []))
        history = format_conversation_history(state["conversation_history"])
        query   = state["query"]

        agents_tried.append("💻 Code Agent")

        try:
            response = llm.invoke([
                SystemMessage(
                    "You are an expert software engineer and competitive programmer.\n"
                    "Structure every answer as:\n"
                    "  1. Approach  — explain the algorithm / idea clearly\n"
                    "  2. Code      — complete, runnable, well-commented implementation\n"
                    "  3. Complexity — time and space analysis\n"
                    "Write clean, production-quality code with clear variable names."
                ),
                HumanMessage(
                    f"Conversation history:\n{history}\n\n"
                    f"Problem:\n{query}\n\n"
                    "Provide a complete, working solution."
                )
            ]).content

            return {"response": f"💻 Code Agent:\n\n{response}", "agents_tried": agents_tried}

        except Exception as e:
            agents_tried.append("🔍 Web Search Agent (fallback)")
            web = _run_web_search(websearch_agent, f"Code solution for: {query}")
            return {
                "response": (
                    f"💻 Code Agent: Failed — `{str(e)[:120]}`\n\n"
                    f"🔍 Web Search Agent (fallback):\n\n{web}"
                ),
                "agents_tried": agents_tried,
            }

    # ── Web Search ────────────────────────────────────────────────────────────
    def process_websearch(state: AgentState) -> Dict:
        agents_tried = list(state.get("agents_tried", []))
        agents_tried.append("🔍 Web Search Agent")
        try:
            response = _run_web_search(websearch_agent, state["query"])
            return {"response": f"🔍 Web Search Agent:\n\n{response}", "agents_tried": agents_tried}
        except Exception as e:
            llm, _ = websearch_agent
            fallback = llm.invoke([
                SystemMessage("Answer from your training knowledge."),
                HumanMessage(state["query"])
            ]).content
            return {
                "response": f"🔍 Web Search Agent (offline):\n\n{fallback}",
                "agents_tried": agents_tried,
            }

    # ── Build graph ───────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("classifier",   classify_query)
    workflow.add_node("math_science", process_math_science)
    workflow.add_node("code",         process_code)
    workflow.add_node("websearch",    process_websearch)

    workflow.add_conditional_edges(
        "classifier", route_to_agent,
        {"math_science": "math_science", "code": "code", "websearch": "websearch"}
    )
    workflow.add_edge("math_science", END)
    workflow.add_edge("code",         END)
    workflow.add_edge("websearch",    END)
    workflow.set_entry_point("classifier")

    return workflow.compile()