import os
from typing import Dict, TypedDict, Literal, List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langgraph.graph import StateGraph, END

# ── State Schema ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    category: str
    response: str
    agents_tried: List[str]           # NEW: tracks every agent that ran
    conversation_history: List[Dict[str, str]]


# ── LLM Factory ─────────────────────────────────────────────────────────────
def create_llms():
    general_llm      = ChatGroq(temperature=0.3, model="llama-3.1-8b-instant")
    math_science_llm = ChatGroq(temperature=0.1, model="meta-llama/llama-4-scout-17b-16e-instruct")
    code_llm         = ChatGroq(temperature=0.1, model="openai/gpt-oss-20b")
    return general_llm, math_science_llm, code_llm


# ── Orchestrator ─────────────────────────────────────────────────────────────
def create_orchestrator(general_llm):
    orchestrator_prompt = PromptTemplate.from_template("""
You are a query classifier. Classify the user query into exactly one category:

- 'math/science' : mathematics, physics, chemistry, biology, equations, derivations, proofs, research
- 'code'         : programming, coding, algorithms, data structures, debugging, LeetCode problems, any request for code
- 'websearch'    : current events, news, history, general knowledge, opinions

Rules:
- If the query explicitly asks for code, an implementation, or a programming solution → ALWAYS classify as 'code'
- If the query is about deriving an equation or formula → classify as 'math/science'
- When in doubt between code and math/science, prefer 'code' if there is any coding element

Previous conversation:
{conversation_history}

Current Query: {query}

Respond with ONLY one of: 'math/science', 'code', 'websearch'
""")
    # Use the new pipe syntax instead of deprecated LLMChain
    return orchestrator_prompt | general_llm


# ── Math / Science Agent ─────────────────────────────────────────────────────
def create_math_science_agent(math_science_llm):
    """Returns a tuple (mode, executor) — 'react' if Wolfram available, else 'direct'."""
    if "WOLFRAM_ALPHA_APPID" in os.environ:
        wolfram = WolframAlphaAPIWrapper()
        wolfram_tool = WolframAlphaQueryRun(api_wrapper=wolfram)
        tools = [
            Tool(
                name="Wolfram Alpha",
                func=wolfram_tool.run,
                description="Use for mathematical calculations, equations, unit conversions, and scientific facts."
            )
        ]
        agent = initialize_agent(
            tools=tools,
            llm=math_science_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=45,
        )
        return ("react", agent)
    return ("direct", math_science_llm)


# ── Code Agent ───────────────────────────────────────────────────────────────
def create_code_agent(code_llm):
    """Always uses direct LLM — placeholder tools cause ReAct parsing failures."""
    return ("direct", code_llm)


# ── Web Search Agent ─────────────────────────────────────────────────────────
def create_websearch_agent(general_llm):
    search_tools = []
    if "SERPAPI_API_KEY" in os.environ:
        search = SerpAPIWrapper()
        search_tools.append(Tool(
            name="Web Search",
            func=search.run,
            description="Searches the web for relevant, up-to-date information."
        ))
    else:
        try:
            ddg = DuckDuckGoSearchRun(timeout=10)
            search_tools.append(Tool(
                name="Web Search",
                func=ddg.run,
                description="Searches the web for relevant information."
            ))
        except Exception:
            search_tools.append(Tool(
                name="Web Search",
                func=lambda x: "Web search unavailable.",
                description="Searches the web."
            ))

    search_tools.append(Tool(
        name="Knowledge Base",
        func=lambda x: general_llm.invoke(
            f"Answer this question using your training knowledge: {x}"
        ).content,
        description="Uses internal knowledge when web search is unavailable."
    ))

    agent = initialize_agent(
        tools=search_tools,
        llm=general_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        max_execution_time=45,
    )
    return ("react", agent)


# ── Helpers ──────────────────────────────────────────────────────────────────
def format_conversation_history(history):
    if not history:
        return "No previous conversation."
    lines = []
    for i, ex in enumerate(history):
        lines += [
            f"Exchange {i+1}:",
            f"  User: {ex['query']}",
            f"  Agent ({ex['category']}): {ex['response'][:300]}...",
        ]
    return "\n".join(lines)


def _invoke_web_fallback(websearch_agent, query: str) -> str:
    """Run the web search agent and return its response string."""
    mode, executor = websearch_agent
    try:
        return executor.run(query)
    except Exception as e:
        return f"Web search also failed: {str(e)}"


# ── Workflow Builder ──────────────────────────────────────────────────────────
def build_workflow(orchestrator_chain, math_science_agent, code_agent, websearch_agent):

    def classify_query(state: AgentState) -> Dict:
        formatted_history = format_conversation_history(state["conversation_history"])
        result = orchestrator_chain.invoke({
            "query": state["query"],
            "conversation_history": formatted_history
        })
        # result is an AIMessage when using pipe syntax
        category = result.content.strip().lower()
        # Sanitise
        if category not in ["math/science", "code", "websearch"]:
            category = "websearch"
        return {"category": category, "agents_tried": []}

    def route_to_agent(state: AgentState) -> Literal["math_science", "code", "websearch"]:
        c = state["category"]
        if c == "math/science":
            return "math_science"
        elif c == "code":
            return "code"
        return "websearch"

    # ── Math / Science node ──────────────────────────────────────────────────
    def process_math_science(state: AgentState) -> Dict:
        formatted_history = format_conversation_history(state["conversation_history"])
        mode, executor = math_science_agent
        agents_tried = list(state.get("agents_tried", []))

        system_prompt = (
            "You are an expert mathematician and scientist. "
            "Solve problems step-by-step with clear working. "
            "For derivations, show every algebraic step. "
            "Be precise and rigorous."
        )
        context_query = (
            f"Previous conversation:\n{formatted_history}\n\n"
            f"Current question: {state['query']}\n\n"
            "Provide a complete, accurate answer."
        )

        # ── Try specialized math agent first ────────────────────────────────
        primary_label = "🧪 Math/Science Agent (Wolfram+LLM)" if mode == "react" else "🧪 Math/Science Agent (LLM)"
        agents_tried.append(primary_label)
        try:
            if mode == "react":
                response = executor.run(context_query)
            else:
                messages = [("system", system_prompt), ("human", context_query)]
                response = executor.invoke(messages).content

            return {
                "response": f"{primary_label}:\n\n{response}",
                "agents_tried": agents_tried
            }

        except Exception as primary_err:
            # ── Fallback: web search ─────────────────────────────────────────
            agents_tried.append("🔍 Web Search Agent (fallback)")
            web_response = _invoke_web_fallback(
                websearch_agent,
                f"Solve this math/science problem: {state['query']}"
            )
            return {
                "response": (
                    f"{primary_label}: Could not solve directly "
                    f"(`{str(primary_err)[:120]}`)\n\n"
                    f"🔍 Web Search Agent (fallback):\n\n{web_response}"
                ),
                "agents_tried": agents_tried
            }

    # ── Code node ────────────────────────────────────────────────────────────
    def process_code(state: AgentState) -> Dict:
        formatted_history = format_conversation_history(state["conversation_history"])
        mode, executor = code_agent
        agents_tried = list(state.get("agents_tried", []))

        system_prompt = (
            "You are an expert software engineer. "
            "Write clean, well-commented, production-quality code. "
            "Explain your approach, provide the full implementation, "
            "and include a brief complexity analysis."
        )
        context_query = (
            f"Previous conversation:\n{formatted_history}\n\n"
            f"Current question: {state['query']}\n\n"
            "Provide a complete, working solution."
        )

        # ── Try specialized code agent first ────────────────────────────────
        agents_tried.append("💻 Code Agent (LLM)")
        try:
            messages = [("system", system_prompt), ("human", context_query)]
            response = executor.invoke(messages).content
            return {
                "response": f"💻 Code Agent (LLM):\n\n{response}",
                "agents_tried": agents_tried
            }

        except Exception as primary_err:
            # ── Fallback: web search ─────────────────────────────────────────
            agents_tried.append("🔍 Web Search Agent (fallback)")
            web_response = _invoke_web_fallback(
                websearch_agent,
                f"Provide a code solution for: {state['query']}"
            )
            return {
                "response": (
                    f"💻 Code Agent: Could not solve directly "
                    f"(`{str(primary_err)[:120]}`)\n\n"
                    f"🔍 Web Search Agent (fallback):\n\n{web_response}"
                ),
                "agents_tried": agents_tried
            }

    # ── Web Search node (primary) ─────────────────────────────────────────────
    def process_websearch(state: AgentState) -> Dict:
        formatted_history = format_conversation_history(state["conversation_history"])
        mode, executor = websearch_agent
        agents_tried = list(state.get("agents_tried", []))
        agents_tried.append("🔍 Web Search Agent")

        context_query = (
            f"Previous conversation:\n{formatted_history}\n\n"
            f"Current question: {state['query']}\n\n"
            "Search for current information and give a comprehensive answer."
        )
        try:
            response = executor.run(context_query)
            return {
                "response": f"🔍 Web Search Agent:\n\n{response}",
                "agents_tried": agents_tried
            }
        except Exception as e:
            _, llm = code_agent  # borrow any LLM for offline fallback
            fallback = llm.invoke(
                f"Answer from your training knowledge: {state['query']}"
            ).content
            return {
                "response": f"🔍 Web Search Agent (offline):\n\n{fallback}",
                "agents_tried": agents_tried
            }

    # ── Build Graph ───────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("classifier",   classify_query)
    workflow.add_node("math_science", process_math_science)
    workflow.add_node("code",         process_code)
    workflow.add_node("websearch",    process_websearch)

    workflow.add_conditional_edges(
        "classifier",
        route_to_agent,
        {"math_science": "math_science", "code": "code", "websearch": "websearch"}
    )
    workflow.add_edge("math_science", END)
    workflow.add_edge("code",         END)
    workflow.add_edge("websearch",    END)
    workflow.set_entry_point("classifier")

    return workflow.compile()