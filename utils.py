import os
from typing import Dict, TypedDict, Literal, List
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langgraph.graph import StateGraph, END

# Define state schema for LangGraph with conversation history
class AgentState(TypedDict):
    query: str
    category: str
    response: str
    conversation_history: List[Dict[str, str]]

def create_llms():
    """Initialize LLMs with Groq API using environment variables"""
    # Initialize LLMs with different configurations
    general_llm = ChatGroq(temperature=0.3, model="llama3-70b-8192")
    math_science_llm = ChatGroq(temperature=0.1, model="llama3-70b-8192")
    code_llm = ChatGroq(temperature=0.1, model="llama3-70b-8192")
    
    return general_llm, math_science_llm, code_llm

def create_orchestrator(general_llm):
    """Create the orchestrator agent that classifies queries"""
    orchestrator_prompt = PromptTemplate.from_template("""
    You are a query classifier. Your job is to classify user queries into one of these categories:
    - 'math/science': Questions about mathematics, physics, chemistry, biology, etc.
    - 'code': Questions about programming, coding, algorithms, debugging, etc.
    - 'websearch': General knowledge questions, current events, history, etc.

    Previous conversation context:
    {conversation_history}

    Current Query: {query}

    Consider both the current query and any relevant context from previous interactions.
    Respond with only one word: 'math/science', 'code', or 'websearch'.
    """)

    return LLMChain(llm=general_llm, prompt=orchestrator_prompt)

def create_math_science_agent(math_science_llm):
    """Create the math/science specialized agent"""
    wolfram_tool = None
    if "WOLFRAM_ALPHA_APPID" in os.environ:
        wolfram = WolframAlphaAPIWrapper()
        wolfram_tool = WolframAlphaQueryRun(api_wrapper=wolfram)

    math_science_tools = []
    if wolfram_tool:
        math_science_tools.append(
            Tool(
                name="Wolfram Alpha",
                func=wolfram_tool.run,
                description="Useful for mathematical calculations, scientific information, and precise factual queries."
            )
        )
    
    # If no Wolfram Alpha, create a dummy calculator tool to ensure we have tools
    if not math_science_tools:
        math_science_tools.append(
            Tool(
                name="Calculator",
                func=lambda x: eval(x),
                description="Useful for performing basic mathematical calculations."
            )
        )

    # Use initialize_agent instead of create_react_agent with increased iteration limit
    return initialize_agent(
        tools=math_science_tools,
        llm=math_science_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,
        max_execution_time=120
    )

def create_code_agent(code_llm):
    """Create the code specialized agent"""
    # Create a dummy tool to satisfy ReAct agent requirements
    code_tools = [
        Tool(
            name="CodeExecutor",
            func=lambda x: "This is a placeholder for code execution. In a real environment, this would execute the code.",
            description="A tool to execute code and return the result."
        )
    ]

    # Use initialize_agent instead of create_react_agent with increased limits
    return initialize_agent(
        tools=code_tools,
        llm=code_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=60
    )

def create_websearch_agent(general_llm):
    """Create the web search specialized agent"""
    # Try to create a SerpAPI search tool if API key is available
    search_tools = []
    
    if "SERPAPI_API_KEY" in os.environ:
        # Use SerpAPI if key is available
        search = SerpAPIWrapper()
        search_tools.append(
            Tool(
                name="Web Search",
                func=search.run,
                description="Searches the web for relevant information on a wide range of topics."
            )
        )
    else:
        try:
            # Fall back to DuckDuckGo with increased timeout
            search_tool = DuckDuckGoSearchRun(timeout=10)
            search_tools.append(
                Tool(
                    name="Web Search",
                    func=search_tool.run,
                    description="Searches the web for relevant information on a wide range of topics."
                )
            )
        except Exception:
            # If both fail, create a dummy search tool
            search_tools.append(
                Tool(
                    name="Web Search",
                    func=lambda x: "I'm unable to search the web right now. I'll try to answer based on my own knowledge.",
                    description="Searches the web for relevant information on a wide range of topics."
                )
            )
    
    # Define a backup tool that doesn't require web access
    backup_tool = Tool(
        name="Knowledge Base",
        func=lambda x: general_llm.invoke(f"Please provide information about: {x}").content,
        description="Uses internal knowledge to answer questions when web search is unavailable."
    )
    
    search_tools.append(backup_tool)

    # Use initialize_agent instead of create_react_agent with increased limits
    return initialize_agent(
        tools=search_tools,
        llm=general_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,
        max_execution_time=60
    )

def format_conversation_history(history):
    """Format conversation history for prompt inclusion"""
    if not history:
        return "No previous conversation."
    
    formatted = []
    for i, exchange in enumerate(history):
        formatted.append(f"Exchange {i+1}:")
        formatted.append(f"User: {exchange['query']}")
        formatted.append(f"Agent ({exchange['category']}): {exchange['response']}")
    
    return "\n".join(formatted)

def build_workflow(orchestrator_chain, math_science_executor, code_executor, websearch_executor):
    """Build the LangGraph workflow with context awareness"""
    
    def classify_query(state: AgentState) -> Dict:
        """Classify the query and determine which agent to route to."""
        formatted_history = format_conversation_history(state["conversation_history"])
        category = orchestrator_chain.run(
            query=state["query"], 
            conversation_history=formatted_history
        ).strip().lower()
        
        if category not in ["math/science", "code", "websearch"]:
            category = "websearch"  # Default to websearch for unrecognized categories
        return {"category": category}

    def route_to_agent(state: AgentState) -> Literal["math_science", "code", "websearch"]:
        """Route to the appropriate agent based on category."""
        category = state["category"]
        if category == "math/science":
            return "math_science"
        elif category == "code":
            return "code"
        else:
            return "websearch"

    def process_math_science(state: AgentState) -> Dict:
        """Process query with math/science agent with context awareness."""
        # Include context in the query sent to the agent
        formatted_history = format_conversation_history(state["conversation_history"])
        context_query = f"""
Previous conversation: 
{formatted_history}

Current question: {state["query"]}

Please answer the current question considering any relevant context from the previous conversation.
Take your time with mathematical and scientific problems, especially for complex topics like linear programming.
"""
        try:
            response = math_science_executor.run(context_query)
            return {"response": f"🧪 Math/Science Agent: {response}"}
        except Exception as e:
            # Handle timeout or other exceptions gracefully
            error_message = str(e)
            return {"response": f"🧪 Math/Science Agent: I encountered a complexity limit with this problem. Here's what I know about this topic: {error_message[:200]}... Please consider reformulating or breaking down your question."}

    def process_code(state: AgentState) -> Dict:
        """Process query with code agent with context awareness."""
        formatted_history = format_conversation_history(state["conversation_history"])
        context_query = f"""
Previous conversation: 
{formatted_history}

Current question: {state["query"]}

Please answer the current question considering any relevant context from the previous conversation.
"""
        try:
            response = code_executor.run(context_query)
            return {"response": f"💻 Code Agent: {response}"}
        except Exception as e:
            error_message = str(e)
            return {"response": f"💻 Code Agent: I hit a complexity limit while working on this problem. Please consider simplifying your query or breaking it into smaller parts."}

    def process_websearch(state: AgentState) -> Dict:
        """Process query with websearch agent with context awareness."""
        formatted_history = format_conversation_history(state["conversation_history"])
        context_query = f"""
Previous conversation: 
{formatted_history}

Current question: {state["query"]}

Please answer the current question considering any relevant context from the previous conversation.
If web search is unavailable, please answer using your internal knowledge.
"""
        try:
            response = websearch_executor.run(context_query)
            return {"response": f"🔍 Web Search Agent: {response}"}
        except Exception as e:
            # Fall back to the LLM's built-in knowledge
            fallback_response = general_llm.invoke(f"""
You are a helpful assistant answering a question based on your training knowledge without web search.
The question is: {state["query"]}
Please provide a helpful response using only what you already know.
""").content
            return {"response": f"🔍 Web Search Agent: I'm unable to search the web right now, but here's what I know: {fallback_response}"}

    # Build the LangGraph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classifier", classify_query)
    workflow.add_node("math_science", process_math_science)
    workflow.add_node("code", process_code)
    workflow.add_node("websearch", process_websearch)

    # Add edges
    workflow.add_conditional_edges(
        "classifier",
        route_to_agent,
        {
            "math_science": "math_science",
            "code": "code",
            "websearch": "websearch"
        }
    )

    # Connect all agents to END
    workflow.add_edge("math_science", END)
    workflow.add_edge("code", END)
    workflow.add_edge("websearch", END)

    # Set entry point
    workflow.set_entry_point("classifier")

    # Compile the graph
    return workflow.compile()