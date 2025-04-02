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

# Load environment variables from .env file
load_dotenv()

# Check if required API keys are present
if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# Set page title and configuration
st.set_page_config(page_title="Multi-Agent System", layout="wide")
st.title("Multi-Agent Query System")

# Initialize session state for conversation history if it doesn't exist
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar with information
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This app demonstrates a multi-agent system built with:
    - LangChain for agent definition
    - LangGraph for workflow orchestration
    - Groq for LLM API access
    - Streamlit for the user interface
    
    Each agent specializes in a different domain to provide accurate responses.
    """)
    
    # Show API status
    st.subheader("API Status")
    st.success("✅ Groq API Key detected")
    
    if "WOLFRAM_ALPHA_APPID" in os.environ:
        st.success("✅ Wolfram Alpha API detected")
    else:
        st.warning("⚠️ Wolfram Alpha API not configured (optional)")
    
    # Add context management options
    st.subheader("Context Management")
    if st.button("Clear Conversation History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")

# Initialize LLMs
general_llm, math_science_llm, code_llm = create_llms()

# Create agents
orchestrator_chain = create_orchestrator(general_llm)
math_science_executor = create_math_science_agent(math_science_llm)
code_executor = create_code_agent(code_llm)
websearch_executor = create_websearch_agent(general_llm)

# Build workflow
agent_graph = build_workflow(
    orchestrator_chain, 
    math_science_executor, 
    code_executor, 
    websearch_executor
)

# Main app content
st.markdown("""
This system routes your questions to specialized agents:
- **Math/Science Agent**: For mathematics, physics, chemistry, etc.
- **Code Agent**: For programming, algorithms, debugging, etc.
- **Web Search Agent**: For general knowledge, history, news, etc.

This system maintains context between queries so you can ask follow-up questions.
""")

# Display conversation history
if st.session_state.conversation_history:
    with st.expander("Conversation History", expanded=False):
        for i, exchange in enumerate(st.session_state.conversation_history):
            st.markdown(f"**Query {i+1}:** {exchange['query']}")
            st.markdown(f"**Response:** {exchange['response']}")
            st.markdown("---")

# User input
query = st.text_input("Enter your question:", key="user_query")

if st.button("Submit") or st.session_state.get("automatic_submit", False):
    if query:
        with st.spinner("Agents are working on your query..."):
            try:
                # Execute workflow with conversation history
                result = agent_graph.invoke({
                    "query": query, 
                    "category": "", 
                    "response": "",
                    "conversation_history": st.session_state.conversation_history
                })
                
                # Display results
                st.subheader("Result")
                st.markdown(result["response"])
                
                # Store in session state for this exchange
                exchange = {
                    "query": query,
                    "category": result["category"],
                    "response": result["response"]
                }
                
                # Add to conversation history
                st.session_state.conversation_history.append(exchange)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question first.")

# Display previous results if available from current exchange
if st.session_state.conversation_history:
    latest = st.session_state.conversation_history[-1]
    with st.expander("Last Query Details", expanded=False):
        st.info(f"Query classified as: {latest['category']}")