import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Agent Basics | Advanced", page_icon="🤖", layout="centered")

st.title("🤖 LangChain Agents")
st.caption("Build agents that can use tools autonomously.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Agents can reason about which tools to use and in what order to accomplish a task.
    Unlike simple chains, agents make decisions based on observations.
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(key="agent_model", location="sidebar")

    # Define tools
    @tool
    def search_knowledge(query: str) -> str:
        """Search for information. Use this when you need to look up facts."""
        # Simulated knowledge base
        knowledge = {
            "capital of france": "Paris",
            "population of tokyo": "37 million",
            "python creator": "Guido van Rossum",
            "deepest ocean": "Pacific Ocean - Mariana Trench"
        }
    
        query_lower = query.lower()
        for key, value in knowledge.items():
            if key in query_lower:
                return f"Found: {value}"
        return "No information found."

    @tool
    def calculator(expression: str) -> str:
        """Calculate mathematical expressions. Use this for math problems."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Invalid expression"

    @tool  
    def word_reverser(text: str) -> str:
        """Reverse text. Use this to reverse strings."""
        return text[::-1]

    tools = [search_knowledge, calculator, word_reverser]

    st.subheader("Available Tools")
    for tool_obj in tools:
        st.write(f"**{tool_obj.name}**: {tool_obj.description}")

    # Create agent
    st.subheader("Agent Task")
    task = st.text_area(
        "Give the agent a task:",
        placeholder="What is the capital of France? Then reverse the name.",
        height=100
    )

    if st.button("🤖 Run Agent"):
        if not task.strip():
            st.warning("Please enter a task.")
            st.stop()
    
        try:
            with st.spinner("Agent working..."):
                llm = ChatOllama(model=model_name, temperature=0)
            
                # Create prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant with access to tools."),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
            
                # Create agent
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    max_iterations=5
                )
            
                # Run agent
                result = agent_executor.invoke({"input": task})
            
                st.markdown("### Agent Output:")
                st.success(result['output'])
    
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("Note: Some models may not support tool calling well. Try llama3.1 or newer.")

    with st.expander("📚 How Agents Work"):
        st.markdown("""
        **Agent Loop:**
        1. Receive task
        2. Decide which tool to use
        3. Execute tool
        4. Observe result
        5. Repeat or provide final answer
    
        **Components:**
        - **LLM**: The reasoning engine
        - **Tools**: Available actions
        - **Agent**: Decision-making logic
        - **Executor**: Runs the agent loop
    
        ```python
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        result = executor.invoke({"input": "task"})
        ```
        """)

    with st.expander("💡 Agent vs Chain"):
        st.markdown("""
        **Chain:**
        - Fixed sequence
        - Predictable flow
        - Simpler, faster
    
        **Agent:**
        - Dynamic decisions
        - Can use tools adaptively
        - More powerful, slower
    
        **When to use Agents:**
        - Task requires tool selection
        - Multi-step reasoning needed
        - Unknown number of steps
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
