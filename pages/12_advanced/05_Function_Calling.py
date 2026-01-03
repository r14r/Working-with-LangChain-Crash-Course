import streamlit as st
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Function Calling | Advanced", page_icon="🔧", layout="centered")

st.title("🔧 LLM Function Calling")
st.caption("Learn how LLMs can call functions and tools.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Function calling allows LLMs to interact with external tools and APIs.
    The model decides when and how to call functions based on user input.
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(key="func_model", location="sidebar")

    # Define tools
    @tool
    def calculate_sum(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool
    def calculate_product(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    @tool
    def get_word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    # Available tools
    tools = [calculate_sum, calculate_product, get_word_count]

    st.subheader("Available Tools")
    for tool_obj in tools:
        st.write(f"**{tool_obj.name}**: {tool_obj.description}")

    # User input
    st.subheader("Try Function Calling")
    user_query = st.text_input(
        "Ask something that requires a tool:",
        placeholder="What is 15 + 27?"
    )

    if st.button("🚀 Execute"):
        if not user_query.strip():
            st.warning("Please enter a query.")
            st.stop()
    
        try:
            with st.spinner("Processing..."):
                llm = ChatOllama(model=model_name, temperature=0)
            
                # Bind tools to model
                llm_with_tools = llm.bind_tools(tools)
            
                # Get response
                response = llm_with_tools.invoke([HumanMessage(content=user_query)])
            
                st.markdown("### Model Response:")
            
                # Check if model wants to call a tool
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        st.info(f"🔧 Calling: {tool_call['name']}")
                        st.json(tool_call['args'])
                    
                        # Execute the tool
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                    
                        # Find and execute tool
                        for t in tools:
                            if t.name == tool_name:
                                result = t.invoke(tool_args)
                                st.success(f"Result: {result}")
                                break
                else:
                    st.write(response.content)
    
        except Exception as exc:
            st.error(f"Error: {exc}")

    with st.expander("📚 How Function Calling Works"):
        st.markdown("""
        **Process:**
        1. Define tools with @tool decorator
        2. Bind tools to LLM
        3. LLM decides if/when to call tools
        4. Execute tools and return results
    
        ```python
        @tool
        def my_function(param: int) -> int:
            '''Description of function'''
            return param * 2
    
        llm_with_tools = llm.bind_tools([my_function])
        response = llm_with_tools.invoke("double 5")
        ```
        """)

    with st.expander("💡 Use Cases"):
        st.markdown("""
        - **Calculations**: Math operations
        - **API calls**: Weather, stock prices
        - **Database queries**: Fetch data
        - **File operations**: Read/write files
        - **Web scraping**: Get web content
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
