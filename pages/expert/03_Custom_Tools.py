import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from lib.helper_streamlit import select_model
import json

st.set_page_config(page_title="Custom Tools | Expert", page_icon="🔨", layout="centered")
st.title("🔨 Build Custom Agent Tools")
st.caption("Create specialized tools for agents to use.")

with st.sidebar:
    model_name = select_model(key="tools_model", location="sidebar")

# Define custom tools
@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text. Returns positive, negative, or neutral."""
    # Simplified sentiment analysis
    positive_words = ["good", "great", "excellent", "amazing", "love", "best"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor"]
    
    text_lower = text.lower()
    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words)
    
    if pos_count > neg_count:
        return "Sentiment: Positive"
    elif neg_count > pos_count:
        return "Sentiment: Negative"
    else:
        return "Sentiment: Neutral"

@tool
def count_characters(text: str) -> str:
    """Count characters, words, and sentences in text."""
    chars = len(text)
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Characters: {chars}, Words: {words}, Sentences: {sentences}"

@tool
def extract_keywords(text: str) -> str:
    """Extract potential keywords from text (words longer than 5 chars)."""
    words = text.split()
    keywords = [w.strip('.,!?') for w in words if len(w) > 5]
    return f"Keywords: {', '.join(set(keywords[:10]))}"

tools = [analyze_sentiment, count_characters, extract_keywords]

st.subheader("Available Custom Tools")
for tool_obj in tools:
    st.write(f"**{tool_obj.name}**: {tool_obj.description}")

st.subheader("Use Tools with Agent")
task = st.text_area("Task for agent:", placeholder="Analyze this text: AI is amazing and revolutionary!")

if st.button("🤖 Run Agent"):
    if not task:
        st.warning("Enter a task.")
        st.stop()
    
    try:
        llm = ChatOllama(model=model_name, temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful text analysis assistant."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        with st.spinner("Agent working..."):
            result = executor.invoke({"input": task})
        
        st.success(result["output"])
    
    except Exception as exc:
        st.error(f"Error: {exc}")

with st.expander("📚 Creating Tools"):
    st.markdown("""
    **Tool Definition:**
    ```python
    from langchain_core.tools import tool
    
    @tool
    def my_tool(param: str) -> str:
        '''Tool description for the LLM.'''
        # Implementation
        return result
    ```
    
    **Best Practices:**
    - Clear descriptions
    - Type hints
    - Error handling
    - Deterministic when possible
    """)
