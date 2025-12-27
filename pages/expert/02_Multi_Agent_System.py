import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from lib.helper_streamlit import select_model

st.set_page_config(page_title="Multi-Agent System | Expert", page_icon="👥", layout="wide")
st.title("👥 Multi-Agent System")
st.caption("Coordinate multiple AI agents for complex tasks.")

st.markdown("""
Multi-agent systems use specialized agents working together.
Each agent has a specific role and expertise.
""")

with st.sidebar:
    st.header("Configuration")
    model_name = select_model(key="multi_agent_model", location="sidebar")

st.subheader("Example: Research Team")

st.info("""
**Agents:**
- **Researcher**: Gathers information
- **Analyzer**: Analyzes data
- **Writer**: Creates final report
""")

topic = st.text_input("Research topic:", placeholder="renewable energy")

if st.button("🚀 Run Multi-Agent System"):
    if not topic:
        st.warning("Enter a topic.")
        st.stop()
    
    try:
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        with st.spinner("Agent 1: Researcher gathering info..."):
            researcher_prompt = SystemMessage(content="You are a researcher. Provide 3 key facts.")
            researcher_msg = HumanMessage(content=f"Research: {topic}")
            research = llm.invoke([researcher_prompt, researcher_msg])
            st.markdown("### 📚 Researcher Output:")
            st.write(research.content)
        
        with st.spinner("Agent 2: Analyzer processing..."):
            analyzer_prompt = SystemMessage(content="You are an analyzer. Analyze these facts.")
            analyzer_msg = HumanMessage(content=f"Facts:\n{research.content}\n\nProvide analysis.")
            analysis = llm.invoke([analyzer_prompt, analyzer_msg])
            st.markdown("### 🔍 Analyzer Output:")
            st.write(analysis.content)
        
        with st.spinner("Agent 3: Writer creating report..."):
            writer_prompt = SystemMessage(content="You are a writer. Create a brief report.")
            writer_msg = HumanMessage(content=f"Research:\n{research.content}\n\nAnalysis:\n{analysis.content}\n\nWrite report.")
            report = llm.invoke([writer_prompt, writer_msg])
            st.markdown("### 📝 Final Report:")
            st.success(report.content)
    
    except Exception as exc:
        st.error(f"Error: {exc}")

with st.expander("📚 Multi-Agent Patterns"):
    st.markdown("""
    **Sequential**: Agents work one after another
    **Parallel**: Agents work simultaneously
    **Hierarchical**: Manager coordinates workers
    **Collaborative**: Agents debate and refine
    """)
