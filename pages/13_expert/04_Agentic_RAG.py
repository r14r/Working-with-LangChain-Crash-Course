import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Agentic RAG | Expert", page_icon="🎯", layout="centered")
st.title("🎯 Agentic RAG")
st.caption("RAG system where agents decide when and how to retrieve.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Unlike fixed RAG pipelines, agentic RAG lets the agent:
    - Decide if retrieval is needed
    - Choose what to search for
    - Determine if more info is required
    """)

    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(key="agentic_model", location="sidebar")
        embed_model = select_model(key="agentic_embed", location="sidebar", default_models=["nomic-embed-text"])

    # Initialize knowledge base
    if "agentic_kb" not in st.session_state:
        st.session_state.agentic_kb = None

    if st.button("📚 Load Sample Knowledge Base"):
        docs = [
            Document(page_content="Python is a high-level programming language."),
            Document(page_content="Machine learning is a subset of AI."),
            Document(page_content="LangChain helps build LLM applications."),
            Document(page_content="Vectors represent text as numbers."),
        ]
        embeddings = OllamaEmbeddings(model=embed_model)
        st.session_state.agentic_kb = DocArrayInMemorySearch.from_documents(docs, embeddings)
        st.success("Knowledge base loaded!")

    if st.session_state.agentic_kb:
        # Create retrieval tool
        @tool
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for information."""
            docs = st.session_state.agentic_kb.similarity_search(query, k=2)
            results = "\n".join([d.page_content for d in docs])
            return f"Found: {results}"
    
        question = st.text_input("Ask a question:", placeholder="What is Python?")
    
        if st.button("🤖 Ask Agent"):
            if not question:
                st.warning("Enter a question.")
                st.stop()
        
            try:
                llm = ChatOllama(model=model_name, temperature=0)
            
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are helpful. Use search_knowledge if you need info."),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
            
                agent = create_tool_calling_agent(llm, [search_knowledge], prompt)
                executor = AgentExecutor(agent=agent, tools=[search_knowledge])
            
                with st.spinner("Agent thinking..."):
                    result = executor.invoke({"input": question})
            
                st.success(result["output"])
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    with st.expander("📚 Agentic RAG Benefits"):
        st.markdown("""
        **Advantages:**
        - Only retrieves when needed
        - Can reformulate queries
        - Handles multi-hop reasoning
        - More flexible than fixed pipelines
    
        **Patterns:**
        - Self-ask: Agent asks itself questions
        - ReAct: Reason and Act
        - Multi-query: Multiple searches
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
