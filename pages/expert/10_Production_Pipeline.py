import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lib.helper_streamlit import select_model
import time

st.set_page_config(page_title="Production Pipeline | Expert", page_icon="🏭", layout="wide")
st.title("🏭 Production-Ready RAG Pipeline")
st.caption("Complete pipeline with error handling, logging, and monitoring.")

with st.sidebar:
    st.header("Configuration")
    model_name = select_model(key="prod_model", location="sidebar")
    embed_model = select_model(key="prod_embed", location="sidebar", default_models=["nomic-embed-text"])

if "prod_metrics" not in st.session_state:
    st.session_state.prod_metrics = {
        "queries": 0,
        "avg_response_time": 0,
        "errors": 0
    }

# Metrics dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Queries", st.session_state.prod_metrics["queries"])
with col2:
    st.metric("Avg Response Time", f"{st.session_state.prod_metrics['avg_response_time']:.2f}s")
with col3:
    st.metric("Errors", st.session_state.prod_metrics["errors"])

# Load data
if st.button("📚 Initialize System"):
    with st.spinner("Setting up production pipeline..."):
        docs = [
            Document(page_content="Production systems require error handling."),
            Document(page_content="Monitoring is crucial for ML systems."),
            Document(page_content="Logging helps debug issues."),
        ]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        splits = splitter.split_documents(docs)
        
        embeddings = OllamaEmbeddings(model=embed_model)
        st.session_state.prod_kb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    st.success("System ready!")

# Query interface
if "prod_kb" in st.session_state:
    question = st.text_input("Query:", placeholder="What is important for production?")
    
    if st.button("🔍 Query System"):
        if not question:
            st.warning("Enter a query.")
            st.stop()
        
        start_time = time.time()
        
        try:
            with st.spinner("Processing..."):
                # Log query
                with st.expander("📝 System Logs", expanded=False):
                    st.write(f"[{time.strftime('%H:%M:%S')}] Query received: {question}")
                    
                    # Retrieval
                    docs = st.session_state.prod_kb.similarity_search(question, k=2)
                    st.write(f"[{time.strftime('%H:%M:%S')}] Retrieved {len(docs)} documents")
                    
                    # Generation
                    context = "\n".join([d.page_content for d in docs])
                    
                    llm = ChatOllama(model=model_name, temperature=0)
                    prompt = ChatPromptTemplate.from_template(
                        "Context: {context}\n\nQ: {question}\n\nA:"
                    )
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": question})
                    
                    st.write(f"[{time.strftime('%H:%M:%S')}] Generated answer")
                
                # Display answer
                st.markdown("### Answer:")
                st.success(answer)
                
                # Update metrics
                response_time = time.time() - start_time
                st.session_state.prod_metrics["queries"] += 1
                
                # Running average
                n = st.session_state.prod_metrics["queries"]
                old_avg = st.session_state.prod_metrics["avg_response_time"]
                st.session_state.prod_metrics["avg_response_time"] = (
                    (old_avg * (n - 1) + response_time) / n
                )
                
                st.info(f"Response time: {response_time:.2f}s")
                st.rerun()
        
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.session_state.prod_metrics["errors"] += 1
            
            with st.expander("📝 Error Log"):
                st.write(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(exc)}")

with st.expander("📚 Production Best Practices"):
    st.markdown("""
    **Essential Components:**
    - ✅ Error handling
    - ✅ Logging
    - ✅ Monitoring/metrics
    - ✅ Caching
    - ✅ Rate limiting
    - ✅ Testing
    - ✅ Documentation
    
    **Reliability:**
    - Graceful degradation
    - Retry logic
    - Fallback responses
    - Health checks
    
    **Performance:**
    - Response time tracking
    - Resource monitoring
    - Cost optimization
    - Load balancing
    """)
