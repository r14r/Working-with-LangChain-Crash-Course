import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source
import validators

st.set_page_config(page_title="Web Scraping QA | Advanced", page_icon="🌐", layout="centered")

st.title("🌐 Web Scraping Q&A")
st.caption("Build RAG systems from web content.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Scrape web pages and answer questions about their content.
    Great for documentation, blogs, and news articles.
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        chat_model = select_model(key="web_chat", location="sidebar")
        embed_model = select_model(
            key="web_embed",
            location="sidebar",
            default_models=["nomic-embed-text"]
        )

    # Initialize
    if "web_vectorstore" not in st.session_state:
        st.session_state.web_vectorstore = None
        st.session_state.web_urls = []

    # URL Input
    st.subheader("1) Load Web Pages")
    url = st.text_input("Enter URL:", placeholder="https://example.com")

    if st.button("🌐 Load URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
            st.stop()
    
        if not validators.url(url):
            st.error("Invalid URL format.")
            st.stop()
    
        try:
            with st.spinner(f"Loading {url}..."):
                loader = WebBaseLoader(url)
                docs = loader.load()
            
                st.info(f"Loaded {len(docs)} document(s)")
        
            with st.spinner("Processing content..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                splits = splitter.split_documents(docs)
            
                embeddings = OllamaEmbeddings(model=embed_model)
            
                if st.session_state.web_vectorstore is None:
                    st.session_state.web_vectorstore = DocArrayInMemorySearch.from_documents(
                        splits, embeddings
                    )
                else:
                    st.session_state.web_vectorstore.add_documents(splits)
            
                st.session_state.web_urls.append(url)
        
            st.success(f"✅ Indexed {len(splits)} chunks from {url}")
    
        except Exception as exc:
            st.error(f"Error: {exc}")

    # Show loaded URLs
    if st.session_state.web_urls:
        with st.expander(f"📎 Loaded URLs ({len(st.session_state.web_urls)})"):
            for u in st.session_state.web_urls:
                st.write(f"- {u}")

    # Query
    if st.session_state.web_vectorstore:
        st.subheader("2) Ask Questions")
    
        question = st.text_input("Question:", placeholder="What is this page about?")
    
        if st.button("🔍 Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
                st.stop()
        
            try:
                with st.spinner("Searching..."):
                    docs = st.session_state.web_vectorstore.similarity_search(question, k=3)
            
                with st.expander("📚 Retrieved Content"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content[:300] + "...")
                        st.divider()
            
                with st.spinner("Generating answer..."):
                    context = "\n\n".join(doc.page_content for doc in docs)
                
                    prompt = ChatPromptTemplate.from_template(
                        """Answer based on the web content below.

    Content:
    {context}

    Question: {question}

    Answer:"""
                    )
                
                    llm = ChatOllama(model=chat_model, temperature=0.7)
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": question})
            
                st.markdown("### 💬 Answer:")
                st.write(answer)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:
        st.info("👆 Load a URL first")

    with st.expander("📚 Web Scraping RAG"):
        st.markdown("""
        **Use Cases:**
        - Documentation Q&A
        - News article analysis  
        - Blog content search
        - Research paper summaries
    
        **Implementation:**
        ```python
        from langchain_community.document_loaders import WebBaseLoader
    
        loader = WebBaseLoader("https://example.com")
        docs = loader.load()
    
        # Process as normal RAG
        splits = splitter.split_documents(docs)
        vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
        ```
    
        **Tips:**
        - Some sites block scrapers
        - Check robots.txt
        - Respect rate limits
        - Cache results
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
