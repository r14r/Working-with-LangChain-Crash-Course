import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Vector Stores
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Vector Stores | Beginner",
    page_icon="🗄️",
    layout="centered",
)

st.title("🗄️ Vector Stores")
st.caption("Learn how to store and search documents using embeddings.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    Vector stores save document embeddings and enable semantic search. Instead 
    of searching for exact text matches, you search by meaning.

    **What you'll learn:**
    1. Create a vector store from documents  
    2. Perform similarity search  
    3. Retrieve relevant documents
    """
    )

    # --------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
    
        from lib.helper_streamlit import select_model
        embedding_model = select_model(
            key="vs_embedding_model",
            location="sidebar",
            label="Embedding Model",
            default_models=["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
        )
    
        k_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=5,
            value=3,
            help="How many similar documents to retrieve"
        )

    # --------------------------------------------------------------------------------------
    # Initialize Vector Store
    # --------------------------------------------------------------------------------------
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.documents = []

    # --------------------------------------------------------------------------------------
    # Add Documents
    # --------------------------------------------------------------------------------------
    st.subheader("1) Add Documents to Vector Store")

    # Sample documents
    sample_docs = [
        "LangChain is a framework for building applications with large language models.",
        "Python is a popular programming language for data science and machine learning.",
        "Vector databases store embeddings and enable semantic search.",
        "Machine learning models can be trained on large datasets.",
        "Natural language processing helps computers understand human language.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Embeddings convert text into numerical vectors.",
        "Retrieval-Augmented Generation combines search with language models.",
    ]

    doc_input_method = st.radio(
        "Choose how to add documents:",
        ["Use Sample Documents", "Add Custom Document"]
    )

    if doc_input_method == "Use Sample Documents":
        st.info("Click below to load sample documents into the vector store")
    
        if st.button("📥 Load Sample Documents"):
            try:
                with st.spinner("Creating vector store..."):
                    # Create documents
                    documents = [Document(page_content=text) for text in sample_docs]
                
                    # Create embeddings
                    embeddings = OllamaEmbeddings(model=embedding_model)
                
                    # Create vector store
                    st.session_state.vectorstore = DocArrayInMemorySearch.from_documents(
                        documents,
                        embeddings
                    )
                    st.session_state.documents = sample_docs
            
                st.success(f"✅ Loaded {len(sample_docs)} documents into vector store")
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:  # Custom document
        new_doc = st.text_area(
            "Enter document text:",
            placeholder="Type or paste your document here...",
            height=100
        )
    
        if st.button("➕ Add Document"):
            if not new_doc.strip():
                st.warning("Please enter document text.")
                st.stop()
        
            try:
                with st.spinner("Adding document..."):
                    embeddings = OllamaEmbeddings(model=embedding_model)
                
                    if st.session_state.vectorstore is None:
                        # Create new vector store
                        st.session_state.vectorstore = DocArrayInMemorySearch.from_documents(
                            [Document(page_content=new_doc)],
                            embeddings
                        )
                        st.session_state.documents = [new_doc]
                    else:
                        # Add to existing vector store
                        st.session_state.vectorstore.add_documents(
                            [Document(page_content=new_doc)]
                        )
                        st.session_state.documents.append(new_doc)
            
                st.success(f"✅ Document added! Total: {len(st.session_state.documents)}")
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # Display current documents
    if st.session_state.documents:
        with st.expander(f"📚 View Stored Documents ({len(st.session_state.documents)})"):
            for i, doc in enumerate(st.session_state.documents, 1):
                st.write(f"**{i}.** {doc[:100]}..." if len(doc) > 100 else f"**{i}.** {doc}")

    # --------------------------------------------------------------------------------------
    # Search Vector Store
    # --------------------------------------------------------------------------------------
    if st.session_state.vectorstore is not None:
        st.subheader("2) Search Vector Store")
    
        st.info("Enter a query to find similar documents using semantic search")
    
        query = st.text_input(
            "Search query:",
            placeholder="What is machine learning?"
        )
    
        if st.button("🔍 Search"):
            if not query.strip():
                st.warning("Please enter a search query.")
                st.stop()
        
            try:
                with st.spinner("Searching..."):
                    # Perform similarity search
                    results = st.session_state.vectorstore.similarity_search(
                        query,
                        k=k_results
                    )
            
                st.markdown(f"### Found {len(results)} similar documents:")
            
                for i, doc in enumerate(results, 1):
                    st.markdown(f"**Result {i}:**")
                    st.write(doc.page_content)
                
                    if i < len(results):
                        st.divider()
            
                # Advanced: Show with scores
                with st.expander("🔢 View with Similarity Scores"):
                    results_with_scores = st.session_state.vectorstore.similarity_search_with_score(
                        query,
                        k=k_results
                    )
                
                    for i, (doc, score) in enumerate(results_with_scores, 1):
                        st.markdown(f"**Result {i}** (Score: {score:.4f})")
                        st.write(doc.page_content)
                    
                        if i < len(results_with_scores):
                            st.divider()
                
                    st.info("Lower scores indicate higher similarity")
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:
        st.info("👆 Add documents to the vector store first")

    # Clear vector store
    if st.session_state.vectorstore is not None:
        st.subheader("3) Manage Vector Store")
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Vector Store"):
                st.session_state.vectorstore = None
                st.session_state.documents = []
                st.rerun()
    
        with col2:
            st.metric("Documents Stored", len(st.session_state.documents))

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 What you learned"):
        st.markdown("""
        **Key Concepts:**
    
        - **Vector Store**: Database for storing and searching embeddings
        - **Similarity Search**: Find documents by semantic meaning
        - **k Parameter**: Number of similar results to return
    
        **How It Works:**
        ```python
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import DocArrayInMemorySearch
        from langchain_core.documents import Document
    
        # Create documents
        docs = [
            Document(page_content="Text 1"),
            Document(page_content="Text 2"),
        ]
    
        # Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
        # Create vector store
        vectorstore = DocArrayInMemorySearch.from_documents(
            docs,
            embeddings
        )
    
        # Search
        results = vectorstore.similarity_search("query", k=3)
    
        # Search with scores
        results = vectorstore.similarity_search_with_score("query", k=3)
        ```
    
        **Vector Store Types:**
    
        | Store | Type | Best For |
        |-------|------|----------|
        | `DocArrayInMemorySearch` | In-memory | Prototyping, small datasets |
        | `Chroma` | Persistent | Development, medium datasets |
        | `FAISS` | Persistent | Fast search, large datasets |
        | `Pinecone` | Cloud | Production, distributed |
        | `Qdrant` | Self-hosted/Cloud | Production, advanced features |
    
        **Why Use Vector Stores?**
        - **Semantic search**: Find by meaning, not just keywords
        - **Fast retrieval**: Efficient similarity matching
        - **Scalable**: Handle large document collections
        - **RAG foundation**: Essential for Q&A systems
        """)

    with st.expander("💡 Understanding Similarity Search"):
        st.markdown("""
        **How It Works:**
        1. Your query is converted to an embedding vector
        2. The vector store compares it to all stored vectors
        3. Returns the k most similar documents
    
        **Similarity Metrics:**
        - **Cosine Similarity**: Measures angle between vectors
        - **Euclidean Distance**: Measures straight-line distance
        - **Dot Product**: Combines magnitude and angle
    
        **Example Searches:**
        - Query: "What is AI?"
          - Matches: Documents about artificial intelligence
        - Query: "How to code in Python?"
          - Matches: Programming tutorials
        - Query: "Machine learning basics"
          - Matches: ML introduction documents
    
        **Tips:**
        - More documents = better coverage
        - Similar documents = redundant results
        - Diverse documents = broader search capability
        """)

    with st.expander("🔧 Next Steps"):
        st.markdown("""
        Now that you understand vector stores:
        - **10_Simple_RAG.py**: Build a complete Q&A system
        - **Advanced/03_MultiQuery_RAG.py**: Improve retrieval
    
        **Production Considerations:**
        - Use persistent stores (Chroma, FAISS)
        - Index optimization for large datasets
        - Metadata filtering
        - Hybrid search (vector + keyword)
        - Regular index updates
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
