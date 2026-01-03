import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Simple RAG (Retrieval-Augmented Generation)
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Simple RAG | Beginner",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 Simple RAG System")
st.caption("Build your first Retrieval-Augmented Generation (RAG) application.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    RAG combines document retrieval with language models to answer questions 
    based on your own documents. This is the foundation for Q&A systems!

    **The RAG Pipeline:**
    1. **Load** documents  
    2. **Split** into chunks  
    3. **Embed** and store in vector database  
    4. **Retrieve** relevant chunks for a query  
    5. **Generate** answer using retrieved context
    """
    )

    # --------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("🔧 Configuration")
    
        chat_model = select_model(
            key="rag_chat_model",
            location="sidebar",
            label="Chat Model"
        )
    
        embed_model = select_model(
            key="rag_embed_model",
            location="sidebar",
            label="Embedding Model",
            default_models=["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
        )
    
        st.divider()
    
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
        k_docs = st.slider("Documents to Retrieve", 1, 5, 3)

    # --------------------------------------------------------------------------------------
    # Initialize Session State
    # --------------------------------------------------------------------------------------
    if "rag_vectorstore" not in st.session_state:
        st.session_state.rag_vectorstore = None
        st.session_state.rag_source_docs = []

    # --------------------------------------------------------------------------------------
    # Step 1: Load Documents
    # --------------------------------------------------------------------------------------
    st.subheader("Step 1: Load Documents")

    doc_method = st.radio(
        "Choose document source:",
        ["Upload File", "Enter Text", "Use Sample"]
    )

    documents_to_process = []

    if doc_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf"]
        )
    
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
            
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
            
                documents_to_process = loader.load()
                os.remove(tmp_path)
            
            except Exception as exc:
                st.error(f"Error loading file: {exc}")

    elif doc_method == "Enter Text":
        text_input = st.text_area(
            "Enter your document text:",
            height=200,
            placeholder="Paste your document here..."
        )
    
        if text_input:
            documents_to_process = [Document(page_content=text_input)]

    else:  # Sample
        sample_text = """Artificial Intelligence (AI) is the simulation of human intelligence by machines. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

    Machine Learning (ML) is a subset of AI that focuses on teaching computers to learn from data without being explicitly programmed. ML algorithms identify patterns in data and make predictions or decisions based on those patterns.

    Deep Learning is a specialized form of machine learning that uses artificial neural networks with multiple layers. These networks can learn complex patterns and representations from large amounts of data. Deep learning has revolutionized fields like computer vision and natural language processing.

    Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. NLP powers applications like chatbots, translation services, sentiment analysis, and text summarization.

    Computer Vision is an AI field focused on enabling computers to understand and interpret visual information from the world. Applications include facial recognition, autonomous vehicles, medical image analysis, and object detection.

    Reinforcement Learning is a type of machine learning where agents learn to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time.

    Large Language Models (LLMs) are AI models trained on vast amounts of text data. They can understand context, generate human-like text, answer questions, and perform various language tasks. Examples include GPT, BERT, and Claude."""
    
        st.info("Using sample text about AI and machine learning")
        with st.expander("📄 View Sample Text"):
            st.write(sample_text)
    
        documents_to_process = [Document(page_content=sample_text)]

    # --------------------------------------------------------------------------------------
    # Step 2: Process and Index
    # --------------------------------------------------------------------------------------
    if documents_to_process:
        if st.button("⚡ Process and Index Documents"):
            try:
                with st.spinner("Step 2: Splitting documents..."):
                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    splits = splitter.split_documents(documents_to_process)
                
                    st.session_state.rag_source_docs = splits
            
                st.success(f"✅ Split into {len(splits)} chunks")
            
                with st.spinner("Step 3: Creating embeddings and vector store..."):
                    # Create embeddings and vector store
                    embeddings = OllamaEmbeddings(model=embed_model)
                    st.session_state.rag_vectorstore = DocArrayInMemorySearch.from_documents(
                        splits,
                        embeddings
                    )
            
                st.success(f"✅ Indexed {len(splits)} chunks in vector store")
            
            except Exception as exc:
                st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Step 3: Query and Answer
    # --------------------------------------------------------------------------------------
    if st.session_state.rag_vectorstore is not None:
        st.subheader("Step 4: Ask Questions")
    
        st.info(f"💡 Vector store ready with {len(st.session_state.rag_source_docs)} chunks")
    
        question = st.text_input(
            "Enter your question:",
            placeholder="What is machine learning?"
        )
    
        if st.button("🔍 Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
                st.stop()
        
            try:
                with st.spinner("Retrieving relevant documents..."):
                    # Retrieve relevant chunks
                    relevant_docs = st.session_state.rag_vectorstore.similarity_search(
                        question,
                        k=k_docs
                    )
            
                st.success(f"✅ Retrieved {len(relevant_docs)} relevant chunks")
            
                with st.expander("📚 View Retrieved Chunks"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content)
                        if i < len(relevant_docs):
                            st.divider()
            
                with st.spinner("Generating answer..."):
                    # Combine chunks into context
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                
                    # Create prompt
                    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below. If you cannot answer the question based on the context, say "I cannot answer this based on the provided documents."

    Context:
    {context}

    Question: {question}

    Answer:""")
                
                    # Create chain
                    llm = ChatOllama(model=chat_model, temperature=0)
                    chain = prompt | llm | StrOutputParser()
                
                    # Get answer
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })
            
                st.markdown("### 💬 Answer:")
                st.write(answer)
            
            except Exception as exc:
                st.error(f"Error: {exc}")
    
        # Clear button
        if st.button("🗑️ Clear Index"):
            st.session_state.rag_vectorstore = None
            st.session_state.rag_source_docs = []
            st.rerun()

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 What you learned"):
        st.markdown("""
        **The Complete RAG Pipeline:**
    
        ```python
        # 1. Load documents
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader("document.txt")
        documents = loader.load()
    
        # 2. Split into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = splitter.split_documents(documents)
    
        # 3. Create vector store
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import DocArrayInMemorySearch
    
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
        # 4. Retrieve relevant docs
        relevant_docs = vectorstore.similarity_search(question, k=3)
    
        # 5. Generate answer
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
    
        context = "\\n\\n".join(doc.page_content for doc in relevant_docs)
    
        prompt = ChatPromptTemplate.from_template(
            "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
        )
    
        llm = ChatOllama(model="llama2")
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": question})
        ```
    
        **Key Components:**
        1. **Document Loader**: Gets text from files
        2. **Text Splitter**: Creates manageable chunks
        3. **Embeddings**: Convert text to vectors
        4. **Vector Store**: Stores and searches embeddings
        5. **Retriever**: Finds relevant chunks
        6. **LLM**: Generates final answer
        """)

    with st.expander("💡 RAG Best Practices"):
        st.markdown("""
        **Chunking Strategy:**
        - **Too small**: Lacks context
        - **Too large**: Noise, less focused
        - **Sweet spot**: 500-700 characters with 50-100 overlap
    
        **Retrieval Settings:**
        - **k=1**: Risk of missing info
        - **k=3-5**: Good balance
        - **k>5**: May include irrelevant docs
    
        **Prompt Engineering:**
        - Explicitly tell model to use provided context
        - Ask model to cite sources
        - Handle cases where answer isn't in context
    
        **Common Issues:**
        - **Hallucination**: Model makes up info not in docs
        - **Solution**: Stricter prompts, lower temperature
    
        - **Poor retrieval**: Doesn't find relevant chunks
        - **Solution**: Better chunking, more context
    
        - **Slow performance**: Takes too long
        - **Solution**: Persistent vector DB, smaller models
        """)

    with st.expander("🎓 Congratulations!"):
        st.markdown("""
        **You've completed the beginner track!** 🎉
    
        You now understand:
        - ✅ Basic chat with LLMs
        - ✅ Prompt templates
        - ✅ Output parsing
        - ✅ Chains
        - ✅ Conversation memory
        - ✅ Embeddings
        - ✅ Document loading
        - ✅ Text splitting
        - ✅ Vector stores
        - ✅ RAG systems
    
        **Next Steps:**
        - Explore **Advanced** examples for more complex patterns
        - Build your own custom RAG application
        - Try different models and parameters
        - Experiment with larger documents
    
        **Advanced Topics to Explore:**
        - Custom chains
        - Advanced memory patterns
        - Multi-query RAG
        - Streaming responses
        - Function calling
        - Agents with tools
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
