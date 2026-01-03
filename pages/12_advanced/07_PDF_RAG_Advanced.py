import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Advanced PDF RAG | Advanced", page_icon="📑", layout="wide")

st.title("📑 Advanced PDF RAG")
st.caption("Enhanced PDF processing with metadata, citations, and filtering.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Advanced RAG with PDF-specific features:
    - Page-level metadata tracking
    - Citation with page numbers
    - Confidence scoring
    - Source attribution
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        chat_model = select_model(key="pdf_chat", location="sidebar")
        embed_model = select_model(
            key="pdf_embed",
            location="sidebar",
            label="Embedding Model",
            default_models=["nomic-embed-text", "mxbai-embed-large"]
        )
    
        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        k_docs = st.slider("Retrieved Chunks", 2, 10, 4)

    # Initialize
    if "pdf_vectorstore" not in st.session_state:
        st.session_state.pdf_vectorstore = None
        st.session_state.pdf_filename = None

    # Upload PDF
    st.subheader("1) Upload PDF")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file and st.button("📄 Process PDF"):
        try:
            with st.spinner("Loading PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
            
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
            
                st.info(f"Loaded {len(pages)} pages")
        
            with st.spinner("Chunking and embedding..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)
            
                embeddings = OllamaEmbeddings(model=embed_model)
                st.session_state.pdf_vectorstore = DocArrayInMemorySearch.from_documents(
                    chunks, embeddings
                )
                st.session_state.pdf_filename = pdf_file.name
        
            os.remove(tmp_path)
            st.success(f"✅ Indexed {len(chunks)} chunks from {pdf_file.name}")
    
        except Exception as exc:
            st.error(f"Error: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Query
    if st.session_state.pdf_vectorstore:
        st.subheader("2) Ask Questions")
    
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input("Question:", placeholder="What is the main topic?")
        with col2:
            include_citations = st.checkbox("Include citations", value=True)
    
        if st.button("🔍 Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
                st.stop()
        
            try:
                with st.spinner("Retrieving relevant sections..."):
                    docs_with_scores = st.session_state.pdf_vectorstore.similarity_search_with_score(
                        question, k=k_docs
                    )
            
                # Display sources
                with st.expander(f"📚 Sources ({len(docs_with_scores)} chunks)"):
                    for i, (doc, score) in enumerate(docs_with_scores, 1):
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Chunk {i}** (Page {page}, Score: {score:.4f})")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()
            
                with st.spinner("Generating answer..."):
                    # Prepare context with citations
                    context_parts = []
                    for i, (doc, score) in enumerate(docs_with_scores, 1):
                        page = doc.metadata.get('page', 'N/A')
                        context_parts.append(f"[Source {i}, Page {page}]\n{doc.page_content}")
                
                    context = "\n\n".join(context_parts)
                
                    # Create prompt
                    if include_citations:
                        prompt_template = """Answer the question using the provided sources. 
    Include [Source X, Page Y] citations in your answer.

    Sources:
    {context}

    Question: {question}

    Answer with citations:"""
                    else:
                        prompt_template = """Answer the question based on the context.

    Context:
    {context}

    Question: {question}

    Answer:"""
                
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    llm = ChatOllama(model=chat_model, temperature=0)
                    chain = prompt | llm | StrOutputParser()
                
                    answer = chain.invoke({"context": context, "question": question})
            
                st.markdown("### 💬 Answer:")
                st.write(answer)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:
        st.info("👆 Upload and process a PDF first")

    with st.expander("📚 Advanced RAG Features"):
        st.markdown("""
        **Enhancements:**
        - ✅ Page-level metadata tracking
        - ✅ Citation generation
        - ✅ Similarity scoring
        - ✅ Source attribution
    
        **Implementation:**
        ```python
        # Load with metadata
        loader = PyPDFLoader("doc.pdf")
        pages = loader.load()  # Each page has metadata
    
        # Retrieve with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query)
    
        # Build context with citations
        for doc, score in docs_with_scores:
            page = doc.metadata['page']
            context += f"[Page {page}] {doc.page_content}"
        ```
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
