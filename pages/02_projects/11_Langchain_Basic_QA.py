import os
import tempfile
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from lib.helper_streamlit import select_model


# --------------------------------------------------------------------------------------
# App: Local PDF Q&A (RAG) with Ollama
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Local PDF Q&A (RAG) | Ollama + LangChain",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Local PDF Q&A (RAG)")
st.caption("Ask questions about a PDF using local embeddings + local chat model (Ollama).")

st.markdown(
    """
This app builds a tiny Retrieval-Augmented Generation (RAG) pipeline:

1. **Load** a PDF and split it into pages  
2. **Embed** pages locally (Ollama embeddings)  
3. **Retrieve** the most relevant pages for your question  
4. **Answer** with an Ollama chat model using the retrieved context

Everything runs locally—no API keys.
"""
)

# --------------------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------------------
st.subheader("1) Model configuration")

with st.sidebar:
    st.header("Model Configuration")
    
    chat_model = select_model(
        key="qa_chat_model",
        location="sidebar",
        label="Chat model (Ollama)"
    )
    
    embed_model = select_model(
        key="qa_embed_model",
        location="sidebar",
        label="Embedding model (Ollama)",
        default_models=["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
    )

# --------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------
st.subheader("2) Upload PDF")
pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# --------------------------------------------------------------------------------------
# Prompt used for answering
# --------------------------------------------------------------------------------------
qa_prompt = ChatPromptTemplate.from_template(
    """You are a precise assistant. Answer the user's question using ONLY the context below.
If the context does not contain the answer, say that you cannot find it in the provided document.

Context:
{context}

Question:
{question}

Answer:"""
)

parser = StrOutputParser()

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(tmp_pdf_path: str, embedding_model: str):
    """Load PDF -> pages -> embeddings -> in-memory vector store (cached per file path)."""
    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()  # already split by pages

    embeddings = OllamaEmbeddings(model=embedding_model)
    return DocArrayInMemorySearch.from_documents(pages, embeddings)

def answer_question(db, question: str, k: int, llm_model: str) -> str:
    """Retrieve top-k docs and answer with local LLM."""
    docs = db.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )

    llm = ChatOllama(model=llm_model, temperature=0)
    chain = qa_prompt | llm | parser
    return chain.invoke({"context": context, "question": question})

# --------------------------------------------------------------------------------------
# Query UI
# --------------------------------------------------------------------------------------
st.subheader("3) Ask your question")

with st.form("basic_qa"):
    query = st.text_input("Question", placeholder="Summarize this document in 5 bullet points.")
    k = st.slider("Retrieved pages", min_value=1, max_value=8, value=4, help="How many pages to retrieve as context.")
    run = st.form_submit_button("🚀 Run")

if run:
    if not pdf_file:
        st.warning("Please upload a PDF first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    tmp_path = None
    try:
        with st.spinner("Indexing PDF (pages → embeddings)…"):
            # Save to a temp file; cache key is path, so keep it stable for this run
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name

            db = build_vectorstore(tmp_path, embed_model)

        with st.spinner("Retrieving + answering…"):
            response = answer_question(db, query, k=k, llm_model=chat_model)

        st.markdown("### Answer")
        st.write(response)

    except Exception as exc:
        st.error(f"Failed to process the request: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

with st.expander("Ideas to extend"):
    st.write(
        """
- Add a proper text splitter (chunking) instead of page-based chunks
- Show retrieved pages/snippets with citations
- Persist the vector index to disk (instead of in-memory)
- Add multi-PDF support and a document selector
"""
    )

st.subheader("Practical limits")
st.write(
    """
This is intentionally lightweight: page-level chunks in an in-memory vector store.
It works well for short PDFs, contracts, small reports, and email exports.

For large PDFs (books, manuals) you’ll typically add:
- chunking + overlap
- persistent vector DB (Chroma/Qdrant/FAISS)
- metadata filters and citations
"""
)
