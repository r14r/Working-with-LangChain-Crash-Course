import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Multi-Query RAG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Query RAG | Advanced",
    page_icon="🔀",
    layout="centered",
)

st.title("🔀 Multi-Query RAG")
st.caption("Improve retrieval by generating multiple search queries.")

st.markdown(
    """
Multi-Query RAG generates multiple variations of the user's question to 
improve retrieval accuracy. Instead of one query, we create several 
perspectives and combine results.

**Benefits:**
- Better coverage of relevant documents
- Handles ambiguous questions
- More robust retrieval
"""
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    
    chat_model = select_model(
        key="mq_chat_model",
        location="sidebar",
        label="Chat Model"
    )
    
    embed_model = select_model(
        key="mq_embed_model",
        location="sidebar",
        label="Embedding Model",
        default_models=["nomic-embed-text", "mxbai-embed-large"]
    )
    
    num_queries = st.slider("Number of queries to generate", 2, 5, 3)
    k_per_query = st.slider("Docs per query", 1, 3, 2)

# Initialize
if "mq_vectorstore" not in st.session_state:
    st.session_state.mq_vectorstore = None

# --------------------------------------------------------------------------------------
# Load Sample Data
# --------------------------------------------------------------------------------------
st.subheader("1) Load Knowledge Base")

if st.button("📚 Load Sample Documents"):
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Natural language processing enables computers to understand and generate human language.",
        "TensorFlow and PyTorch are popular frameworks for building deep learning models.",
        "Supervised learning requires labeled training data, while unsupervised learning works with unlabeled data.",
        "Computer vision allows machines to interpret visual information from images and videos.",
        "Reinforcement learning trains agents through trial and error using rewards and penalties.",
    ]
    
    try:
        with st.spinner("Creating vector store..."):
            documents = [Document(page_content=text) for text in sample_docs]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            splits = splitter.split_documents(documents)
            
            embeddings = OllamaEmbeddings(model=embed_model)
            st.session_state.mq_vectorstore = DocArrayInMemorySearch.from_documents(
                splits, embeddings
            )
        
        st.success(f"✅ Loaded {len(sample_docs)} documents")
    
    except Exception as exc:
        st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Multi-Query RAG
# --------------------------------------------------------------------------------------
if st.session_state.mq_vectorstore:
    st.subheader("2) Ask a Question")
    
    question = st.text_input(
        "Your question:",
        placeholder="What is machine learning?"
    )
    
    if st.button("🔍 Search with Multi-Query"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        
        try:
            llm = ChatOllama(model=chat_model, temperature=0.7)
            parser = StrOutputParser()
            
            # Generate multiple query variations
            with st.spinner("Generating query variations..."):
                query_prompt = ChatPromptTemplate.from_template(
                    """Generate {num} different ways to ask this question for better search results.
Original: {question}

Return only the questions, one per line, without numbers or bullets."""
                )
                
                query_chain = query_prompt | llm | parser
                variations_text = query_chain.invoke({
                    "num": num_queries,
                    "question": question
                })
                
                # Parse variations
                variations = [q.strip() for q in variations_text.split('\n') if q.strip()]
                variations = [question] + variations[:num_queries-1]  # Include original
            
            st.markdown("### 📝 Generated Queries:")
            for i, q in enumerate(variations, 1):
                st.write(f"{i}. {q}")
            
            # Retrieve documents for each query
            with st.spinner("Retrieving documents..."):
                all_docs = []
                seen_contents = set()
                
                for query in variations:
                    docs = st.session_state.mq_vectorstore.similarity_search(
                        query, k=k_per_query
                    )
                    
                    # Deduplicate
                    for doc in docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
            
            st.success(f"✅ Retrieved {len(all_docs)} unique documents")
            
            with st.expander("📚 Retrieved Documents"):
                for i, doc in enumerate(all_docs, 1):
                    st.markdown(f"**Doc {i}:**")
                    st.write(doc.page_content)
                    st.divider()
            
            # Generate answer
            with st.spinner("Generating answer..."):
                context = "\n\n".join(doc.page_content for doc in all_docs)
                
                answer_prompt = ChatPromptTemplate.from_template(
                    """Answer based on the context. If not found, say so.

Context:
{context}

Question: {question}

Answer:"""
                )
                
                answer_chain = answer_prompt | llm | parser
                answer = answer_chain.invoke({
                    "context": context,
                    "question": question
                })
            
            st.markdown("### 💬 Answer:")
            st.write(answer)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

else:
    st.info("👆 Load documents first")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 How Multi-Query RAG Works"):
    st.markdown("""
    **Process:**
    
    1. **Generate Variations**: Create multiple versions of the question
    2. **Parallel Retrieval**: Search with each variation
    3. **Deduplicate**: Remove duplicate documents
    4. **Generate Answer**: Use all relevant docs
    
    **Example:**
    - Original: "What is ML?"
    - Variation 1: "Explain machine learning"
    - Variation 2: "Define artificial intelligence learning systems"
    - Variation 3: "How do ML algorithms work?"
    
    Each variation may retrieve different relevant documents!
    
    **Code Pattern:**
    ```python
    # Generate query variations
    variations = generate_variations(original_query)
    
    # Retrieve for each
    all_docs = []
    for query in variations:
        docs = vectorstore.search(query)
        all_docs.extend(docs)
    
    # Deduplicate and answer
    unique_docs = deduplicate(all_docs)
    answer = llm.invoke(context=unique_docs, question=original_query)
    ```
    """)

with st.expander("💡 Benefits & Trade-offs"):
    st.markdown("""
    **Advantages:**
    - ✅ Better recall (finds more relevant docs)
    - ✅ Handles ambiguous questions
    - ✅ More comprehensive answers
    
    **Trade-offs:**
    - ⚠️ More LLM calls (slower, higher cost)
    - ⚠️ More retrieval operations
    - ⚠️ Need deduplication logic
    
    **When to Use:**
    - Critical questions need thorough search
    - Document collection is large
    - Questions can be interpreted multiple ways
    
    **Alternatives:**
    - Query expansion with synonyms
    - Hybrid search (vector + keyword)
    - Re-ranking retrieved documents
    """)
