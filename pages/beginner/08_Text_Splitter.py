import streamlit as st
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

# --------------------------------------------------------------------------------------
# App: Text Splitters
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Splitters | Beginner",
    page_icon="✂️",
    layout="centered",
)

st.title("✂️ Text Splitters")
st.caption("Learn how to split large documents into smaller, manageable chunks.")

st.markdown(
    """
Text splitters divide long documents into smaller chunks. This is essential for:
- Fitting text into model context windows
- Creating focused embeddings
- Improving search relevance

**What you'll learn:**
1. Different splitting strategies  
2. Chunk size and overlap  
3. How splitting affects your data
"""
)

# --------------------------------------------------------------------------------------
# Splitter Configuration
# --------------------------------------------------------------------------------------
st.subheader("1) Choose Splitter Type")

splitter_type = st.radio(
    "Select a text splitter:",
    [
        "Recursive Character Splitter (Recommended)",
        "Character Splitter",
        "Token Splitter"
    ]
)

st.info({
    "Recursive Character Splitter (Recommended)": "Tries to split on paragraphs, then sentences, then words. Best for most cases.",
    "Character Splitter": "Splits on a specific character (like newline). Simple but less smart.",
    "Token Splitter": "Splits by token count (what the model sees). Most precise for context limits."
}[splitter_type])

# --------------------------------------------------------------------------------------
# Splitter Parameters
# --------------------------------------------------------------------------------------
st.subheader("2) Configure Parameters")

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum size of each chunk"
    )

with col2:
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=50,
        step=10,
        help="Number of characters to overlap between chunks"
    )

st.markdown("""
**Why overlap?** Overlap ensures that information split between chunks isn't lost. 
For example, if a sentence is split, the overlap captures it in both chunks.
""")

# --------------------------------------------------------------------------------------
# Text Input
# --------------------------------------------------------------------------------------
st.subheader("3) Enter Text to Split")

text_option = st.radio(
    "Choose input method:",
    ["Use Sample Text", "Enter Custom Text"]
)

if text_option == "Use Sample Text":
    sample_text = """LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason about how to answer based on provided context.

The framework consists of several key components. First, there are LLM integrations that provide a standard interface for interacting with various language models. This abstraction makes it easy to swap between different providers.

Second, prompt templates help structure inputs to language models in a consistent way. Instead of manually formatting strings, templates provide reusable patterns with variable substitution.

Third, chains allow you to combine multiple components into a single workflow. A simple chain might consist of a prompt template, an LLM call, and an output parser.

Fourth, memory components enable applications to maintain context across multiple interactions. This is crucial for building conversational applications where the model needs to remember previous exchanges.

Fifth, agents extend chains with the ability to use tools and make decisions. An agent can decide which actions to take based on user input and can use external tools to accomplish tasks.

Finally, document loaders and text splitters help process various data sources. They can load documents from files, databases, or web pages, and split them into chunks suitable for embedding and retrieval.

Together, these components enable powerful applications like chatbots, question-answering systems, and document analysis tools."""
    
    text_input = st.text_area(
        "Sample text:",
        value=sample_text,
        height=300
    )
else:
    text_input = st.text_area(
        "Enter your text:",
        placeholder="Paste or type your text here...",
        height=300
    )

# --------------------------------------------------------------------------------------
# Split Text
# --------------------------------------------------------------------------------------
if st.button("✂️ Split Text"):
    if not text_input.strip():
        st.warning("Please enter some text to split.")
        st.stop()
    
    try:
        with st.spinner("Splitting text..."):
            # Create the appropriate splitter
            if splitter_type == "Recursive Character Splitter (Recommended)":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            elif splitter_type == "Character Splitter":
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separator="\n"
                )
            else:  # Token Splitter
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            # Split the text
            chunks = splitter.split_text(text_input)
        
        st.success(f"✅ Split into {len(chunks)} chunks")
        
        # Display chunks
        st.subheader("4) Resulting Chunks")
        
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1} ({len(chunk)} characters)"):
                st.text_area(
                    f"Chunk {i+1}",
                    value=chunk,
                    height=150,
                    key=f"chunk_{i}",
                    label_visibility="collapsed"
                )
                
                # Show overlap with previous chunk
                if i > 0 and chunk_overlap > 0:
                    # Find overlapping text
                    prev_chunk = chunks[i-1]
                    overlap_text = ""
                    
                    # Simple overlap detection
                    for j in range(min(len(prev_chunk), len(chunk))):
                        if prev_chunk[-j:] == chunk[:j]:
                            overlap_text = chunk[:j]
                    
                    if overlap_text and len(overlap_text) > 10:
                        st.caption(f"Overlaps with previous: '{overlap_text[:50]}...'")
        
        # Statistics
        st.subheader("5) Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", len(chunks))
        with col2:
            avg_size = sum(len(c) for c in chunks) / len(chunks)
            st.metric("Avg Chunk Size", f"{avg_size:.0f}")
        with col3:
            min_size = min(len(c) for c in chunks)
            st.metric("Min Size", min_size)
        with col4:
            max_size = max(len(c) for c in chunks)
            st.metric("Max Size", max_size)
        
        # Visualize chunk sizes
        st.bar_chart([len(c) for c in chunks])
        
    except Exception as exc:
        st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **Chunk Size**: Maximum characters/tokens per chunk
    - **Chunk Overlap**: Characters shared between chunks
    - **Separators**: Where to preferably split (paragraphs, sentences, etc.)
    
    **How It Works:**
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    
    # Split text
    chunks = splitter.split_text(long_text)
    
    # Or split documents
    docs = splitter.split_documents(documents)
    ```
    
    **Splitter Types:**
    
    | Splitter | Best For | How It Works |
    |----------|----------|--------------|
    | `RecursiveCharacterTextSplitter` | Most cases | Tries paragraph → sentence → word |
    | `CharacterTextSplitter` | Simple splitting | Splits on specific character |
    | `TokenTextSplitter` | Precise limits | Splits by model tokens |
    | `MarkdownTextSplitter` | Markdown | Respects MD structure |
    | `PythonCodeTextSplitter` | Code | Respects code syntax |
    
    **Why Split Text?**
    - Models have context limits (e.g., 4096 tokens)
    - Smaller chunks = better embeddings
    - Focused chunks = more relevant search
    - Easier to process and cache
    """)

with st.expander("💡 Choosing Parameters"):
    st.markdown("""
    **Chunk Size Guidelines:**
    - **Small (200-300)**: Very focused, good for precise search
    - **Medium (500-700)**: Balanced, works for most cases
    - **Large (1000-1500)**: More context, but less focused
    
    **Overlap Guidelines:**
    - **None (0)**: Simple, but might split important info
    - **Small (20-50)**: Good balance
    - **Large (100-200)**: Ensures continuity, but redundant
    
    **General Rule:** Overlap should be ~10% of chunk size
    
    **Consider Your Use Case:**
    - **Q&A**: Medium chunks with overlap
    - **Summarization**: Larger chunks
    - **Classification**: Smaller, focused chunks
    - **Search**: Medium chunks with overlap
    """)

with st.expander("🔧 Next Steps"):
    st.markdown("""
    Now that you can split text:
    - **09_Vector_Store.py**: Store split chunks as embeddings
    - **10_Simple_RAG.py**: Use chunks in a Q&A system
    
    **Advanced Topics:**
    - Semantic splitting (split by meaning, not just size)
    - Custom splitters for specific formats
    - Dynamic chunk sizing based on content
    - Preserving document structure in chunks
    """)
