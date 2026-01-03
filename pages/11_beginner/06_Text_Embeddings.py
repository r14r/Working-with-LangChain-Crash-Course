import streamlit as st
from langchain_ollama import OllamaEmbeddings
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source
import numpy as np

# --------------------------------------------------------------------------------------
# App: Text Embeddings
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Embeddings | Beginner",
    page_icon="🔢",
    layout="centered",
)

st.title("🔢 Text Embeddings")
st.caption("Learn how to convert text into numerical vectors for AI applications.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    Embeddings convert text into numerical vectors (arrays of numbers) that 
    represent the meaning of the text. Similar texts have similar vectors.

    **What you'll learn:**
    1. Generate embeddings with Ollama  
    2. Understand embedding dimensions  
    3. Compare similarity between texts
    """
    )

    # --------------------------------------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        embedding_model = select_model(
            key="embedding_model",
            location="sidebar",
            label="Embedding Model",
            default_models=["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
        )
    
        st.info("Embedding models are specialized for creating text vectors. "
                "They're different from chat models.")

    # --------------------------------------------------------------------------------------
    # Embedding Examples
    # --------------------------------------------------------------------------------------
    st.subheader("1) Generate Embeddings")

    example_choice = st.radio(
        "Choose an example:",
        ["Single Text", "Compare Similarity", "Multiple Texts"]
    )

    if example_choice == "Single Text":
        st.info("Generate an embedding vector for a single piece of text.")
    
        text = st.text_area(
            "Enter text:",
            placeholder="The quick brown fox jumps over the lazy dog.",
            height=100
        )
    
        if st.button("🚀 Generate Embedding"):
            if not text.strip():
                st.warning("Please enter some text.")
                st.stop()
        
            try:
                with st.spinner("Generating embedding..."):
                    embeddings = OllamaEmbeddings(model=embedding_model)
                    vector = embeddings.embed_query(text)
            
                st.success(f"✅ Generated embedding with {len(vector)} dimensions")
            
                # Show first few values
                st.markdown("### Embedding Vector (first 10 values):")
                st.code(str(vector[:10]))
            
                with st.expander("📊 View Full Vector"):
                    st.write(vector)
            
                with st.expander("📈 Vector Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dimensions", len(vector))
                    with col2:
                        st.metric("Mean", f"{np.mean(vector):.6f}")
                    with col3:
                        st.metric("Std Dev", f"{np.std(vector):.6f}")
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    elif example_choice == "Compare Similarity":
        st.info("Compare how similar two texts are using their embeddings.")
    
        col1, col2 = st.columns(2)
        with col1:
            text1 = st.text_area(
                "Text 1:",
                placeholder="I love programming in Python.",
                height=100
            )
        with col2:
            text2 = st.text_area(
                "Text 2:",
                placeholder="Python is my favorite programming language.",
                height=100
            )
    
        if st.button("🚀 Compare Texts"):
            if not text1.strip() or not text2.strip():
                st.warning("Please enter both texts.")
                st.stop()
        
            try:
                with st.spinner("Generating embeddings..."):
                    embeddings = OllamaEmbeddings(model=embedding_model)
                
                    # Generate embeddings for both texts
                    vector1 = embeddings.embed_query(text1)
                    vector2 = embeddings.embed_query(text2)
                
                    # Calculate cosine similarity
                    v1 = np.array(vector1)
                    v2 = np.array(vector2)
                
                    dot_product = np.dot(v1, v2)
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                
                    similarity = dot_product / (norm_v1 * norm_v2)
            
                st.markdown("### Similarity Score:")
                st.metric(
                    "Cosine Similarity",
                    f"{similarity:.4f}",
                    help="1.0 = identical, 0.0 = unrelated, -1.0 = opposite"
                )
            
                # Visual representation
                similarity_percent = (similarity + 1) / 2 * 100  # Convert to 0-100%
                st.progress(similarity_percent / 100)
            
                if similarity > 0.8:
                    st.success("🎯 Very similar texts!")
                elif similarity > 0.5:
                    st.info("📝 Somewhat similar texts")
                else:
                    st.warning("📊 Not very similar")
            
                with st.expander("🔍 Understanding Similarity"):
                    st.markdown("""
                    **Cosine Similarity Scale:**
                    - **0.9 - 1.0**: Nearly identical meaning
                    - **0.7 - 0.9**: Very similar
                    - **0.5 - 0.7**: Somewhat similar
                    - **0.3 - 0.5**: Loosely related
                    - **< 0.3**: Different topics
                
                    Try comparing:
                    - Similar sentences with different wording
                    - Sentences about different topics
                    - Sentences in different languages
                    """)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    else:  # Multiple Texts
        st.info("Generate embeddings for multiple texts at once.")
    
        num_texts = st.slider("Number of texts:", 2, 5, 3)
    
        texts = []
        for i in range(num_texts):
            text = st.text_input(
                f"Text {i+1}:",
                placeholder=f"Enter text {i+1}...",
                key=f"text_{i}"
            )
            texts.append(text)
    
        if st.button("🚀 Generate All Embeddings"):
            if any(not t.strip() for t in texts):
                st.warning("Please enter all texts.")
                st.stop()
        
            try:
                with st.spinner("Generating embeddings..."):
                    embeddings = OllamaEmbeddings(model=embedding_model)
                
                    # Generate embeddings for all texts
                    vectors = embeddings.embed_documents(texts)
            
                st.success(f"✅ Generated {len(vectors)} embeddings")
            
                # Show summary
                for i, (text, vector) in enumerate(zip(texts, vectors)):
                    with st.expander(f"Text {i+1}: {text[:50]}..."):
                        st.write(f"**Dimensions:** {len(vector)}")
                        st.write(f"**First 10 values:** {vector[:10]}")
            
                # Similarity matrix
                st.markdown("### Similarity Matrix:")
            
                n = len(vectors)
                similarity_matrix = np.zeros((n, n))
            
                for i in range(n):
                    for j in range(n):
                        v1 = np.array(vectors[i])
                        v2 = np.array(vectors[j])
                    
                        dot_product = np.dot(v1, v2)
                        norm_v1 = np.linalg.norm(v1)
                        norm_v2 = np.linalg.norm(v2)
                    
                        similarity_matrix[i][j] = dot_product / (norm_v1 * norm_v2)
            
                st.dataframe(
                    similarity_matrix,
                    column_config={i: f"Text {i+1}" for i in range(n)}
                )
            
                st.info("Each cell shows how similar two texts are (1.0 = identical)")
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 What you learned"):
        st.markdown("""
        **Key Concepts:**
    
        - **Embeddings**: Numerical representations of text
        - **Vector**: An array of numbers (e.g., 768 or 384 dimensions)
        - **Cosine Similarity**: Measure of how similar two vectors are
    
        **How It Works:**
        ```python
        from langchain_ollama import OllamaEmbeddings
    
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
        # Single text
        vector = embeddings.embed_query("Hello world")
    
        # Multiple texts
        vectors = embeddings.embed_documents([
            "Text 1",
            "Text 2",
            "Text 3"
        ])
        ```
    
        **Why Use Embeddings:**
        - **Semantic Search**: Find similar documents
        - **Clustering**: Group similar texts
        - **Classification**: Categorize text
        - **Recommendation**: Suggest similar content
    
        **Common Embedding Models:**
        - `nomic-embed-text`: Good general-purpose model
        - `mxbai-embed-large`: High-quality embeddings
        - `bge-m3`: Multilingual support
        """)

    with st.expander("💡 Next Steps"):
        st.markdown("""
        Now that you understand embeddings:
        - **07_Document_Loader.py**: Load documents to embed
        - **08_Text_Splitter.py**: Split documents into chunks
        - **09_Vector_Store.py**: Store embeddings for search
        - **10_Simple_RAG.py**: Build a complete RAG system
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
