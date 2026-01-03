import streamlit as st
from langchain_ollama import OllamaEmbeddings
import numpy as np
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(
    page_title="Embeddings & Vektorspeicher",
    page_icon="📊"
)

st.header('📊 Embeddings & Vector Stores')

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.write('''
    **Embeddings** sind die Brücke zwischen Text und Mathematik. Sie konvertieren Wörter,
    Sätze oder ganze Dokumente in numerische Vektoren, die semantische Bedeutung erfassen.

    **Warum sind Embeddings wichtig?**
    - 🔍 Semantische Suche (ähnliche Texte finden)
    - 🧠 Speicher für LLM-Anwendungen
    - 📚 Retrieval-Augmented Generation (RAG)
    - 🎯 Klassifikation und Clustering
    ''')

    st.info("💡 Mit Ollama können wir Embeddings komplett lokal generieren!", icon="💡")

    st.divider()

    # -------------------------------------------------------------------
    # Embedding Basics
    # -------------------------------------------------------------------
    st.subheader('🔢 Text zu Vektoren')

    st.write('''
    Ollama bietet mehrere Modelle für Embeddings. Wir nutzen `nomic-embed-text` - 
    ein spezialisiertes Modell für deutsche und englische Texte.
    ''')

    st.code('''
    from langchain_ollama import OllamaEmbeddings

    # Embedding-Modell initialisieren
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Text in Vektor umwandeln
    text = "LangChain vereinfacht die LLM-Entwicklung"
    vector = embeddings.embed_query(text)

    # vector ist nun ein Array mit ~768 Zahlen
    print(f"Vektor-Dimension: {len(vector)}")
    print(f"Erste 5 Werte: {vector[:5]}")
    ''', language='python')

    st.markdown("### 🧪 Interaktive Embedding-Demo")

    model_choice = st.selectbox(
        "Embedding-Modell",
        ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        help="Verschiedene Modelle haben unterschiedliche Dimensionen"
    )

    with st.form("embedding_demo"):
        text_input = st.text_area(
            "Text eingeben",
            placeholder="Beispiel: Machine Learning ist ein Teilgebiet der künstlichen Intelligenz",
            height=100
        )
    
        show_full = st.checkbox("Kompletten Vektor anzeigen (kann lang sein)")
    
        execute = st.form_submit_button("🔄 Embedding generieren", use_container_width=True)

        if execute and text_input.strip():
            with st.spinner(f"Generiere Embedding mit {model_choice}..."):
                embeddings_model = OllamaEmbeddings(model=model_choice)
                vector = embeddings_model.embed_query(text_input)

            st.success(f"✅ Embedding erfolgreich generiert!")
        
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dimension", len(vector))
            with col2:
                st.metric("Min-Wert", f"{min(vector):.4f}")
            with col3:
                st.metric("Max-Wert", f"{max(vector):.4f}")
        
            if show_full:
                st.json(vector)
            else:
                st.write("**Erste 10 Werte:**")
                st.code([round(v, 4) for v in vector[:10]])

    st.divider()

    # -------------------------------------------------------------------
    # Semantic Similarity
    # -------------------------------------------------------------------
    st.subheader('🎯 Semantische Ähnlichkeit')

    st.write('''
    Embeddings ermöglichen es uns, **semantische Ähnlichkeit** zu berechnen.
    Texte mit ähnlicher Bedeutung haben ähnliche Vektoren.
    ''')

    st.code('''
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(vec1, vec2):
        """Berechnet Cosinus-Ähnlichkeit (0-1, höher = ähnlicher)"""
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    # Beispiel
    text1 = "Der Hund spielt im Garten"
    text2 = "Ein Welpe tobt draußen"
    text3 = "Python ist eine Programmiersprache"

    vec1 = embeddings.embed_query(text1)
    vec2 = embeddings.embed_query(text2)
    vec3 = embeddings.embed_query(text3)

    print(f"Ähnlichkeit (1↔2): {cosine_similarity(vec1, vec2):.3f}")  # Hoch
    print(f"Ähnlichkeit (1↔3): {cosine_similarity(vec1, vec3):.3f}")  # Niedrig
    ''', language='python')

    st.markdown("### 🔍 Ähnlichkeits-Vergleich")

    with st.form("similarity_demo"):
        col1, col2 = st.columns(2)
    
        with col1:
            text1 = st.text_area("Text 1", placeholder="Erster Satz...", key="t1")
        with col2:
            text2 = st.text_area("Text 2", placeholder="Zweiter Satz...", key="t2")
    
        compare_btn = st.form_submit_button("🎯 Ähnlichkeit berechnen", use_container_width=True)
    
        if compare_btn and text1.strip() and text2.strip():
            with st.spinner("Berechne Embeddings und Ähnlichkeit..."):
                embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
            
                vec1 = embeddings_model.embed_query(text1)
                vec2 = embeddings_model.embed_query(text2)
            
                # Cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
            st.metric(
                "Semantische Ähnlichkeit",
                f"{similarity:.2%}",
                help="100% = identisch, 0% = völlig unterschiedlich"
            )
        
            if similarity > 0.8:
                st.success("🟢 Sehr ähnlich!")
            elif similarity > 0.5:
                st.info("🟡 Mäßig ähnlich")
            else:
                st.warning("🔴 Wenig Ähnlichkeit")

    st.divider()

    # -------------------------------------------------------------------
    # Vector Stores
    # -------------------------------------------------------------------
    st.subheader('🗄️ Vector Stores')

    st.write('''
    **Vector Stores** (Vektordatenbanken) speichern Embeddings und ermöglichen
    effiziente Ähnlichkeitssuchen über große Datenmengen.

    **Populäre Vector Stores:**
    - 💾 **Chroma** - Lokal, einfach, schnell
    - ☁️ **Pinecone** - Cloud, skalierbar
    - 🔷 **Qdrant** - Self-hosted, performant
    - 📦 **FAISS** - Facebook AI, In-Memory

    **Typischer Workflow:**
    1. Dokumente in Chunks aufteilen
    2. Embeddings für jeden Chunk generieren
    3. In Vector Store speichern
    4. Semantische Suche durchführen
    5. Relevante Chunks an LLM übergeben (RAG)
    ''')

    st.code('''
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 1. Dokumente laden und splitten
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_documents(documents)

    # 2. Embeddings erstellen
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 3. Vector Store aufbauen
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 4. Ähnlichkeitssuche
    query = "Wie funktioniert LangChain?"
    results = vectorstore.similarity_search(query, k=3)

    # 5. Ergebnisse nutzen
    for doc in results:
        print(doc.page_content)
    ''', language='python')

    st.info("🚀 In den Hands-on-Projekten bauen wir ein vollständiges RAG-System mit Vector Stores!", icon="🚀")

    st.divider()

    with st.expander("📚 Weiterführende Konzepte"):
        st.markdown("""
        **Advanced Topics:**
        - 🔄 **Hybrid Search**: Kombination aus semantischer und Keyword-Suche
        - 🎚️ **Reranking**: Ergebnisse nach Relevanz neu sortieren
        - 📊 **Metadata Filtering**: Suche mit zusätzlichen Filtern
        - ⚡ **Caching**: Embeddings wiederverwenden für bessere Performance
        - 🔀 **Multi-Vector Retrieval**: Mehrere Embedding-Modelle kombinieren
        """)

    st.caption("Workshop-Material: Embeddings & Vektorspeicher")
with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
