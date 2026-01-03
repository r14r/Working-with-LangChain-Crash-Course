import os
import tempfile
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, WebBaseLoader
from lib.helper_streamlit.show_source import show_source

st.set_page_config(
    page_title="Document Loader & Text Splitter",
    page_icon="📄"
)

st.header('📄 Dokumente Laden & Intelligent Splitten')

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.write('''
    **Das Problem:** LLMs haben begrenzte Kontextfenster und können nicht beliebig lange
    Dokumente verarbeiten. **Die Lösung:** Dokumente intelligent laden und in semantisch
    sinnvolle Chunks aufteilen.

    **In diesem Modul lernst du:**
    - 📂 Verschiedene Dokumenttypen laden (PDF, CSV, TXT, Web)
    - ✂️ Text-Splitting-Strategien
    - 🎯 Chunk-Size und Overlap optimieren
    - 🔍 Dokumente für RAG vorbereiten
    ''')

    st.info("💡 Gutes Splitting ist entscheidend für die Qualität von RAG-Systemen!", icon="💡")

    st.divider()

    # -------------------------------------------------------------------
    # Document Loaders
    # -------------------------------------------------------------------
    st.subheader('📥 Document Loaders')

    st.write('''
    LangChain bietet **Document Loader** für die gängigsten Formate. Alle Loader
    konvertieren Inhalte in ein einheitliches `Document`-Objekt mit:
    - `page_content`: Der eigentliche Text
    - `metadata`: Zusätzliche Infos (Quelle, Seite, etc.)
    ''')

    st.code('''
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader("dokument.pdf")
    documents = loader.load()

    # Jedes Document hat:
    print(documents[0].page_content)  # Text
    print(documents[0].metadata)       # {'source': '...', 'page': 1}
    ''', language='python')

    st.markdown("### 📝 TextLoader")

    txt_file = st.file_uploader("📄 TXT-Datei hochladen", type=["txt"], key="txt")

    if txt_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(txt_file.read())
            tmp_path = tmp.name
    
        loader = TextLoader(tmp_path)
        docs = loader.load()
    
        st.success(f"✅ {len(docs)} Dokument(e) geladen")
    
        with st.expander("📄 Dokument ansehen"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Dokument {i+1}**")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.json(doc.metadata)
    
        os.remove(tmp_path)

    st.markdown("### 📊 CSVLoader")

    csv_file = st.file_uploader("📊 CSV-Datei hochladen", type=["csv"], key="csv")

    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.read())
            tmp_path = tmp.name
    
        loader = CSVLoader(tmp_path)
        docs = loader.load()
    
        st.success(f"✅ {len(docs)} Zeile(n) als Dokumente geladen")
    
        with st.expander("📊 Erste 3 Zeilen"):
            for doc in docs[:3]:
                st.text(doc.page_content)
                st.caption(f"Metadata: {doc.metadata}")
    
        os.remove(tmp_path)

    st.markdown("### 📕 PyPDFLoader")

    pdf_file = st.file_uploader("📕 PDF-Datei hochladen", type=["pdf"], key="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
    
        with st.spinner("PDF wird verarbeitet..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
    
        st.success(f"✅ {len(docs)} Seite(n) geladen")
    
        page_select = st.selectbox("Seite auswählen", range(len(docs)))
    
        with st.expander(f"📄 Seite {page_select + 1}"):
            st.text(docs[page_select].page_content)
            st.json(docs[page_select].metadata)
    
        os.remove(tmp_path)

    st.markdown("### 🌐 WebBaseLoader")

    with st.form("web_loader"):
        url_input = st.text_input(
            "URL eingeben",
            placeholder="https://example.com/artikel"
        )
        load_btn = st.form_submit_button("🌐 Webseite laden")
    
        if load_btn and url_input:
            with st.spinner("Lade Webseite..."):
                try:
                    loader = WebBaseLoader(url_input)
                    docs = loader.load()
                
                    st.success(f"✅ Webseite geladen ({len(docs[0].page_content)} Zeichen)")
                
                    with st.expander("📄 Inhalt ansehen"):
                        st.text(docs[0].page_content[:1000] + "...")
                except Exception as e:
                    st.error(f"❌ Fehler beim Laden: {e}")

    st.divider()

    # -------------------------------------------------------------------
    # Text Splitters
    # -------------------------------------------------------------------
    st.subheader('✂️ Text Splitters')

    st.write('''
    **Warum splitten?**
    - 📏 LLMs haben begrenzte Kontextfenster (z.B. 4k, 8k, 32k Tokens)
    - 🎯 Kleinere Chunks → präzisere Retrieval-Ergebnisse
    - 💰 Weniger Tokens → geringere Kosten (bei API-basierten LLMs)
    - 🧠 Bessere semantische Granularität

    **Strategien:**
    - **CharacterTextSplitter**: Einfaches Splitting an Zeichen
    - **RecursiveCharacterTextSplitter**: Intelligent (Absätze → Sätze → Wörter)
    - **Language-aware**: Code-spezifisches Splitting
    ''')

    st.markdown("### 🔪 CharacterTextSplitter")

    st.write("Splittet Text an einem festen Trennzeichen (z.B. Leerzeichen, Newline).")

    with st.form("char_splitter"):
        text1 = st.text_area(
            "Text eingeben",
            height=150,
            placeholder="Füge einen längeren Text ein zum Splitten..."
        )
    
        col1, col2 = st.columns(2)
        with col1:
            chunk_size1 = st.slider("Chunk-Größe", 20, 200, 80, key="cs1")
        with col2:
            overlap1 = st.slider("Overlap", 0, 50, 10, key="o1")
    
        split_btn1 = st.form_submit_button("✂️ Splitten")
    
        if split_btn1 and text1.strip():
            splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=chunk_size1,
                chunk_overlap=overlap1,
                length_function=len
            )
        
            chunks = splitter.split_text(text1)
        
            st.success(f"✅ {len(chunks)} Chunks erstellt")
        
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk)} Zeichen)"):
                    st.text(chunk)

    st.markdown("### 🧠 RecursiveCharacterTextSplitter (Empfohlen)")

    st.write('''
    Der **intelligente Splitter**: Versucht zuerst an Absätzen zu trennen, dann an
    Sätzen, dann an Wörtern. Erhält semantischen Zusammenhang besser.
    ''')

    st.code('''
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Max. Zeichen pro Chunk
        chunk_overlap=200,      # Überlappung für Kontext
        separators=["\\n\\n", "\\n", ". ", " ", ""]  # Hierarchie
    )

    docs = splitter.create_documents([text])
    ''', language='python')

    with st.form("recursive_splitter"):
        text2 = st.text_area(
            "Text mit Absätzen eingeben",
            height=200,
            placeholder="Absatz 1.\n\nAbsatz 2.\n\nAbsatz 3...",
            value="Künstliche Intelligenz revolutioniert viele Branchen.\n\nMachine Learning ist ein Teilbereich der KI.\n\nDeep Learning nutzt neuronale Netze mit vielen Schichten."
        )
    
        col1, col2 = st.columns(2)
        with col1:
            chunk_size2 = st.slider("Chunk-Größe", 50, 500, 100, key="cs2")
        with col2:
            overlap2 = st.slider("Overlap", 0, 100, 20, key="o2")
    
        split_btn2 = st.form_submit_button("✂️ Intelligent splitten")
    
        if split_btn2 and text2.strip():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size2,
                chunk_overlap=overlap2,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
            docs = splitter.create_documents([text2])
        
            st.success(f"✅ {len(docs)} Dokumente erstellt")
        
            for i, doc in enumerate(docs):
                with st.expander(f"📄 Dokument {i+1}"):
                    st.text(doc.page_content)
                    st.caption(f"Länge: {len(doc.page_content)} Zeichen")

    st.markdown("### 💻 Code-Splitting")

    st.write("Für Code gibt es spezialisierte Splitter, die Funktionen/Klassen respektieren.")

    st.code('''
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )

    code_docs = python_splitter.create_documents([python_code])
    ''', language='python')

    st.divider()

    with st.expander("🎯 Best Practices für Splitting"):
        st.markdown("""
        **Chunk-Size wählen:**
        - 📚 **Lange Dokumente** (Bücher): 1000-2000 Zeichen
        - 📄 **Artikel/Berichte**: 500-1000 Zeichen
        - 💬 **Chat/FAQ**: 200-500 Zeichen
        - 💻 **Code**: Funktions-/Klassen-basiert
    
        **Overlap einstellen:**
        - ✅ 10-20% der Chunk-Size ist ein guter Start
        - ✅ Mehr Overlap = mehr Kontext, aber auch Redundanz
        - ✅ Zu wenig = Informationsverlust an Grenzen
    
        **Embeddings berücksichtigen:**
        - ⚠️ Embedding-Modelle haben Max-Token-Limits
        - ⚠️ `nomic-embed-text`: ~8k Tokens
        - ⚠️ Chunk-Size sollte deutlich darunter liegen
        """)

    st.caption("Workshop-Material: Document Loading & Intelligent Text Splitting")

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
