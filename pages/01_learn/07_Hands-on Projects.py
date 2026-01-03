import streamlit as st
from lib.helper_streamlit.show_source import show_source

st.set_page_config(
    page_title="Praxis-Projekte",
    page_icon="🚀"
)

st.header('🚀 Hands-on Projekte')

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.write('''
    Der beste Weg, LangChain und Ollama zu meistern, ist **Learning by Doing**! 
    In diesem Abschnitt findest du praktische Projekte, die reale Anwendungsfälle 
    demonstrieren.

    **Alle Projekte nutzen:**
    - ✅ Lokale LLMs via Ollama
    - ✅ Keine API-Kosten
    - ✅ Volle Datenkontrolle
    - ✅ Production-ready Patterns
    ''')

    st.info("💡 Jedes Projekt baut auf den vorherigen Modulen auf!", icon="💡")

    st.divider()

    # -------------------------------------------------------------------
    # Project Categories
    # -------------------------------------------------------------------
    st.subheader('📚 Projekt-Kategorien')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔰 Einsteiger-Projekte
        - 💬 **CLI Chatbot**: Einfacher konversationeller Bot
        - 📝 **Text Summarizer**: Lange Texte zusammenfassen
        - 🏷️ **Text Classifier**: Texte kategorisieren
        - 🔄 **Translator**: Übersetzungs-Tool
        """)

    with col2:
        st.markdown("""
        ### 🚀 Fortgeschritten
        - 📚 **RAG System**: Q&A über eigene Dokumente
        - 🔍 **Semantic Search**: Intelligente Dokumentensuche
        - 🤖 **Agent System**: LLM mit Tools
        - 💼 **Business Analytics**: Datenanalyse mit LLM
        """)

    st.divider()

    # -------------------------------------------------------------------
    # Featured Projects
    # -------------------------------------------------------------------
    st.subheader('⭐ Featured Projects')

    # Project 1: RAG System
    with st.expander("📚 Projekt #1: RAG System (Retrieval-Augmented Generation)", expanded=True):
        st.markdown("""
        ### 🎯 Ziel
        Baue ein System, das Fragen über deine eigenen Dokumente beantworten kann.
    
        **Was du lernst:**
        - Document Loader für PDFs, CSVs, etc.
        - Text Splitting-Strategien
        - Embeddings mit Ollama
        - Vector Store (Chroma)
        - Retrieval-Chain aufbauen
    
        **Tech Stack:**
        - `langchain_ollama.OllamaEmbeddings` - Lokale Embeddings
        - `langchain_community.vectorstores.Chroma` - Vector Store
        - `langchain_text_splitters` - Intelligentes Splitting
        - `langchain_ollama.ChatOllama` - Lokales LLM
    
        **Workflow:**
        ```python
        1. Dokumente laden (PDF/CSV/TXT)
        2. In Chunks aufteilen (RecursiveCharacterTextSplitter)
        3. Embeddings generieren (OllamaEmbeddings)
        4. In Vector Store speichern (Chroma)
        5. Retrieval-Chain aufbauen
        6. Fragen beantworten
        ```
    
        **Demo-Code:**
        ```python
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
    
        # 1. Dokumente laden
        loader = PyPDFLoader("dokument.pdf")
        docs = loader.load()
    
        # 2. Splitten
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
    
        # 3. Embeddings + Vector Store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
        # 4. RAG Chain
        template = \"\"\"Beantworte die Frage basierend auf folgendem Kontext:
    
        {context}
    
        Frage: {question}
        \"\"\"
    
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOllama(model="llama3.2")
    
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
        # 5. Fragen stellen
        answer = rag_chain.invoke("Was ist das Hauptthema des Dokuments?")
        print(answer)
        ```
        """)
    
        st.success("🎓 Schwierigkeit: Mittel | ⏱️ Zeitaufwand: 2-3 Stunden")

    # Project 2: Semantic Search
    with st.expander("🔍 Projekt #2: Semantische Suche über Code-Base"):
        st.markdown("""
        ### 🎯 Ziel
        Durchsuche Python-Code semantisch (nicht nur Keyword-Suche).
    
        **Anwendungsfall:**
        - Finde ähnliche Code-Snippets
        - "Wo wird Feature X implementiert?"
        - Code-Dokumentation intelligent durchsuchen
    
        **Besonderheiten:**
        - Language-aware Text Splitting für Python
        - Metadata-Filtering (Dateiname, Funktionsname)
        - Hybrid Search (Semantic + Keyword)
    
        **Quick Start:**
        ```python
        from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
    
        # Code-spezifischer Splitter
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=100
        )
    
        # Alle .py Dateien laden
        code_docs = load_python_files("./src")
        chunks = python_splitter.split_documents(code_docs)
    
        # Vector Store mit Metadata
        vectorstore = Chroma.from_documents(
            chunks,
            OllamaEmbeddings(model="nomic-embed-text")
        )
    
        # Semantische Suche
        results = vectorstore.similarity_search(
            "Funktion für Datenbankverbindung",
            k=5
        )
        ```
        """)
    
        st.success("🎓 Schwierigkeit: Mittel | ⏱️ Zeitaufwand: 2-4 Stunden")

    # Project 3: Conversational Agent
    with st.expander("🤖 Projekt #3: Konversationeller Agent mit Tools"):
        st.markdown("""
        ### 🎯 Ziel
        LLM, das Tools nutzen kann (Rechner, Web-Search, Datenbank-Queries).
    
        **Agent-Workflow:**
        1. User stellt Frage
        2. Agent entscheidet: Welches Tool brauche ich?
        3. Tool wird ausgeführt
        4. Ergebnis an LLM zurück
        5. LLM formuliert finale Antwort
    
        **Beispiel-Tools:**
        - 🧮 Rechner (für mathematische Operationen)
        - 🌐 Wikipedia-Suche
        - 📊 SQL-Datenbank-Query
        - 📧 E-Mail senden
    
        **Code-Skeleton:**
        ```python
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.tools import Tool
        from langchain_ollama import ChatOllama
    
        # Tools definieren
        def calculator(expression: str) -> str:
            try:
                return str(eval(expression))
            except:
                return "Fehler bei Berechnung"
    
        tools = [
            Tool(
                name="Rechner",
                func=calculator,
                description="Für mathematische Berechnungen"
            )
        ]
    
        # Agent erstellen
        llm = ChatOllama(model="llama3.2", temperature=0)
        agent = create_react_agent(llm, tools, prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
        # Nutzen
        result = agent_executor.invoke({"input": "Was ist 234 * 567?"})
        ```
        """)
    
        st.warning("🎓 Schwierigkeit: Fortgeschritten | ⏱️ Zeitaufwand: 4-6 Stunden")

    # Project 4: Document Analysis
    with st.expander("📊 Projekt #4: Multi-Document Analyse & Vergleich"):
        st.markdown("""
        ### 🎯 Ziel
        Mehrere Dokumente parallel analysieren und vergleichen.
    
        **Use Cases:**
        - Vertragsvergleich
        - Literatur-Review
        - Policy-Analyse
        - Wettbewerber-Analyse
    
        **Techniken:**
        - Map-Reduce Pattern für große Dokumente
        - Multi-Query Retrieval
        - Cross-Document Citations
        - Strukturierte Output-Parser (Pydantic)
    
        **Ansatz:**
        ```python
        # 1. Mehrere Dokumente laden
        docs_a = loader_a.load()
        docs_b = loader_b.load()
    
        # 2. Separate Vector Stores
        vectorstore_a = Chroma.from_documents(docs_a, embeddings, collection_name="doc_a")
        vectorstore_b = Chroma.from_documents(docs_b, embeddings, collection_name="doc_b")
    
        # 3. Vergleichs-Chain
        comparison_prompt = \"\"\"
        Vergleiche folgende Textstellen:
    
        Dokument A: {context_a}
        Dokument B: {context_b}
    
        Frage: {question}
        \"\"\"
    
        # 4. Parallel retrieval
        retriever_a = vectorstore_a.as_retriever()
        retriever_b = vectorstore_b.as_retriever()
        ```
        """)
    
        st.success("🎓 Schwierigkeit: Fortgeschritten | ⏱️ Zeitaufwand: 4-6 Stunden")

    st.divider()

    # -------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------
    st.subheader('📦 Projekt-Ressourcen')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🛠️ Hilfreiche Tools:**
        - [LangSmith](https://smith.langchain.com/) - Debugging & Tracing
        - [Chroma](https://www.trychroma.com/) - Vector Store
        - [Streamlit](https://streamlit.io/) - Web UI
        - [Ollama Library](https://ollama.com/library) - Modelle
        """)

    with col2:
        st.markdown("""
        **📚 Lernressourcen:**
        - [LangChain Docs](https://python.langchain.com/)
        - [Ollama Docs](https://ollama.com/)
        - [LCEL Guide](https://python.langchain.com/docs/expression_language/)
        - [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
        """)

    st.divider()

    with st.expander("💡 Tipps für erfolgreiche Projekte"):
        st.markdown("""
        **Projekt-Planung:**
        1. 🎯 **Klein starten**: Einfache Version zuerst
        2. 🧪 **Experimentieren**: Verschiedene Modelle/Chunk-Sizes testen
        3. 📊 **Evaluieren**: Qualität der Ergebnisse messen
        4. 🔄 **Iterieren**: Schrittweise verbessern
    
        **Debugging:**
        - 🔍 `verbose=True` bei Chains aktivieren
        - 📝 Zwischenergebnisse ausgeben
        - 🧩 Komponenten einzeln testen
        - 📊 LangSmith für Tracing nutzen
    
        **Performance:**
        - ⚡ Kleinere Modelle für schnelle Iteration
        - 💾 Vector Store cachen
        - 🎯 Retrieval k-Parameter optimieren
        - 📏 Chunk-Size balancieren
        """)

    st.caption("Workshop-Material: Praktische Projekte mit LangChain & Ollama")

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
