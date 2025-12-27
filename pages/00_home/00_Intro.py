import streamlit as st

st.set_page_config(
    page_title="LangChain & Ollama Workshop",
    page_icon="🚀"
)

st.header('🚀 LangChain mit Ollama: Fortgeschrittener Workshop')

st.subheader('Lokale KI-Anwendungen entwickeln - Praktisch und hands-on!')

st.write('''
Willkommen zu diesem fortgeschrittenen Workshop über LangChain mit Ollama! Hier lernst du,
wie du leistungsstarke KI-Anwendungen entwickelst, die vollständig lokal auf deinem System 
laufen - ohne Cloud-Abhängigkeiten, ohne API-Kosten, mit voller Datenkontrolle.

LangChain ist ein mächtiges Framework für die Entwicklung von Large Language Model (LLM) 
Anwendungen. Es bietet Abstraktionen und Werkzeuge, die das Arbeiten mit verschiedenen 
LLMs, Vektorspeichern, Dokumenten und APIs erheblich vereinfachen.
''')

st.subheader('🎯 Was du lernen wirst')

col1, col2 = st.columns(2)

with col1:
    st.write('''
    **Kernkonzepte:**
    - 🤖 LLM-Integration (Ollama)
    - 📝 Prompt Engineering & Templates
    - ⛓️ Chain-Patterns (LCEL)
    - 🧠 Memory-Management
    ''')

with col2:
    st.write('''
    **Fortgeschrittene Themen:**
    - 📊 Embeddings & Vektorspeicher
    - 📄 Document Processing
    - 🔍 RAG-Implementierungen
    - 🛠️ Praktische Projekte
    ''')

st.info('💡 Dieser Workshop nutzt ausschließlich **Ollama** für lokale LLM-Inferenz - keine OpenAI API Keys erforderlich!', icon="💡")

st.subheader('🎓 Voraussetzungen')

st.write('''
Um das Beste aus diesem Workshop herauszuholen, solltest du mitbringen:

- **Python-Grundkenntnisse**: Du solltest mit Python-Syntax, Funktionen und Klassen vertraut sein
- **Ollama-Installation**: Stelle sicher, dass Ollama auf deinem System installiert ist
- **Grundlegendes Verständnis von LLMs**: Was sind Large Language Models und wie funktionieren sie?

**Neu in Python?** Hier sind hilfreiche Ressourcen:
- [Python Tutorial Deutsch](https://www.python-kurs.eu/)
- [Python lernen - freeCodeCamp](https://www.freecodecamp.org/news/learn-python-free-python-courses-for-beginners/)
''')

st.subheader('💻 Verwendete Technologien')

st.write('''
In diesem Workshop arbeiten wir mit modernen, Open-Source-Technologien:

- **[LangChain](https://www.langchain.com/)** - Das führende Framework für LLM-Anwendungen (Python)
- **[Ollama](https://ollama.com/)** - Lokale LLM-Inferenz (Llama, Mistral, Gemma, etc.)
- **[Streamlit](https://streamlit.io/)** - Schnelle Web-UI-Entwicklung für Data Science & KI
- **[LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)** - Moderne Chain-Komposition

Alle verwendeten Tools sind Open Source und können kostenlos genutzt werden.
''')

