import streamlit as st
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from lib.helper_streamlit import select_model

# Standardmodell für diesen Workshop
DEFAULT_MODEL = "llama3.2"

st.set_page_config(
    page_title="LLMs mit Ollama",
    page_icon="🤖"
)

st.header('🤖 LLMs: Lokale Sprachmodelle mit Ollama')

st.write('''
Large Language Models (LLMs) sind neuronale Netzwerke, die auf riesigen Textmengen
trainiert wurden. Sie können menschenähnlichen Text verstehen und generieren.
Mit **Ollama** können wir diese Modelle komplett lokal auf unserem Rechner ausführen -
ohne Cloud, ohne Kosten, mit voller Datenkontrolle.

LangChain bietet uns eine einheitliche Schnittstelle zu verschiedenen LLMs. So können
wir flexibel zwischen Modellen wechseln, ohne unseren Code grundlegend ändern zu müssen.
''')

st.subheader('💬 Basis-LLM Integration')

st.write('''
Der einfachste Einstieg: Wir instanziieren ein Ollama-Modell und stellen eine Frage.
Die `Ollama`-Klasse aus LangChain kommuniziert direkt mit deiner lokalen Ollama-Instanz.
''')

st.code(f'''
from langchain_community.llms import Ollama

llm = Ollama(model="{DEFAULT_MODEL}")
response = llm.invoke("Erkläre LangChain in einem Satz")
print(response)
''', language='python')

st.info("💡 Stelle sicher, dass Ollama läuft und das Modell heruntergeladen ist: `ollama pull llama3.2`", icon="💡")

st.markdown("### 🎯 Interaktive Demo")

model_choice = select_model(key="llm_model_demo", location="sidebar", label="Wähle ein Modell")

with st.form("ollama_test"):
    user_prompt = st.text_area(
        "Deine Frage an das LLM",
        placeholder="Beispiel: Erkläre den Unterschied zwischen LLM und ChatModel",
        height=80
    )
    run_btn = st.form_submit_button("🚀 Ausführen", use_container_width=True)

    if run_btn and user_prompt.strip():
        with st.spinner(f"Anfrage an {model_choice}..."):
            llm = Ollama(model=model_choice)
            answer = llm.invoke(user_prompt)
        
        st.success("Antwort vom LLM:")
        st.markdown(answer)

st.divider()

st.write('''
**Warum LangChain verwenden?** Auf den ersten Blick scheint es trivial, eine 
Ollama-API aufzurufen. Aber LangChain bietet uns:

- 🔄 **Modell-Abstraktion**: Einfacher Wechsel zwischen verschiedenen LLMs
- ⛓️ **Chain-Komposition**: Verkettung mehrerer LLM-Aufrufe
- 🎯 **Prompt-Management**: Wiederverwendbare Templates
- 📊 **Strukturierte Outputs**: Parser für konsistente Datenformate
''')

st.subheader('💭 LLM vs. ChatModel')

st.write('''
In LangChain gibt es zwei Haupttypen von Modell-Wrappern:

**1. LLM (Text Completion)**
- Text rein → Text raus
- Einfache String-Verarbeitung
- Beispiel: `llm.invoke("Fortsetzung des Satzes...")`

**2. ChatModel (Konversations-basiert)**
- Arbeitet mit Nachrichten (`HumanMessage`, `AIMessage`, `SystemMessage`)
- Unterstützt Gesprächskontext
- Besser für interaktive Anwendungen
''')

st.code('''
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOllama(model="llama3.2")

messages = [
    SystemMessage(content="Du bist ein hilfreicher Assistent."),
    HumanMessage(content="Was ist LangChain?")
]

response = chat.invoke(messages)
print(response.content)
''', language='python')

st.markdown("### 🧪 ChatModel Demo")

with st.form("chat_demo"):
    system_msg = st.text_input(
        "System-Nachricht (Rolle/Kontext)",
        value="Du bist ein Python-Experte, der präzise und knappe Antworten gibt."
    )
    user_msg = st.text_area(
        "Deine Nachricht",
        placeholder="Frage das LLM etwas...",
        height=100
    )
    
    if st.form_submit_button("💬 Chat", use_container_width=True):
        if user_msg.strip():
            from langchain_core.messages import HumanMessage, SystemMessage
            
            chat_model = ChatOllama(model=DEFAULT_MODEL, temperature=0.7)
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg)
            ]
            
            with st.spinner("Denke nach..."):
                result = chat_model.invoke(messages)
            
            st.info("🤖 Antwort:")
            st.markdown(result.content)

st.divider()

with st.expander("📚 Weiterführende Ressourcen"):
    st.markdown('''
    - [Ollama Dokumentation](https://ollama.com/)
    - [LangChain LLM Guide](https://python.langchain.com/docs/modules/model_io/llms/)
    - [Verfügbare Ollama-Modelle](https://ollama.com/library)
    ''')

st.caption("Workshop-Material: LangChain & Ollama Integration")
