import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from lib.helper_streamlit import select_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Memory Management",
    page_icon="🧠"
)

st.header('🧠 Memory: Kontext & Gesprächshistorie')

st.write('''
LLMs sind standardmäßig **zustandslos** - sie haben kein Gedächtnis über vorherige 
Interaktionen. Für Chatbots und konversationelle Anwendungen brauchen wir aber **Memory**.

**Was du hier lernst:**
- 💬 Chat-Historie verwalten
- 🔄 Kontext über mehrere Turns bewahren
- ⚙️ Memory-Strategien (Full, Window, Summary)
- 🏗️ Memory mit LCEL
''')

st.info("💡 Mit Memory kann das LLM auf frühere Nachrichten Bezug nehmen!", icon="💡")

st.divider()

# -------------------------------------------------------------------
# Tabs for different sections
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📋 Memory-Strategien", "💻 Memory in LCEL", "🎮 Interaktive Demo"])

with tab1:
    st.write('''
    Je länger ein Gespräch wird, desto mehr Tokens werden verbraucht. Wir brauchen
    Strategien, um Memory zu managen:
    ''')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🗂️ Full History**
        
        Alle Nachrichten behalten
        
        ✅ Vollständiger Kontext  
        ❌ Token-Limit kann erreicht werden  
        ❌ Langsamer & teurer
        """)
    
    with col2:
        st.markdown("""
        **🪟 Window Memory**
        
        Nur letzte N Nachrichten
        
        ✅ Begrenzte Tokens  
        ✅ Schneller  
        ❌ Früher Kontext verloren
        """)
    
    with col3:
        st.markdown("""
        **📝 Summary Memory**
        
        Frühere Nachrichten zusammenfassen
        
        ✅ Kompakter Kontext  
        ✅ Längere Gespräche möglich  
        ❌ Komplexer zu implementieren
        """)

with tab2:
    st.subheader('💻 Memory in LCEL: Full History')
    
    st.code('''
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# Historie-Speicher
history = ChatMessageHistory()

# Prompt mit Message-Placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein hilfreicher Assistent."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Chain
llm = ChatOllama(model="llama3.2")
chain = prompt | llm | StrOutputParser()

# Gespräch führen
def chat(user_input):
    # Invoke mit aktueller Historie
    response = chain.invoke({
        "input": user_input,
        "history": history.messages
    })
    
    # Historie aktualisieren
    history.add_user_message(user_input)
    history.add_ai_message(response)
    
    return response
''', language='python')
    
    st.markdown("### 🪟 Window Memory")
    
    st.code('''
# Nur letzte N Nachrichten verwenden
def chat_with_window(user_input, window_size=4):
    messages = history.messages[-window_size:]  # Letzte 4 Nachrichten
    
    response = chain.invoke({
        "input": user_input,
        "history": messages
    })
    
    history.add_user_message(user_input)
    history.add_ai_message(response)
    
    return response
''', language='python')

with tab3:
    memory_strategy = st.selectbox(
        "Memory-Strategie",
        ["Vollständige Historie", "Window (letzte 4 Msgs)", "Window (letzte 2 Msgs)"],
        help="Wähle, wie viel Kontext das LLM sehen soll"
    )
    
    # Initialize session state
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ChatMessageHistory()
        st.session_state.messages = []
        st.session_state.current_strategy = memory_strategy
    
    # Reset if strategy changed
    if st.session_state.current_strategy != memory_strategy:
        st.session_state.chat_memory = ChatMessageHistory()
        st.session_state.messages = []
        st.session_state.current_strategy = memory_strategy
    
    # Display chat history
    st.markdown("### 💬 Gesprächsverlauf")
    
    if st.session_state.messages:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    else:
        st.info("👋 Starte ein Gespräch! Stelle Fragen, die sich auf frühere Nachrichten beziehen.")
    
    # Chat input
    if user_input := st.chat_input("Nachricht eingeben..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Prepare LLM
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Du bist ein freundlicher und hilfsbereiter Assistent. Beantworte Fragen präzise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Get relevant history based on strategy
        if memory_strategy == "Window (letzte 2 Msgs)":
            history_messages = st.session_state.chat_memory.messages[-2:] if len(st.session_state.chat_memory.messages) > 2 else st.session_state.chat_memory.messages
        elif memory_strategy == "Window (letzte 4 Msgs)":
            history_messages = st.session_state.chat_memory.messages[-4:] if len(st.session_state.chat_memory.messages) > 4 else st.session_state.chat_memory.messages
        else:
            history_messages = st.session_state.chat_memory.messages
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                response = chain.invoke({
                    "input": user_input,
                    "history": history_messages
                })
            st.write(response)
        
        # Update memory
        st.session_state.chat_memory.add_user_message(user_input)
        st.session_state.chat_memory.add_ai_message(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Show memory stats
    if st.session_state.chat_memory.messages:
        with st.expander("📊 Memory-Statistik"):
            total_msgs = len(st.session_state.chat_memory.messages)
            st.metric("Gesamt-Nachrichten", total_msgs)
            
            if memory_strategy == "Vollständige Historie":
                st.metric("Verwendete Nachrichten", total_msgs)
            elif memory_strategy == "Window (letzte 4 Msgs)":
                st.metric("Verwendete Nachrichten", min(4, total_msgs))
            else:
                st.metric("Verwendete Nachrichten", min(2, total_msgs))
            
            st.markdown("**Komplette Historie:**")
            for i, msg in enumerate(st.session_state.chat_memory.messages):
                st.caption(f"{i+1}. [{msg.type}] {msg.content[:50]}...")
    
    # Reset button
    if st.button("🔄 Gespräch zurücksetzen", use_container_width=True):
        st.session_state.chat_memory = ChatMessageHistory()
        st.session_state.messages = []
        st.rerun()

st.divider()

with st.expander("🎯 Best Practices für Memory"):
    st.markdown("""
    **Memory-Strategie wählen:**
    - 🤖 **Kurze FAQs**: Full History (wenig Tokens)
    - 💬 **Chat Support**: Window Memory (letzte 6-10 Nachrichten)
    - 📚 **Lange Gespräche**: Summary Memory oder Window
    - 💾 **Persistente Apps**: Vector Store als Long-Term Memory
    
    **Performance-Tipps:**
    - ⚡ Nur relevante Nachrichten senden
    - 🎯 Klare System-Prompts definieren
    - 📏 Token-Limits im Auge behalten
    - 💾 Wichtige Infos in Metadata speichern
    """)

st.caption("Workshop-Material: Memory Management für Konversationelle AI")
