import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from lib.helper_streamlit import select_model

st.set_page_config(
    page_title="LCEL Chains",
    page_icon="⛓️"
)

st.header("⛓️ Chains: LLM-Pipelines mit LCEL")

st.write("""
**Chains** sind das Herzstück von LangChain. Sie erlauben uns, mehrere Komponenten 
(LLMs, Prompts, Parser, Tools) zu sequentiellen oder parallelen Workflows zu verbinden.

**LCEL (LangChain Expression Language)** ist die moderne, deklarative Art, Chains zu 
bauen. Statt imperativer Klassen nutzen wir den Pipe-Operator `|` für elegante Komposition.
""")

st.info("💡 LCEL-Chains sind composable, streamable und einfach testbar!", icon="💡")

st.divider()

# -------------------------------------------------------------------
# Basic Chain
# -------------------------------------------------------------------
st.subheader("🔗 Basis-Chain mit LCEL")

st.write("""
Die einfachste Chain: `Prompt | LLM | Parser`

Jede Komponente ist ein "Runnable" und kann mit `|` verkettet werden.
""")

st.code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2", temperature=0.7)
prompt = ChatPromptTemplate.from_template("Erkläre {topic} in einfachen Worten")
parser = StrOutputParser()

# Chain-Komposition mit Pipe-Operator
chain = prompt | llm | parser

# Aufruf
result = chain.invoke({"topic": "Quantencomputing"})
""", language='python')

st.markdown("### 🎬 Demo: Filmtitel-Generator")

# Model selection in sidebar
model_name = select_model(key="chains_model", location="sidebar", label="Modell wählen")

# Shared LLM instance
llm = ChatOllama(model=model_name, temperature=0.9)
parser = StrOutputParser()

basic_prompt = ChatPromptTemplate.from_template("""
Du bist ein kreativer Drehbuchautor. 
Erfinde einen alternativen Filmtitel für: "{movie}"
Der Titel soll die Geschichte respektieren.
Antworte nur mit dem Titel, nichts weiter.
""")

basic_chain = basic_prompt | llm | parser

with st.form("basic_chain"):
    movie_input = st.text_input(
        "Original-Filmtitel", 
        placeholder="z.B. Die Verurteilten, Inception, Forrest Gump",
        key="movie_basic"
    )
    execute_btn = st.form_submit_button("✨ Generiere alternativen Titel", use_container_width=True)

    if execute_btn and movie_input.strip():
        with st.spinner("Kreative Idee wird entwickelt..."):
            result = basic_chain.invoke({"movie": movie_input})
        
        st.success("🎬 Alternativer Filmtitel:")
        st.markdown(f"### {result}")

st.divider()

# -------------------------------------------------------------------
# Sequential Chain
# -------------------------------------------------------------------
st.subheader("🔄 Sequentielle Chains")

st.write("""
Oft brauchen wir **mehrere LLM-Aufrufe hintereinander**, wobei der Output des ersten
als Input für den zweiten dient.

**Beispiel**: 
1. Generiere alternativen Filmtitel
2. Schreibe Werbetext für diesen Titel
""")

st.code("""
# Zwei Prompts
first_prompt = ChatPromptTemplate.from_template("Alternativer Titel für {movie}")
second_prompt = ChatPromptTemplate.from_template("Werbetext für Film: {title}")

# Sequentielle Pipeline
pipeline = (
    first_prompt 
    | llm 
    | StrOutputParser()
    | RunnableLambda(lambda title: {"title": title})
    | second_prompt
    | llm
    | StrOutputParser()
)

result = pipeline.invoke({"movie": "Matrix"})
""", language='python')

st.markdown("### 🎯 Demo: Film-Marketing-Pipeline")

# Define prompts
second_prompt = ChatPromptTemplate.from_template("""
Schreibe einen knackigen Werbetext (max. 25 Wörter) für den Film mit Titel:
"{movie_title}"

Der Text soll Spannung erzeugen und zum Kinobesuch motivieren.
""")

# Build chains
first_chain = basic_prompt | llm | parser
second_chain = second_prompt | llm | parser

# Sequential pipeline with RunnableLambda for data transformation
sequential_pipeline = (
    first_chain
    | RunnableLambda(lambda title: {"movie_title": title.strip()})
    | second_chain
)

with st.form("sequential_chain"):
    movie_seq = st.text_input(
        "Original-Film", 
        placeholder="z.B. Blade Runner, Titanic, Interstellar",
        key="movie_seq"
    )
    execute_seq = st.form_submit_button("🚀 Pipeline starten", use_container_width=True)

    if execute_seq and movie_seq.strip():
        with st.spinner("Pipeline läuft..."):
            # Execute both steps
            new_title = first_chain.invoke({"movie": movie_seq}).strip()
            ad_copy = sequential_pipeline.invoke({"movie": movie_seq}).strip()

        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🎬 Neuer Titel")
            st.markdown(f"**{new_title}**")
        
        with col2:
            st.success("📢 Werbetext")
            st.write(ad_copy)

st.divider()

# -------------------------------------------------------------------
# Advantages & Best Practices
# -------------------------------------------------------------------
st.subheader("💡 Warum Chains verwenden?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Vorteile von LCEL:**
    - ✅ **Lesbar**: Klare Pipe-Syntax
    - ✅ **Testbar**: Jede Komponente einzeln
    - ✅ **Debuggbar**: Zwischenergebnisse inspizieren
    - ✅ **Streambar**: Ergebnisse live streamen
    """)

with col2:
    st.markdown("""
    **Best Practices:**
    - 🎯 Klein beginnen, dann verketten
    - 🧪 Komponenten einzeln testen
    - 📝 Aussagekräftige Prompt-Templates
    - ⚡ Caching für wiederholte Aufrufe
    """)

with st.expander("🤔 Warum nicht alles in einem Prompt?"):
    st.markdown("""
    **Mehrere Schritte sind besser als einer, weil:**
    
    1. **Präzisere Ergebnisse**: Fokussierte Prompts → bessere Outputs
    2. **Debugging**: Fehler in einzelnen Schritten identifizieren
    3. **Wiederverwendbarkeit**: Komponenten in anderen Chains nutzen
    4. **Flexibilität**: Dynamisches Routing basierend auf Zwischenergebnissen
    5. **Modularität**: Einfach LLMs oder Prompts austauschen
    """)

st.divider()

st.caption("Workshop-Material: LCEL Chains & Pipeline-Patterns")
