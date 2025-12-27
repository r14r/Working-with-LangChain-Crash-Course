import streamlit as st
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from lib.helper_streamlit import select_model

st.set_page_config(
    page_title="Prompts & Output Parser",
    page_icon="📝"
)

st.header("📝 Prompt Engineering & Strukturierte Outputs")

st.write("""
Prompts sind das Herzstück jeder LLM-Anwendung. Ein gut strukturierter Prompt macht
den Unterschied zwischen brauchbaren und exzellenten Ergebnissen.

**Was du hier lernst:**
- 🎯 Prompt Templates für Wiederverwendbarkeit
- 📊 Strukturierte Outputs mit Pydantic
- 🧩 Output Parser für konsistente Datenformate
""")

st.info(
    "💡 Mit lokalen Ollama-Modellen können wir prompt engineering ohne API-Kosten experimentieren!",
    icon="💡"
)

# -------------------------------------------------------------------
# Modellauswahl
# -------------------------------------------------------------------
model_name = select_model(
    key="prompts_model",
    location="sidebar",
    label="🤖 Wähle dein Ollama-Modell"
)

st.divider()

# -------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------
st.subheader("🎯 Prompt Templates")

st.write("""
Prompt Templates machen deine Prompts wiederverwendbar und wartbar. Statt jeden
Prompt neu zu schreiben, definierst du Variablen, die dynamisch gefüllt werden.
""")

st.code('''
from langchain_core.prompts import ChatPromptTemplate

template = "Du bist {rolle}. Erkläre {thema} in {stil} Sprache."
prompt = ChatPromptTemplate.from_template(template)

messages = prompt.format_messages(
    rolle="Informatik-Professor",
    thema="Rekursion",
    stil="einfacher"
)
''', language='python')

st.markdown("### 🎭 Demo: Firmenname-Generator")

with st.form("prompt_templates"):
    template = """\
Du bist ein kreativer Branding-Experte.
Generiere einen {adjective} Firmennamen für ein Unternehmen, das {product} herstellt.
Antworte nur mit dem Namen, ohne zusätzliche Erklärungen.
"""

    prompt_template = ChatPromptTemplate.from_template(template)

    name_type = st.selectbox(
        "Stil des Namens",
        ("innovativen", "traditionellen", "humorvollen", "minimalistischen")
    )

    business_type = st.text_input(
        "Produktkategorie",
        placeholder="z.B. nachhaltige Mode, KI-Software, Bio-Lebensmittel..."
    )

    execute = st.form_submit_button("🚀 Namen generieren", use_container_width=True)

    if execute and business_type.strip():
        with st.spinner(f"Kreative Ideen von {model_name}..."):
            chat = ChatOllama(model=model_name, temperature=0.9)
            messages = prompt_template.format_messages(
                adjective=name_type, 
                product=business_type
            )
            response = chat.invoke(messages)
        
        st.success("✨ Generierter Firmenname:")
        st.markdown(f"### {response.content}")

st.divider()

# -------------------------------------------------------------------
# Output Parser
# -------------------------------------------------------------------
st.subheader("📊 Strukturierte Outputs mit Pydantic")

st.write("""
LLMs geben standardmäßig Fließtext zurück. Für produktive Anwendungen benötigen
wir aber oft **strukturierte Daten** (JSON, Listen, validierte Objekte).

**Lösung**: Pydantic-Modelle + JsonOutputParser
""")

st.code('''
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class KundenFeedback(BaseModel):
    zufrieden: bool = Field(description="Ist der Kunde zufrieden?")
    stichworte: List[str] = Field(description="3 relevante Schlüsselwörter")
    bewertung: int = Field(ge=1, le=5, description="Bewertung 1-5")

parser = JsonOutputParser(pydantic_object=KundenFeedback)
''', language='python')

st.markdown("### 🔍 Demo: Feedback-Analyse")

class ReviewAnalysis(BaseModel):
    satisfied: Optional[bool] = Field(
        default=None,
        description="True wenn zufrieden, False wenn nicht, null wenn unklar."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Bis zu 3 relevante Stichwörter über Zufriedenheit/Qualität/Probleme."
    )

output_parser = JsonOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = output_parser.get_format_instructions()

with st.form("output_parsers"):
    review_template = """\
Analysiere das folgende Kundenfeedback zu einem Service.

Extrahiere:
- satisfied: War der Kunde zufrieden? (true/false/null bei Unklarheit)
- keywords: Bis zu 3 relevante Stichwörter

Feedback:
{text}

{format_instructions}
"""

    prompt_template = ChatPromptTemplate.from_template(review_template)

    review_text = st.text_area(
        "Kundenfeedback (auf Deutsch oder Englisch)",
        placeholder="Beispiel: Der Service war ausgezeichnet! Schnelle Lieferung und tolle Qualität. Preis-Leistung stimmt.",
        height=120
    )
    execute = st.form_submit_button("🔍 Analysieren", use_container_width=True)

    if execute and review_text.strip():
        with st.spinner("Analysiere Feedback..."):
            chat = ChatOllama(model=model_name, temperature=0)

            # Chain: prompt -> model -> JSON parser
            chain = prompt_template | chat | output_parser

            output = chain.invoke(
                {"text": review_text, "format_instructions": format_instructions}
            )

        st.success("✅ Strukturierte Analyse:")
        st.json(output)
        
        # Visuelle Darstellung
        if output.get("satisfied") is not None:
            status = "😀 Zufrieden" if output["satisfied"] else "😟 Unzufrieden"
            st.metric("Status", status)
        
        if output.get("keywords"):
            st.markdown("**Stichwörter:** " + ", ".join(output["keywords"]))

st.divider()

with st.expander("📚 Best Practices"):
    st.markdown("""
    **Prompt Templates:**
    - ✅ Verwende klare Variablennamen
    - ✅ Gib konkrete Instruktionen
    - ✅ Definiere gewünschtes Format
    
    **Output Parser:**
    - ✅ Nutze Pydantic für Typ-Validierung
    - ✅ Beschreibe Felder klar in `Field(description=...)`
    - ✅ Teste mit verschiedenen Inputs
    """)

st.caption("Workshop-Material: Prompt Engineering & Strukturierte Outputs")
