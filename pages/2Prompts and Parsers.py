import streamlit as st
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(
    page_title="Learn LangChain | Prompts and Parsers (Ollama)",
    page_icon="📝"
)

st.header("📝 Prompts and Parsers (with Ollama)")

st.write("""
Prompts are another key component in LangChain and in general in Large Language Models.
A well structured prompt can make all the difference in getting a quality outcome.
""")

st.info(
    "Prompt engineering allows us to use local LLMs efficiently with clear instructions.",
    icon="ℹ️"
)

# -------------------------------------------------------------------
# Model selection (local Ollama)
# -------------------------------------------------------------------
model_name = st.selectbox(
    "Ollama model",
    ["llama3", "mistral", "gemma", "qwen2.5"],
    index=0
)

# -------------------------------------------------------------------
# Prompt Templates demo
# -------------------------------------------------------------------
st.subheader("Prompt Templates")

with st.form("prompt_templates"):
    template = """\
You are a naming consultant for new companies.
What is a good {adjective} name for a company that makes {product}?
Please reply only with the name, no comments attached.
"""

    prompt_template = ChatPromptTemplate.from_template(template)

    name_type = st.selectbox(
        "Name type",
        ("funny", "serious", "irreverent")
    )

    business_type = st.text_input(
        "Type of business",
        placeholder="fruit shop, fashion atelier, ..."
    )

    execute = st.form_submit_button("🚀 Execute")

    if execute and business_type:
        chat = ChatOllama(model=model_name, temperature=0.9)
        messages = prompt_template.format_messages(adjective=name_type, product=business_type)
        response = chat.invoke(messages)
        st.code(response.content)

# -------------------------------------------------------------------
# Output Parsers demo (modern replacement)
# -------------------------------------------------------------------
st.subheader("Output Parsers")

class ReviewAnalysis(BaseModel):
    satisfied: Optional[bool] = Field(
        default=None,
        description="True if satisfied, False if not, null if unclear."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Up to 3 relevant keywords about satisfaction/value/issues."
    )

output_parser = JsonOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = output_parser.get_format_instructions()

with st.form("output_parsers"):
    review_template = """\
The following text is a review from a customer reviewing a service.

Extract:
- satisfied: Was the customer satisfied? Return true/false/null (null if unclear).
- keywords: Up to 3 relevant keywords.

Text:
{text}

{format_instructions}
"""

    prompt_template = ChatPromptTemplate.from_template(review_template)

    review_text = st.text_area("Customer review")
    execute = st.form_submit_button("🚀 Execute")

    if execute and review_text:
        chat = ChatOllama(model=model_name, temperature=0)

        # Build a runnable chain: prompt -> model -> JSON parser
        chain = prompt_template | chat | output_parser

        # JsonOutputParser expects the model to output JSON matching the schema
        output = chain.invoke(
            {"text": review_text, "format_instructions": format_instructions}
        )

        st.json(output)
