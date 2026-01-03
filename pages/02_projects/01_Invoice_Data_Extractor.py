import os
import tempfile
import streamlit as st
from typing import Optional

from pydantic import BaseModel, Field
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader


st.set_page_config(
    page_title="Invoice Data Extractor | Learn LangChain",
    page_icon="🧾"
)

st.header("🧾 Invoice Data Extractor")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.write("""
    In this project we use document loaders, prompts and parsers to extract structured data
    from PDF invoices and return them as JSON.
    """)

    st.info(
        "You need your own keys to run commercial LLM models. "
        "The form will process your keys safely and never store them anywhere.",
        icon="🔒"
    )

    model_name = select_model(key="invoice_model", location="sidebar", label="Select Model")

    openai_key = st.text_input("OpenAI Api Key", type="password")
    invoice_file = st.file_uploader("Upload a PDF invoice", type=["pdf"])


    # -----------------------------
    # Pydantic schema (modern replacement for ResponseSchema list)
    # -----------------------------
    class InvoiceData(BaseModel):
        number: Optional[str] = Field(default=None, description="Invoice number, null if unclear.")
        date: Optional[str] = Field(default=None, description="Issued date as mm-dd-yyyy, null if unclear.")
        company: Optional[str] = Field(default=None, description="Company name, null if unclear.")
        address: Optional[str] = Field(default=None, description="Full address: address, city (state), country. Null if unclear.")
        service: Optional[str] = Field(default=None, description="Service purchased, null if unclear.")
        total: Optional[float] = Field(default=None, description="Grand total as a number, null if unclear.")


    parser = JsonOutputParser(pydantic_object=InvoiceData)
    format_instructions = parser.get_format_instructions()

    template = """\
    This document is an invoice. Extract the following information and return it as JSON.

    Rules:
    - If a field is unclear, return null.
    - date must be formatted as mm-dd-yyyy.
    - total must be a number (no currency symbol).

    Text:
    {text}

    {format_instructions}
    """

    prompt_template = ChatPromptTemplate.from_template(template)


    if invoice_file is not None:
        if not openai_key.strip():
            st.warning("Please enter your OpenAI API key.")
            st.stop()

        with st.spinner("Processing your request..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                    temporary_file.write(invoice_file.read())
                    tmp_path = temporary_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                # Merge PDF pages into a single text block
                text_invoice = "\n\n".join(d.page_content for d in docs)

                llm = ChatOllama(model=model_name, temperature=0)


                # Modern chain: prompt -> model -> JSON parser
                chain = prompt_template | llm | parser
                json_invoice = chain.invoke(
                    {"text": text_invoice, "format_instructions": format_instructions}
                )

                st.write("Here is your JSON invoice:")
                st.json(json_invoice)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


    with st.expander("Exercise Tips"):
        st.write("""
        - Browse the code on GitHub and make sure you understand it.
        - Fork the repository to customize the code.
        - Try to add new elements (VAT number, taxes, payment terms).
        - Improve this code with additional validation and error recovery if JSON parsing fails.
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
