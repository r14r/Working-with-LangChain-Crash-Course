import validators
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source


# --------------------------------------------------------------------------------------
# App: Web Page Summarizer (Local) — Ollama + Modern LangChain (LCEL)
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Web Page Summarizer (Ollama + LangChain)",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Web Page Summarizer")
st.caption("Summarize any public web page using local Ollama models and modern LangChain (LCEL).")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    This app pulls a web page, extracts text, and summarizes it using one of three strategies:

    - **stuff**: summarize everything in one go (fast, but can hit context limits)
    - **map_reduce**: summarize chunks, then summarize the summaries (scales better)
    - **refine**: build a summary iteratively, refining it chunk by chunk (often higher fidelity, slower)
    """
    )

    # --------------------------------------------------------------------------------------
    # Controls
    # --------------------------------------------------------------------------------------

    with st.sidebar:
        st.header("⚙️ Configuration")
    
        ollama_model = select_model(
            key="summarization_model",
            location="sidebar",
            label="Ollama chat model"
        )
    
        chain_type = st.selectbox(
            "Summarization strategy",
            ("stuff", "map_reduce", "refine"),
            index=1,
        )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk size (chars)", 800, 6000, 2400, 200)
    with col2:
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 800, 200, 50)

    url = st.text_input(
        "Web page URL",
        placeholder="https://example.com/some-article",
    )

    run = st.button("🚀 Summarize", type="primary")

    # --------------------------------------------------------------------------------------
    # Model + shared components
    # --------------------------------------------------------------------------------------
    llm = ChatOllama(model=ollama_model, temperature=0.2)
    parser = StrOutputParser()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Prompts (LCEL)
    stuff_prompt = ChatPromptTemplate.from_template(
        """You are a careful summarizer.

    Summarize the text below in a clear, structured way:
    - 5–10 bullet points for key takeaways
    - 1 short paragraph for overall summary

    Text:
    {text}

    Summary:"""
    )

    map_prompt = ChatPromptTemplate.from_template(
        """Summarize this chunk in 3–5 bullet points, focusing on factual content.

    Chunk:
    {text}

    Chunk summary:"""
    )

    reduce_prompt = ChatPromptTemplate.from_template(
        """You are consolidating multiple partial summaries into one final summary.

    Combine and deduplicate the points. Produce:
    - 8–12 bullet points max
    - 1 short concluding paragraph

    Partial summaries:
    {summaries}

    Final summary:"""
    )

    refine_init_prompt = ChatPromptTemplate.from_template(
        """You are a summarizer.

    Create an initial summary for the first chunk:
    {text}

    Initial summary:"""
    )

    refine_step_prompt = ChatPromptTemplate.from_template(
        """You are refining an existing summary with new information.

    Current summary:
    {summary}

    New chunk:
    {text}

    Update the summary to include important new facts.
    Keep it concise and structured (bullets + short paragraph).

    Refined summary:"""
    )

    # Chains
    stuff_chain = stuff_prompt | llm | parser
    map_chain = map_prompt | llm | parser
    reduce_chain = reduce_prompt | llm | parser
    refine_init_chain = refine_init_prompt | llm | parser
    refine_step_chain = refine_step_prompt | llm | parser


    def load_web_text(web_url: str) -> str:
        loader = WebBaseLoader(web_url)
        docs = loader.load()  # list of Documents
        return "\n\n".join(d.page_content for d in docs).strip()


    def summarize_stuff(text: str) -> str:
        return stuff_chain.invoke({"text": text})


    def summarize_map_reduce(chunks: list[str]) -> str:
        summaries: list[str] = []
        prog = st.progress(0, text="Summarizing chunks…")

        total = max(len(chunks), 1)
        for i, chunk in enumerate(chunks, start=1):
            s = map_chain.invoke({"text": chunk})
            summaries.append(s)
            prog.progress(int(i / total * 100), text=f"Summarizing chunks… ({i}/{total})")

        prog.progress(100, text="Reducing summaries…")
        return reduce_chain.invoke({"summaries": "\n\n---\n\n".join(summaries)})


    def summarize_refine(chunks: list[str]) -> str:
        if not chunks:
            return "No content to summarize."

        prog = st.progress(0, text="Building initial summary…")
        summary = refine_init_chain.invoke({"text": chunks[0]})

        total = max(len(chunks), 1)
        for i in range(1, len(chunks)):
            prog.progress(int(i / total * 100), text=f"Refining summary… ({i+1}/{total})")
            summary = refine_step_chain.invoke({"summary": summary, "text": chunks[i]})

        prog.progress(100, text="Done.")
        return summary


    # --------------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------------
    if run:
        if not validators.url(url):
            st.warning("Please enter a valid URL.")
            st.stop()

        with st.spinner("Loading page…"):
            raw_text = load_web_text(url)

        if not raw_text:
            st.error("No text could be extracted from this page.")
            st.stop()

        with st.expander("Extracted text preview", expanded=False):
            st.text_area("Preview", raw_text[:8000], height=240)

        with st.spinner("Splitting into chunks…"):
            chunks = splitter.split_text(raw_text)

        st.caption(f"Extracted {len(raw_text):,} characters · {len(chunks)} chunk(s)")

        with st.spinner("Generating summary…"):
            if chain_type == "stuff":
                result = summarize_stuff(raw_text)
            elif chain_type == "map_reduce":
                result = summarize_map_reduce(chunks)
            else:
                result = summarize_refine(chunks)

        st.subheader("Summary")
        st.write(result)

        with st.expander("Debug: chunks", expanded=False):
            for idx, c in enumerate(chunks[:10], start=1):
                st.markdown(f"**Chunk {idx}** ({len(c)} chars)")
                st.write(c)
            if len(chunks) > 10:
                st.info(f"Showing first 10 of {len(chunks)} chunks.")


    with st.expander("Upgrade ideas"):
        st.write(
            """
    - Add a “target length” control (short / medium / long summary)
    - Add citation output (show which chunks the bullets came from)
    - Use a retriever (RAG) for Q&A over the page content
    - Cache loaded pages by URL to speed up repeated runs
    """
        )

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
