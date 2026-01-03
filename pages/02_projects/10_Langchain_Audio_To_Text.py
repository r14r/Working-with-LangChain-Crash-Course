import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source


# --------------------------------------------------------------------------------------
# App: Local Voice Memo Transcriber + Optional Ollama Post-Processing
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Local Voice Memo Transcriber (Whisper + Ollama)",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 Local Voice Memo Transcriber")
st.caption("Transcribe audio locally with faster-whisper and optionally refine the text with Ollama.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    This app runs fully on your machine:

    - **Speech-to-text:** faster-whisper (local Whisper inference)
    - **Text refinement (optional):** Ollama via LangChain (`ChatOllama`)

    Typical refinements: clean up grammar, change tone, translate, or structure notes into sections.
    """
    )

    # --------------------------------------------------------------------------------------
    # Transcription settings
    # --------------------------------------------------------------------------------------
    st.subheader("1) Transcription (faster-whisper)")

    c1, c2, c3 = st.columns(3)
    with c1:
        whisper_size = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
            index=2,
            help="Larger models tend to be more accurate but slower.",
        )
    with c2:
        whisper_device = st.selectbox(
            "Device",
            ["cpu", "cuda"],
            index=0,
            help="cuda requires an NVIDIA GPU.",
        )
    with c3:
        whisper_compute = st.selectbox(
            "Compute type",
            ["int8", "int8_float16", "float16", "float32"],
            index=0,
            help="CPU: int8 is usually best. GPU: float16 is commonly used.",
        )

    language = st.selectbox(
        "Language",
        ["auto", "de", "en", "es", "fr", "it", "pt", "nl", "pl", "ru", "ja", "zh"],
        index=0,
        help="auto will try to detect the language.",
    )


    @st.cache_resource(show_spinner=False)
    def get_whisper_model(size: str, device: str, compute_type: str) -> WhisperModel:
        return WhisperModel(size, device=device, compute_type=compute_type)


    def transcribe_to_document(file_path: str, file_name: str) -> Document:
        model = get_whisper_model(whisper_size, whisper_device, whisper_compute)
        lang = None if language == "auto" else language

        segments, info = model.transcribe(
            file_path,
            language=lang,
            beam_size=5,
            vad_filter=False,  # Disabled VAD to avoid onnxruntime/numpy compatibility issues
        )

        # segments is an iterator; consume it once
        segment_list = list(segments)
        full_text = "".join(seg.text for seg in segment_list).strip()

        return Document(
            page_content=full_text,
            metadata={
                "file_name": file_name,
                "detected_language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
            },
        )


    # --------------------------------------------------------------------------------------
    # Post-processing settings
    # --------------------------------------------------------------------------------------
    st.subheader("2) Optional refinement (Ollama)")

    post_processing = st.checkbox("Refine transcript with a custom prompt", value=False)

    ollama_model = select_model(
        key="audio_model",
        location="sidebar",
        label="Ollama model"
    )

    custom_prompt = ""
    if post_processing:
        custom_prompt = st.text_area(
            "Custom prompt",
            placeholder="Given the following transcript, please …",
            help='A good pattern is: "Given the following transcript, …".',
            height=140,
        )

    post_prompt = ChatPromptTemplate.from_template(
        """{prompt}

    Transcript:
    {transcript}

    Return only the final result."""
    )

    post_chain = post_prompt | ChatOllama(model=ollama_model, temperature=0) | StrOutputParser()

    # --------------------------------------------------------------------------------------
    # Upload + Run
    # --------------------------------------------------------------------------------------
    st.subheader("3) Upload and run")

    voice_memos = st.file_uploader(
        "Upload audio files",
        type=["m4a", "mp4", "mp3", "wav", "flac", "ogg"],
        accept_multiple_files=True,
    )

    with st.form("run_transcription"):
        run = st.form_submit_button("🖊️ Transcribe")

    if run:
        if not voice_memos:
            st.warning("Please upload at least one audio file.")
            st.stop()

        with st.spinner("Transcribing…"):
            for voice_memo in voice_memos:
                file_stem, file_ext = os.path.splitext(voice_memo.name)
                tmp_path = None

                try:
                    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                        tmp.write(voice_memo.read())
                        tmp_path = tmp.name

                    doc = transcribe_to_document(tmp_path, file_stem)

                    with st.expander(f"{voice_memo.name}", expanded=True):
                        if language == "auto" and doc.metadata.get("detected_language"):
                            prob = doc.metadata.get("language_probability")
                            if prob is not None:
                                st.info(
                                    f"Detected language: {doc.metadata['detected_language']} ({prob:.2%})"
                                )
                            else:
                                st.info(f"Detected language: {doc.metadata['detected_language']}")

                        if post_processing and custom_prompt.strip():
                            refined = post_chain.invoke(
                                {"prompt": custom_prompt, "transcript": doc.page_content}
                            )
                            st.write(refined)
                        else:
                            if post_processing and not custom_prompt.strip():
                                st.warning("Refinement is enabled, but the custom prompt is empty. Showing raw transcript.")
                            st.write(doc.page_content)

                except Exception as exc:
                    st.error(f"Failed to process {voice_memo.name}: {exc}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

    with st.expander("Next improvements"):
        st.write(
            """
    - Add downloads (TXT / Markdown) per file
    - Keep session history in `st.session_state`
    - Chunk transcripts and add retrieval (RAG) over your notes
    - Add presets (translate, formalize, meeting minutes, chapter outline)
    """
        )

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
