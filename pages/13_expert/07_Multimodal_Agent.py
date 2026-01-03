import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Multimodal Agent | Expert", page_icon="🖼️", layout="centered")
st.title("🖼️ Multimodal Agent")
st.caption("Work with both text and images in LangChain.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Multimodal models can process:
    - Text
    - Images  
    - Combined text + image inputs

    **Note:** Requires multimodal models like `llava` or `bakllava`.
    """)

    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select multimodal model:",
            ["llava:latest", "bakllava:latest", "llava:7b"],
            key="mm_model"
        )
        st.info("Make sure the model is installed via Ollama.")

    st.subheader("Example: Image Description")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
        question = st.text_input(
            "Ask about the image:",
            placeholder="What do you see in this image?"
        )
    
        if st.button("🔍 Analyze"):
            if not question:
                st.warning("Enter a question.")
                st.stop()
        
            try:
                import base64
            
                with st.spinner("Analyzing image..."):
                    # Read and encode image
                    image_data = uploaded_file.read()
                    image_b64 = base64.b64encode(image_data).decode()
                
                    # Create multimodal message
                    llm = ChatOllama(model=model_name, temperature=0.7)
                
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                        ]
                    )
                
                    response = llm.invoke([message])
            
                st.markdown("### 💬 Response:")
                st.write(response.content)
        
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.info("Make sure you have a multimodal model installed.")

    with st.expander("📚 Multimodal Concepts"):
        st.markdown("""
        **Message Format:**
        ```python
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": "data:image/..."}
            ]
        )
        ```
    
        **Use Cases:**
        - Image captioning
        - Visual question answering  
        - Object detection description
        - Scene understanding
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
