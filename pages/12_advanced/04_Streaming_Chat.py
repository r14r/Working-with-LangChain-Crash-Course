import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Streaming Chat
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Streaming Chat | Advanced",
    page_icon="🌊",
    layout="centered",
)

st.title("🌊 Streaming Chat Responses")
st.caption("Learn to stream AI responses token-by-token for better UX.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    Streaming provides a better user experience by showing responses as they're 
    generated, rather than waiting for the complete response.

    **Benefits:**
    - Immediate feedback
    - Perceived faster responses
    - Better for long outputs
    """
    )

    # --------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(
            key="stream_model",
            location="sidebar"
        )
    
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
        if st.button("🗑️ Clear Chat"):
            st.session_state.stream_messages = []
            st.rerun()

    # Initialize
    if "stream_messages" not in st.session_state:
        st.session_state.stream_messages = []

    # --------------------------------------------------------------------------------------
    # Display Chat History
    # --------------------------------------------------------------------------------------
    st.subheader("Chat")

    for message in st.session_state.stream_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # --------------------------------------------------------------------------------------
    # Chat Input
    # --------------------------------------------------------------------------------------
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Add user message
        st.session_state.stream_messages.append(HumanMessage(content=user_input))
    
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
    
        try:
            # Stream AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
            
                # Create prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Respond naturally and conversationally."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
            
                # Create streaming chain
                llm = ChatOllama(
                    model=model_name,
                    temperature=temperature,
                    streaming=True
                )
            
                chain = prompt | llm | StrOutputParser()
            
                # Stream response
                for chunk in chain.stream({
                    "history": st.session_state.stream_messages[:-1],
                    "input": user_input
                }):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
                message_placeholder.markdown(full_response)
        
            # Save AI response
            st.session_state.stream_messages.append(AIMessage(content=full_response))
    
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.session_state.stream_messages.pop()

    # --------------------------------------------------------------------------------------
    # Stats
    # --------------------------------------------------------------------------------------
    if st.session_state.stream_messages:
        with st.expander("📊 Chat Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Messages", len(st.session_state.stream_messages))
            with col2:
                chars = sum(len(m.content) for m in st.session_state.stream_messages)
                st.metric("Total Characters", chars)

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 How Streaming Works"):
        st.markdown("""
        **Streaming Pattern:**
    
        ```python
        # Enable streaming in LLM
        llm = ChatOllama(
            model="llama2",
            streaming=True  # Enable streaming
        )
    
        # Use .stream() instead of .invoke()
        for chunk in chain.stream(input_data):
            # Process each chunk as it arrives
            print(chunk, end="")
        ```
    
        **Streamlit Implementation:**
        ```python
        placeholder = st.empty()
        full_response = ""
    
        for chunk in chain.stream(input):
            full_response += chunk
            placeholder.markdown(full_response + "▌")  # Cursor effect
    
        placeholder.markdown(full_response)  # Final
        ```
    
        **Key Concepts:**
        - **Chunks**: Small pieces of the response
        - **Accumulation**: Building the full response
        - **Real-time Display**: Updating UI progressively
        """)

    with st.expander("💡 Streaming Benefits"):
        st.markdown("""
        **User Experience:**
        - ✅ Immediate feedback
        - ✅ Engagement during generation
        - ✅ Can interrupt if needed
    
        **Technical:**
        - ✅ Lower perceived latency
        - ✅ Better for long responses
        - ✅ Progressive rendering
    
        **Use Cases:**
        - Long-form content generation
        - Interactive conversations
        - Real-time translation
        - Code generation
    
        **When NOT to Stream:**
        - Need complete response for processing
        - Parsing structured output
        - Response needs validation first
        """)

    with st.expander("🎯 Try This"):
        st.markdown("""
        **Test streaming behavior:**
    
        1. Ask for a long response:
           - "Write a 500-word essay about AI"
           - "Explain quantum computing in detail"
    
        2. Notice how text appears gradually
    
        3. Compare with non-streaming (single block)
    
        4. Try interrupting (refresh) mid-stream
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
