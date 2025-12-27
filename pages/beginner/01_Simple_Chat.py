import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Simple Chat with Ollama
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Simple Chat | Beginner",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Simple Chat with Ollama")
st.caption("Learn the basics of chatting with an LLM using LangChain and Ollama.")

st.markdown(
    """
This is your first LangChain example! This app demonstrates:

1. **Connect** to a local Ollama model  
2. **Send** a simple message  
3. **Receive** a response from the AI

Everything runs locally—no API keys needed.
"""
)

# --------------------------------------------------------------------------------------
# Model Selection
# --------------------------------------------------------------------------------------
st.subheader("1) Select Model")

with st.sidebar:
    st.header("Configuration")
    model_name = select_model(
        key="simple_chat_model",
        location="sidebar",
        label="Choose your model"
    )

# --------------------------------------------------------------------------------------
# Simple Chat Interface
# --------------------------------------------------------------------------------------
st.subheader("2) Chat with the AI")

user_message = st.text_area(
    "Your message:",
    placeholder="Hello! Can you tell me what LangChain is?",
    height=100
)

if st.button("🚀 Send Message"):
    if not user_message.strip():
        st.warning("Please enter a message first.")
        st.stop()

    try:
        with st.spinner("Getting response..."):
            # Initialize the Ollama chat model
            llm = ChatOllama(model=model_name, temperature=0.7)
            
            # Create a simple message
            messages = [HumanMessage(content=user_message)]
            
            # Get response
            response = llm.invoke(messages)
            
        st.markdown("### AI Response:")
        st.write(response.content)
        
    except Exception as exc:
        st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **ChatOllama**: The LangChain class to interact with Ollama models
    - **HumanMessage**: Represents a message from the user
    - **invoke()**: Method to send messages and get a response
    
    **Try this:**
    - Change the temperature (0.0 = more focused, 1.0 = more creative)
    - Try different models
    - Ask different types of questions
    """)

with st.expander("💡 Next Steps"):
    st.markdown("""
    Now that you understand basic chat, try:
    - **02_Prompt_Template.py**: Learn to structure your prompts
    - **03_Output_Parser.py**: Parse AI responses in specific formats
    - **04_Simple_Chain.py**: Combine multiple steps together
    """)
