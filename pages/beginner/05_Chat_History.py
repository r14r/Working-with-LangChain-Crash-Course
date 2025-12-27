import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Chat with History
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat History | Beginner",
    page_icon="💭",
    layout="centered",
)

st.title("💭 Chat with History")
st.caption("Learn how to maintain conversation context across multiple messages.")

st.markdown(
    """
Conversation memory allows the AI to remember previous messages in a chat. 
This makes the conversation feel natural, as the AI can reference what was 
said earlier.

**What you'll learn:**
1. Store and retrieve chat messages  
2. Use **MessagesPlaceholder** for conversation history  
3. Implement a stateful chat interface
"""
)

# --------------------------------------------------------------------------------------
# Model Selection
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(
        key="history_model",
        location="sidebar",
        label="Choose your model"
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

# --------------------------------------------------------------------------------------
# Initialize Chat History
# --------------------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --------------------------------------------------------------------------------------
# Display Chat History
# --------------------------------------------------------------------------------------
st.subheader("Chat")

# Display all previous messages
for message in st.session_state.chat_messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, SystemMessage):
        with st.chat_message("system"):
            st.write(message.content)

# --------------------------------------------------------------------------------------
# Chat Input
# --------------------------------------------------------------------------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.chat_messages.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        with st.spinner("Thinking..."):
            # Initialize LLM
            llm = ChatOllama(model=model_name, temperature=0.7)
            
            # Create prompt with history placeholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Have a natural conversation with the user."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # Create chain
            chain = prompt | llm | StrOutputParser()
            
            # Get response with history
            response = chain.invoke({
                "history": st.session_state.chat_messages[:-1],  # All messages except the current one
                "input": user_input
            })
            
            # Add AI response to history
            st.session_state.chat_messages.append(AIMessage(content=response))
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(response)
    
    except Exception as exc:
        st.error(f"Error: {exc}")
        # Remove the failed user message
        st.session_state.chat_messages.pop()

# --------------------------------------------------------------------------------------
# Chat Statistics
# --------------------------------------------------------------------------------------
if st.session_state.chat_messages:
    with st.expander("📊 Chat Statistics"):
        human_messages = sum(1 for m in st.session_state.chat_messages if isinstance(m, HumanMessage))
        ai_messages = sum(1 for m in st.session_state.chat_messages if isinstance(m, AIMessage))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_messages))
        with col2:
            st.metric("Your Messages", human_messages)
        with col3:
            st.metric("AI Messages", ai_messages)

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **Message Types**:
      - `HumanMessage`: User messages
      - `AIMessage`: AI responses
      - `SystemMessage`: System instructions
    
    - **MessagesPlaceholder**: Insert conversation history into prompts
    
    - **Session State**: Streamlit's way to persist data across reruns
    
    **How It Works:**
    ```python
    # Store messages in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Add messages
    st.session_state.chat_messages.append(
        HumanMessage(content="Hello")
    )
    
    # Use in prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Pass history to chain
    response = chain.invoke({
        "history": st.session_state.chat_messages,
        "input": user_input
    })
    ```
    
    **Benefits:**
    - Natural conversation flow
    - AI can reference earlier messages
    - Context-aware responses
    - Better user experience
    """)

with st.expander("💡 Try These"):
    st.markdown("""
    **Test the memory:**
    1. Tell the AI your name
    2. Ask it to remember something about you
    3. A few messages later, ask "What's my name?"
    
    **Example conversation:**
    - You: "My name is Alex and I love Python"
    - AI: "Nice to meet you, Alex! Python is a great language..."
    - You: "What programming language did I mention?"
    - AI: "You mentioned Python..."
    
    **Next steps:**
    - **06_Text_Embeddings.py**: Learn about embeddings
    - **Advanced/02_Conversation_Buffer.py**: Advanced memory patterns
    - **Advanced/04_Streaming_Chat.py**: Add streaming responses
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Message History Management:**
    
    In this example, we're storing the entire conversation in memory. 
    This is simple but has limitations:
    
    - **Memory grows** with each message
    - **Context window** limits (models have max token limits)
    - **Cost** (if using paid APIs, longer context = higher cost)
    
    **For production apps, consider:**
    - Limiting history to last N messages
    - Summarizing old conversations
    - Using conversation buffers
    - Implementing conversation memory classes
    
    These advanced patterns are covered in later examples!
    """)
