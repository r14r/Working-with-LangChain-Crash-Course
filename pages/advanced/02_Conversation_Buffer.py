import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Conversation Buffer Memory
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Conversation Buffer | Advanced",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Advanced Conversation Memory")
st.caption("Learn different memory strategies for managing conversation history.")

st.markdown(
    """
Different memory types help manage conversation context efficiently:

- **Buffer Memory**: Store all messages (simple but grows)
- **Window Memory**: Keep only last N messages (bounded size)
- **Summary Memory**: Summarize old conversations (compact)

**What you'll learn:**
1. Different memory patterns  
2. Trade-offs between memory types  
3. When to use each strategy
"""
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(
        key="memory_model",
        location="sidebar"
    )
    
    memory_type = st.selectbox(
        "Memory Type:",
        ["Buffer Memory", "Window Memory (Last 3)", "Summary Memory"]
    )
    
    if st.button("🗑️ Clear Memory"):
        for key in ["buffer_history", "window_history", "summary_history", "summary_memory_obj"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --------------------------------------------------------------------------------------
# Initialize Memory Based on Type
# --------------------------------------------------------------------------------------
if memory_type == "Buffer Memory":
    if "buffer_history" not in st.session_state:
        st.session_state.buffer_history = []
    
    st.info("💾 **Buffer Memory**: Stores all messages. Simple but can grow very large.")
    current_memory = st.session_state.buffer_history
    
elif memory_type == "Window Memory (Last 3)":
    if "window_history" not in st.session_state:
        st.session_state.window_history = []
    
    st.info("🪟 **Window Memory**: Keeps only the last 3 message pairs. Fixed size, loses old context.")
    current_memory = st.session_state.window_history
    
else:  # Summary Memory
    if "summary_history" not in st.session_state:
        st.session_state.summary_history = []
    if "summary_memory_obj" not in st.session_state:
        st.session_state.summary_memory_obj = None
    
    st.info("📝 **Summary Memory**: Summarizes old messages. Compact but may lose details.")
    current_memory = st.session_state.summary_history

# --------------------------------------------------------------------------------------
# Display Chat
# --------------------------------------------------------------------------------------
st.subheader("Chat")

# Display messages
for message in current_memory:
    if isinstance(message, (HumanMessage, dict)) and (isinstance(message, dict) or message.type == "human"):
        content = message["content"] if isinstance(message, dict) else message.content
        with st.chat_message("user"):
            st.write(content)
    elif isinstance(message, (AIMessage, dict)) and (isinstance(message, dict) or message.type == "ai"):
        content = message["content"] if isinstance(message, dict) else message.content
        with st.chat_message("assistant"):
            st.write(content)

# --------------------------------------------------------------------------------------
# Chat Input
# --------------------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        # Add user message to memory
        user_msg = HumanMessage(content=user_input)
        
        with st.spinner("Thinking..."):
            llm = ChatOllama(model=model_name, temperature=0.7)
            parser = StrOutputParser()
            
            # Handle different memory types
            if memory_type == "Buffer Memory":
                # Simple: use all messages
                st.session_state.buffer_history.append(user_msg)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant with perfect memory."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                
                chain = prompt | llm | parser
                response = chain.invoke({
                    "history": st.session_state.buffer_history[:-1],
                    "input": user_input
                })
                
                st.session_state.buffer_history.append(AIMessage(content=response))
            
            elif memory_type == "Window Memory (Last 3)":
                # Keep only last 3 pairs (6 messages)
                st.session_state.window_history.append(user_msg)
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. You can only remember the last few messages."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                
                # Use only recent messages
                recent_history = st.session_state.window_history[-6:] if len(st.session_state.window_history) > 6 else st.session_state.window_history[:-1]
                
                chain = prompt | llm | parser
                response = chain.invoke({
                    "history": recent_history[:-1] if recent_history else [],
                    "input": user_input
                })
                
                st.session_state.window_history.append(AIMessage(content=response))
                
                # Trim to keep only last 6 messages
                if len(st.session_state.window_history) > 6:
                    st.session_state.window_history = st.session_state.window_history[-6:]
            
            else:  # Summary Memory
                # Summarize old messages when they exceed a threshold
                st.session_state.summary_history.append(user_msg)
                
                # If more than 6 messages, summarize the older ones
                if len(st.session_state.summary_history) > 6 and st.session_state.summary_memory_obj is None:
                    with st.spinner("Summarizing conversation history..."):
                        # Get messages to summarize (exclude last 4)
                        messages_to_summarize = st.session_state.summary_history[:-4]
                        
                        # Create summary
                        summary_prompt = ChatPromptTemplate.from_template(
                            "Summarize this conversation in 2-3 sentences:\n\n{conversation}"
                        )
                        
                        conv_text = "\n".join([
                            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                            for m in messages_to_summarize
                        ])
                        
                        summary_chain = summary_prompt | llm | parser
                        summary = summary_chain.invoke({"conversation": conv_text})
                        
                        st.session_state.summary_memory_obj = summary
                        
                        # Keep only summary + recent messages
                        st.session_state.summary_history = [
                            SystemMessage(content=f"Previous conversation summary: {summary}")
                        ] + st.session_state.summary_history[-4:]
                
                # Generate response
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                
                chain = prompt | llm | parser
                response = chain.invoke({
                    "history": st.session_state.summary_history[:-1],
                    "input": user_input
                })
                
                st.session_state.summary_history.append(AIMessage(content=response))
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(response)
    
    except Exception as exc:
        st.error(f"Error: {exc}")
        # Remove failed message
        current_memory.pop()

# --------------------------------------------------------------------------------------
# Memory Statistics
# --------------------------------------------------------------------------------------
if current_memory:
    st.subheader("📊 Memory Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", len(current_memory))
    
    with col2:
        total_chars = sum(len(str(m.content if hasattr(m, 'content') else m.get('content', ''))) for m in current_memory)
        st.metric("Total Characters", total_chars)
    
    with col3:
        if memory_type == "Summary Memory" and st.session_state.summary_memory_obj:
            st.metric("Summary Active", "Yes")
        else:
            memory_limit = {
                "Buffer Memory": "Unlimited",
                "Window Memory (Last 3)": "6 messages",
                "Summary Memory": "Dynamic"
            }[memory_type]
            st.metric("Limit", memory_limit)
    
    # Show summary if exists
    if memory_type == "Summary Memory" and st.session_state.summary_memory_obj:
        with st.expander("📝 Conversation Summary"):
            st.info(st.session_state.summary_memory_obj)

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 Memory Types Comparison"):
    st.markdown("""
    | Memory Type | Storage | Context Size | Use Case |
    |-------------|---------|--------------|----------|
    | **Buffer** | All messages | Growing | Short conversations |
    | **Window** | Last N messages | Fixed | Limited context needed |
    | **Summary** | Summary + recent | Bounded | Long conversations |
    
    **Buffer Memory:**
    ```python
    # Stores everything
    messages = []
    messages.append(HumanMessage(content="Hello"))
    messages.append(AIMessage(content="Hi!"))
    # ... continues growing
    ```
    
    **Window Memory:**
    ```python
    # Keep only last N
    window_size = 6
    if len(messages) > window_size:
        messages = messages[-window_size:]
    ```
    
    **Summary Memory:**
    ```python
    # Summarize old messages
    if len(messages) > threshold:
        old_messages = messages[:-keep]
        summary = summarize(old_messages)
        messages = [SystemMessage(summary)] + messages[-keep:]
    ```
    """)

with st.expander("💡 Choosing Memory Strategy"):
    st.markdown("""
    **Use Buffer Memory when:**
    - Conversations are short (< 20 messages)
    - You need complete history
    - Memory isn't a concern
    
    **Use Window Memory when:**
    - Only recent context matters
    - Predictable memory usage needed
    - Old context not relevant
    
    **Use Summary Memory when:**
    - Long conversations expected
    - Important to remember key points
    - Memory efficiency crucial
    
    **Production Tips:**
    - Add token counting to monitor usage
    - Implement conversation expiry
    - Store history in database
    - Consider user-specific limits
    """)

with st.expander("🧪 Test Different Memory Types"):
    st.markdown("""
    **Try this experiment:**
    
    1. Start with **Buffer Memory**
    2. Tell the assistant your name
    3. Have a 5-message conversation
    4. Ask "What's my name?" - should remember
    
    5. Switch to **Window Memory (Last 3)**
    6. Clear and repeat above
    7. Have a longer conversation (10 messages)
    8. Ask "What's my name?" - might forget!
    
    9. Switch to **Summary Memory**
    10. Clear and repeat
    11. Have a long conversation
    12. Check the summary - key info preserved!
    """)

with st.expander("🎯 Next Steps"):
    st.markdown("""
    - **03_MultiQuery_RAG.py**: Use memory in RAG
    - **04_Streaming_Chat.py**: Stream with memory
    - **Expert/08_Advanced_Memory.py**: Custom memory implementations
    """)
