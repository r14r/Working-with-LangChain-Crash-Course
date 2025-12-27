import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from lib.helper_streamlit import select_model

st.set_page_config(page_title="Advanced Memory | Expert", page_icon="🧠", layout="centered")
st.title("🧠 Advanced Memory Implementations")
st.caption("Custom memory strategies for complex applications.")

with st.sidebar:
    model_name = select_model(key="adv_mem_model", location="sidebar")

if "custom_memory" not in st.session_state:
    st.session_state.custom_memory = {
        "messages": [],
        "facts": {},  # Long-term facts
        "summary": None
    }

st.subheader("Multi-Layer Memory System")

st.info("""
**Memory Layers:**
- **Short-term**: Recent conversation
- **Long-term**: Extracted facts
- **Summary**: Condensed history
""")

# Show current memory state
with st.expander("🧠 Memory State"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.custom_memory["messages"]))
        st.metric("Facts", len(st.session_state.custom_memory["facts"]))
    with col2:
        has_summary = st.session_state.custom_memory["summary"] is not None
        st.metric("Summary", "Yes" if has_summary else "No")

# Chat interface
user_input = st.chat_input("Message...")

if user_input:
    # Add to short-term memory
    st.session_state.custom_memory["messages"].append(
        HumanMessage(content=user_input)
    )
    
    try:
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        # Extract facts if user shares information
        if any(word in user_input.lower() for word in ["my name is", "i am", "i like", "i work"]):
            fact_prompt = f"Extract the key fact from: '{user_input}'. Return just the fact."
            fact = llm.invoke([HumanMessage(content=fact_prompt)])
            st.session_state.custom_memory["facts"][len(st.session_state.custom_memory["facts"])] = fact.content
        
        # Create context with all memory layers
        context_parts = []
        
        # Add long-term facts
        if st.session_state.custom_memory["facts"]:
            facts_str = "\n".join(st.session_state.custom_memory["facts"].values())
            context_parts.append(f"Known facts:\n{facts_str}")
        
        # Add recent messages
        recent = st.session_state.custom_memory["messages"][-4:]
        
        # Generate response
        response = llm.invoke(recent)
        st.session_state.custom_memory["messages"].append(AIMessage(content=response.content))
        
        # Display
        with st.chat_message("assistant"):
            st.write(response.content)
    
    except Exception as exc:
        st.error(f"Error: {exc}")

with st.expander("📚 Advanced Memory Patterns"):
    st.markdown("""
    **Custom Memory Architectures:**
    
    1. **Hierarchical Memory:**
       - Working memory (immediate)
       - Short-term (recent)
       - Long-term (persistent facts)
    
    2. **Semantic Memory:**
       - Store knowledge separately
       - Index by topic/entity
       - Efficient retrieval
    
    3. **Episodic Memory:**
       - Time-tagged events
       - Contextual recall
       - Memory consolidation
    """)
