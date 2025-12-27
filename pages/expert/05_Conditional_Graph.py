import streamlit as st
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict
from lib.helper_streamlit import select_model

st.set_page_config(page_title="Conditional Graph | Expert", page_icon="🔀", layout="wide")
st.title("🔀 Conditional Graph Routing")
st.caption("Build graphs with dynamic branching based on conditions.")

with st.sidebar:
    model_name = select_model(key="conditional_model", location="sidebar")

class State(TypedDict):
    question: str
    category: str
    answer: str

st.subheader("Example: Question Router")

st.info("""
Routes questions to specialized handlers:
- **Math** → Math solver
- **Creative** → Creative generator
- **Other** → General handler
""")

question = st.text_input("Ask a question:", placeholder="What is 5 + 3?")

if st.button("🚀 Process with Conditional Graph"):
    if not question:
        st.warning("Enter a question.")
        st.stop()
    
    try:
        def classify_question(state: State) -> State:
            st.info("📍 Classifying question...")
            if any(word in state["question"].lower() for word in ["calculate", "+", "-", "*", "/"]):
                category = "math"
            elif any(word in state["question"].lower() for word in ["write", "story", "poem"]):
                category = "creative"
            else:
                category = "general"
            st.write(f"Category: {category}")
            return {**state, "category": category}
        
        def handle_math(state: State) -> State:
            st.info("📍 Math handler")
            llm = ChatOllama(model=model_name, temperature=0)
            answer = llm.invoke(f"Solve: {state['question']}")
            return {**state, "answer": answer.content}
        
        def handle_creative(state: State) -> State:
            st.info("📍 Creative handler")
            llm = ChatOllama(model=model_name, temperature=0.9)
            answer = llm.invoke(f"Create: {state['question']}")
            return {**state, "answer": answer.content}
        
        def handle_general(state: State) -> State:
            st.info("📍 General handler")
            llm = ChatOllama(model=model_name, temperature=0.7)
            answer = llm.invoke(state["question"])
            return {**state, "answer": answer.content}
        
        def route_question(state: State) -> str:
            """Decide which handler to use."""
            if state["category"] == "math":
                return "math_handler"
            elif state["category"] == "creative":
                return "creative_handler"
            else:
                return "general_handler"
        
        workflow = StateGraph(State)
        workflow.add_node("classify", classify_question)
        workflow.add_node("math_handler", handle_math)
        workflow.add_node("creative_handler", handle_creative)
        workflow.add_node("general_handler", handle_general)
        
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges("classify", route_question)
        workflow.add_edge("math_handler", END)
        workflow.add_edge("creative_handler", END)
        workflow.add_edge("general_handler", END)
        
        app = workflow.compile()
        
        with st.spinner("Running graph..."):
            result = app.invoke({"question": question, "category": "", "answer": ""})
        
        st.markdown("### 💬 Answer:")
        st.success(result["answer"])
    
    except Exception as exc:
        st.error(f"Error: {exc}")

with st.expander("📚 Conditional Routing"):
    st.markdown("""
    **Pattern:**
    ```python
    def router(state):
        if condition:
            return "node_a"
        else:
            return "node_b"
    
    workflow.add_conditional_edges(
        "source_node",
        router
    )
    ```
    """)
