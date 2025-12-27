import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from lib.helper_streamlit import select_model

st.set_page_config(page_title="LangGraph Basics | Expert", page_icon="🕸️", layout="wide")

st.title("🕸️ LangGraph Basics")
st.caption("Build stateful, multi-step AI workflows with LangGraph.")

st.markdown("""
LangGraph extends LangChain with graph-based workflows for:
- Complex multi-step processes
- Conditional branching
- Cyclic workflows
- State management

**Key Concepts:**
- **States**: Data passed between nodes
- **Nodes**: Processing steps
- **Edges**: Connections between nodes
- **Conditional Edges**: Dynamic routing
""")

# Configuration
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(key="langgraph_model", location="sidebar")

# Define state
class GraphState(TypedDict):
    """State for the graph workflow."""
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    next_step: str

st.subheader("Example: Simple Graph Workflow")

st.info("""
This example creates a graph with three nodes:
1. **Start Node**: Initialize the process
2. **Process Node**: LLM processing
3. **End Node**: Finalize result
""")

# Show graph structure
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Graph Structure")
    st.code("""
START
  ↓
process_input
  ↓
generate_response
  ↓
END
    """)

with col2:
    st.markdown("### Node Functions")
    st.code("""
def process_input(state):
    # Process input
    return updated_state

def generate_response(state):
    # Call LLM
    return response
    """)

# User input
st.subheader("Try It Out")
user_input = st.text_input("Enter a message:", placeholder="Hello! Tell me about graphs.")

if st.button("🚀 Execute Graph"):
    if not user_input.strip():
        st.warning("Please enter a message.")
        st.stop()
    
    try:
        # Define nodes
        def process_input(state: GraphState) -> GraphState:
            """First node: process input."""
            st.info("📍 Node: process_input")
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=user_input))
            return {"messages": messages, "next_step": "generate"}
        
        def generate_response(state: GraphState) -> GraphState:
            """Second node: generate response."""
            st.info("📍 Node: generate_response")
            
            llm = ChatOllama(model=model_name, temperature=0.7)
            messages = state["messages"]
            response = llm.invoke(messages)
            
            messages.append(response)
            return {"messages": messages, "next_step": "end"}
        
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("process_input", process_input)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile
        app = workflow.compile()
        
        with st.spinner("Running graph..."):
            # Execute
            result = app.invoke({
                "messages": [],
                "next_step": "process"
            })
        
        st.markdown("### 💬 Final Result:")
        final_message = result["messages"][-1]
        st.write(final_message.content)
        
        with st.expander("🔍 View Full State"):
            st.json({
                "messages": [{"role": m.type, "content": m.content} for m in result["messages"]],
                "next_step": result["next_step"]
            })
    
    except Exception as exc:
        st.error(f"Error: {exc}")

# Learning section
with st.expander("📚 LangGraph Fundamentals"):
    st.markdown("""
    **Basic Structure:**
    ```python
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    
    # 1. Define state
    class State(TypedDict):
        messages: list
        step: str
    
    # 2. Create graph
    workflow = StateGraph(State)
    
    # 3. Add nodes
    workflow.add_node("node1", node1_function)
    workflow.add_node("node2", node2_function)
    
    # 4. Add edges
    workflow.set_entry_point("node1")
    workflow.add_edge("node1", "node2")
    workflow.add_edge("node2", END)
    
    # 5. Compile and run
    app = workflow.compile()
    result = app.invoke(initial_state)
    ```
    
    **Key Differences from Chains:**
    - Chains: Linear, sequential
    - Graphs: Non-linear, conditional
    - Graphs support loops and branching
    - Better for complex workflows
    """)

with st.expander("💡 When to Use LangGraph"):
    st.markdown("""
    **Use LangGraph for:**
    - Multi-step workflows with decisions
    - Cyclic processes (feedback loops)
    - Conditional branching
    - Complex state management
    - Agent coordination
    
    **Use Regular Chains for:**
    - Simple linear processes
    - Single-path workflows
    - Quick prototypes
    - Straightforward transformations
    
    **Examples:**
    - ✅ Graph: Multi-agent debate system
    - ✅ Graph: Self-correcting RAG
    - ❌ Chain: Simple Q&A
    - ❌ Chain: Text summarization
    """)

with st.expander("🎯 Next Steps"):
    st.markdown("""
    Continue with expert examples:
    - **02_Multi_Agent_System.py**: Multiple agents working together
    - **05_Conditional_Graph.py**: Branching logic in graphs
    - **06_Human_In_Loop.py**: Interactive workflows
    """)
