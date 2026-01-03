import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

st.set_page_config(page_title="Human in the Loop | Expert", page_icon="🤝", layout="centered")

st.title("🤝 Human in the Loop")
st.caption("Build interactive workflows that require human input and approval.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Human-in-the-loop systems pause execution to:
    - Get human input/feedback
    - Request approval for actions
    - Allow corrections
    - Enable interactive refinement
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(key="hitl_model", location="sidebar")

    # Initialize session state
    if "hitl_state" not in st.session_state:
        st.session_state.hitl_state = "start"
        st.session_state.draft = None
        st.session_state.iterations = 0

    st.subheader("Example: Iterative Content Creation")

    st.info("""
    **Workflow:**
    1. AI generates draft
    2. Human reviews and provides feedback
    3. AI refines based on feedback
    4. Repeat until approved
    """)

    # Step 1: Initial request
    if st.session_state.hitl_state == "start":
        topic = st.text_input("What would you like content about?", placeholder="sustainable agriculture")
    
        if st.button("🚀 Generate Draft"):
            if not topic:
                st.warning("Enter a topic.")
                st.stop()
        
            try:
                with st.spinner("Generating draft..."):
                    llm = ChatOllama(model=model_name, temperature=0.7)
                    prompt = f"Write a short paragraph about: {topic}"
                    response = llm.invoke([HumanMessage(content=prompt)])
                
                    st.session_state.draft = response.content
                    st.session_state.topic = topic
                    st.session_state.hitl_state = "review"
                    st.session_state.iterations += 1
                    st.rerun()
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # Step 2: Human review
    elif st.session_state.hitl_state == "review":
        st.markdown(f"### Draft (Iteration {st.session_state.iterations}):")
        st.write(st.session_state.draft)
    
        st.markdown("---")
        st.subheader("Your Review")
    
        col1, col2 = st.columns(2)
    
        with col1:
            if st.button("✅ Approve", use_container_width=True):
                st.session_state.hitl_state = "approved"
                st.rerun()
    
        with col2:
            if st.button("🔄 Request Changes", use_container_width=True):
                st.session_state.hitl_state = "feedback"
                st.rerun()

    # Step 3: Get feedback
    elif st.session_state.hitl_state == "feedback":
        st.markdown("### Current Draft:")
        st.info(st.session_state.draft)
    
        feedback = st.text_area(
            "What would you like to change?",
            placeholder="Make it more technical and add statistics...",
            height=100
        )
    
        if st.button("💡 Regenerate with Feedback"):
            if not feedback:
                st.warning("Please provide feedback.")
                st.stop()
        
            try:
                with st.spinner("Refining based on your feedback..."):
                    llm = ChatOllama(model=model_name, temperature=0.7)
                    prompt = f"""Previous draft:
    {st.session_state.draft}

    Feedback: {feedback}

    Revise the draft based on the feedback."""
                
                    response = llm.invoke([HumanMessage(content=prompt)])
                
                    st.session_state.draft = response.content
                    st.session_state.iterations += 1
                    st.session_state.hitl_state = "review"
                    st.rerun()
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # Step 4: Final approved state
    elif st.session_state.hitl_state == "approved":
        st.success("✅ Content Approved!")
    
        st.markdown("### Final Version:")
        st.markdown(st.session_state.draft)
    
        st.info(f"Total iterations: {st.session_state.iterations}")
    
        if st.button("🔄 Start New"):
            st.session_state.hitl_state = "start"
            st.session_state.draft = None
            st.session_state.iterations = 0
            st.rerun()

    # Learning section
    with st.expander("📚 Human-in-the-Loop Patterns"):
        st.markdown("""
        **Common Patterns:**
    
        **1. Approval Gate:**
        - AI proposes action
        - Human approves/rejects
        - Execute if approved
    
        **2. Feedback Loop:**
        - AI generates output
        - Human provides feedback
        - AI refines
        - Repeat until satisfied
    
        **3. Interactive Refinement:**
        - AI creates draft
        - Human edits directly
        - AI learns from edits
    
        **4. Quality Control:**
        - AI processes data
        - Human reviews results
        - Flag issues for improvement
    
        **Implementation:**
        ```python
        # State-based approach
        if state == "draft":
            draft = ai_generate()
            state = "review"
    
        elif state == "review":
            approval = human_review(draft)
            if approval:
                state = "done"
            else:
                feedback = get_feedback()
                state = "refine"
    
        elif state == "refine":
            draft = ai_refine(draft, feedback)
            state = "review"
        ```
        """)

    with st.expander("💡 Use Cases"):
        st.markdown("""
        **Best for:**
        - High-stakes decisions
        - Creative work requiring judgment
        - Compliance requirements
        - Learning/training systems
        - Quality assurance
    
        **Examples:**
        - Content moderation
        - Medical diagnosis support
        - Legal document review
        - Financial trading approval
        - Customer service escalation
        """)

    with st.expander("🎯 Implementation Tips"):
        st.markdown("""
        **State Management:**
        - Use session state to track workflow
        - Persist state across interactions
        - Clear state appropriately
    
        **User Experience:**
        - Show progress indicators
        - Provide clear options
        - Save history
        - Allow undo
    
        **Technical:**
        - Handle timeouts
        - Save checkpoints
        - Enable async workflows
        - Support collaboration
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
