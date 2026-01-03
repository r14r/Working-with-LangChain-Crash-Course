import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Custom Chains
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Custom Chains | Advanced",
    page_icon="⛓️",
    layout="centered",
)

st.title("⛓️ Custom LangChain Chains")
st.caption("Learn to build sophisticated custom chains with LCEL.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    Advanced chains go beyond simple linear flows. You can:
    - Pass through data unchanged
    - Transform data with custom functions
    - Branch based on conditions
    - Combine multiple outputs

    **What you'll learn:**
    1. **RunnablePassthrough**: Pass data through unchanged  
    2. **RunnableLambda**: Apply custom functions  
    3. **Complex Chain Patterns**: Multi-step data transformations
    """
    )

    # --------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(
            key="custom_chain_model",
            location="sidebar"
        )

    # --------------------------------------------------------------------------------------
    # Chain Examples
    # --------------------------------------------------------------------------------------
    st.subheader("Choose a Custom Chain Example")

    example = st.radio(
        "Select an example:",
        [
            "Passthrough Chain",
            "Data Transformation Chain",
            "Multi-Step Analysis Chain"
        ]
    )

    # --------------------------------------------------------------------------------------
    # Example 1: Passthrough Chain
    # --------------------------------------------------------------------------------------
    if example == "Passthrough Chain":
        st.info("Pass input data through the chain while also using it in prompts.")
    
        topic = st.text_input("Topic:", placeholder="quantum computing")
    
        if st.button("🚀 Run Chain"):
            if not topic.strip():
                st.warning("Please enter a topic.")
                st.stop()
        
            try:
                with st.spinner("Running passthrough chain..."):
                    # Create chain that passes through original input
                    # while also generating content
                
                    prompt = ChatPromptTemplate.from_template(
                        "Write a one-sentence definition of {topic}."
                    )
                
                    llm = ChatOllama(model=model_name, temperature=0.7)
                    parser = StrOutputParser()
                
                    # Chain that returns both input and output
                    chain = {
                        "topic": RunnablePassthrough(),
                        "definition": prompt | llm | parser
                    }
                
                    result = chain.invoke(topic)
            
                st.markdown("### Results:")
                st.write(f"**Original Topic:** {result['topic']}")
                st.write(f"**Generated Definition:** {result['definition']}")
            
                with st.expander("🔍 How it works"):
                    st.code("""
    # RunnablePassthrough passes data unchanged
    chain = {
        "topic": RunnablePassthrough(),
        "definition": prompt | llm | parser
    }

    # Input: "quantum computing"
    # Output: {
    #     "topic": "quantum computing",
    #     "definition": "Quantum computing is..."
    # }
    """)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Example 2: Data Transformation Chain
    # --------------------------------------------------------------------------------------
    elif example == "Data Transformation Chain":
        st.info("Transform data at each step using custom functions.")
    
        text = st.text_area(
            "Enter text:",
            placeholder="The quick brown fox jumps over the lazy dog.",
            height=100
        )
    
        if st.button("🚀 Run Transformation Chain"):
            if not text.strip():
                st.warning("Please enter some text.")
                st.stop()
        
            try:
                # Define custom transformation functions
                def count_words(text: str) -> dict:
                    """Count words in text."""
                    words = text.split()
                    return {
                        "text": text,
                        "word_count": len(words),
                        "words": words
                    }
            
                def analyze_text(data: dict) -> dict:
                    """Analyze text characteristics."""
                    text = data["text"]
                    return {
                        **data,
                        "char_count": len(text),
                        "avg_word_length": sum(len(w) for w in data["words"]) / len(data["words"]) if data["words"] else 0
                    }
            
                def format_report(data: dict) -> str:
                    """Format analysis as a report."""
                    return f"""Text Analysis Report:
    - Characters: {data['char_count']}
    - Words: {data['word_count']}
    - Average word length: {data['avg_word_length']:.2f}"""
            
                with st.spinner("Running transformation chain..."):
                    # Create chain with custom transformations
                    chain = (
                        RunnableLambda(count_words) |
                        RunnableLambda(analyze_text) |
                        RunnableLambda(format_report)
                    )
                
                    result = chain.invoke(text)
            
                st.markdown("### Analysis Results:")
                st.text(result)
            
                with st.expander("🔍 How it works"):
                    st.code("""
    # RunnableLambda wraps custom functions
    def custom_function(input):
        # Process input
        return output

    chain = (
        RunnableLambda(function1) |
        RunnableLambda(function2) |
        RunnableLambda(function3)
    )

    # Data flows through each function
    result = chain.invoke(input_data)
    """)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Example 3: Multi-Step Analysis Chain
    # --------------------------------------------------------------------------------------
    else:  # Multi-Step Analysis Chain
        st.info("Complex chain that combines LLM calls with data transformations.")
    
        product = st.text_input("Product name:", placeholder="eco-friendly water bottle")
        target_audience = st.text_input("Target audience:", placeholder="fitness enthusiasts")
    
        if st.button("🚀 Run Analysis Chain"):
            if not product.strip() or not target_audience.strip():
                st.warning("Please fill in all fields.")
                st.stop()
        
            try:
                llm = ChatOllama(model=model_name, temperature=0.7)
                parser = StrOutputParser()
            
                # Step 1: Generate features
                with st.spinner("Step 1: Generating product features..."):
                    features_prompt = ChatPromptTemplate.from_template(
                        "List 5 key features of a {product} in bullet points."
                    )
                    features_chain = features_prompt | llm | parser
                    features = features_chain.invoke({"product": product})
                
                    st.markdown("**Step 1 Output:**")
                    st.write(features)
            
                # Step 2: Create marketing angles
                with st.spinner("Step 2: Creating marketing angles..."):
                    # Custom function to structure data
                    def combine_data(features_text):
                        return {
                            "product": product,
                            "audience": target_audience,
                            "features": features_text
                        }
                
                    marketing_prompt = ChatPromptTemplate.from_template(
                        """Based on these features of {product}, create 3 marketing angles 
    targeting {audience}:

    Features:
    {features}

    Format as numbered list."""
                    )
                
                    marketing_chain = (
                        RunnableLambda(combine_data) |
                        marketing_prompt |
                        llm |
                        parser
                    )
                
                    marketing_angles = marketing_chain.invoke(features)
                
                    st.markdown("**Step 2 Output:**")
                    st.write(marketing_angles)
            
                # Step 3: Generate tagline
                with st.spinner("Step 3: Creating tagline..."):
                    def prepare_tagline_input(marketing_text):
                        return {
                            "product": product,
                            "marketing": marketing_text
                        }
                
                    tagline_prompt = ChatPromptTemplate.from_template(
                        """Create a catchy tagline (max 10 words) for {product} based on:

    {marketing}

    Tagline:"""
                    )
                
                    tagline_chain = (
                        RunnableLambda(prepare_tagline_input) |
                        tagline_prompt |
                        llm |
                        parser
                    )
                
                    tagline = tagline_chain.invoke(marketing_angles)
                
                    st.markdown("**Step 3 Output:**")
                    st.info(f"💡 {tagline}")
            
                # Summary
                st.markdown("---")
                st.markdown("### 📋 Complete Marketing Package")
            
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Product:**")
                    st.write(product)
                    st.markdown("**Audience:**")
                    st.write(target_audience)
            
                with col2:
                    st.markdown("**Tagline:**")
                    st.info(tagline)
            
                with st.expander("🔍 Chain Architecture"):
                    st.code("""
    # Complex multi-step chain with custom functions

    # Step 1: Generate features
    features_chain = prompt1 | llm | parser

    # Step 2: Combine with custom data
    combine_chain = (
        RunnableLambda(combine_data) |
        prompt2 |
        llm |
        parser
    )

    # Step 3: Final processing
    final_chain = (
        RunnableLambda(prepare_input) |
        prompt3 |
        llm |
        parser
    )

    # Each step builds on previous outputs
    """)
        
            except Exception as exc:
                st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 What you learned"):
        st.markdown("""
        **Key Components:**
    
        - **RunnablePassthrough**: Pass data through unchanged
        - **RunnableLambda**: Wrap custom Python functions
        - **Dictionary Chains**: Return multiple outputs
    
        **Patterns:**
    
        ```python
        # Pattern 1: Passthrough with processing
        chain = {
            "original": RunnablePassthrough(),
            "processed": prompt | llm
        }
    
        # Pattern 2: Custom transformations
        chain = (
            RunnableLambda(transform1) |
            RunnableLambda(transform2)
        )
    
        # Pattern 3: Mixed LLM and custom functions
        chain = (
            prompt1 | llm |
            RunnableLambda(custom_function) |
            prompt2 | llm
        )
        ```
    
        **When to Use Custom Chains:**
        - Data preprocessing needed
        - Multiple outputs required
        - Complex business logic
        - Combining LLM calls with calculations
        - Building reusable components
        """)

    with st.expander("💡 Advanced Tips"):
        st.markdown("""
        **Performance:**
        - Cache expensive operations
        - Parallelize independent steps
        - Use async for better throughput
    
        **Debugging:**
        - Add logging in RunnableLambda functions
        - Test each component separately
        - Use .invoke() for synchronous debugging
    
        **Best Practices:**
        - Keep functions pure (no side effects)
        - Handle errors gracefully
        - Document expected input/output
        - Type hints for clarity
        """)

    with st.expander("🎯 Next Steps"):
        st.markdown("""
        Continue with advanced topics:
        - **02_Conversation_Buffer.py**: Advanced memory patterns
        - **04_Streaming_Chat.py**: Stream responses in chains
        - **10_Routing_Chain.py**: Dynamic chain routing
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
