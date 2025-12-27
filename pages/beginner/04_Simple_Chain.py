import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Simple Chains
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Simple Chains | Beginner",
    page_icon="🔗",
    layout="centered",
)

st.title("🔗 Simple LangChain Chains")
st.caption("Learn how to chain together prompts, LLMs, and parsers.")

st.markdown(
    """
Chains are the core concept in LangChain! A chain connects multiple components 
together to create a complete workflow. The most basic chain is:

**Prompt Template → LLM → Output Parser**

This is called the LangChain Expression Language (LCEL) pattern.

**What you'll learn:**
1. Create a basic chain with the `|` operator  
2. Chain multiple steps together  
3. Build a multi-step workflow
"""
)

# --------------------------------------------------------------------------------------
# Model Selection
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(
        key="chain_model",
        location="sidebar",
        label="Choose your model"
    )

# --------------------------------------------------------------------------------------
# Example Selection
# --------------------------------------------------------------------------------------
st.subheader("1) Choose a Chain Example")

chain_example = st.radio(
    "Select a chain to try:",
    [
        "Single Step Chain",
        "Two Step Chain",
        "Three Step Chain"
    ]
)

# --------------------------------------------------------------------------------------
# Single Step Chain
# --------------------------------------------------------------------------------------
if chain_example == "Single Step Chain":
    st.info("A simple chain: Prompt → LLM → Parser")
    
    topic = st.text_input("Topic:", placeholder="artificial intelligence")
    
    if st.button("🚀 Run Chain"):
        if not topic.strip():
            st.warning("Please enter a topic.")
            st.stop()
        
        try:
            with st.spinner("Running chain..."):
                # Step 1: Create prompt template
                prompt = ChatPromptTemplate.from_template(
                    "Explain {topic} in one paragraph for a beginner."
                )
                
                # Step 2: Create LLM
                llm = ChatOllama(model=model_name, temperature=0.7)
                
                # Step 3: Create parser
                parser = StrOutputParser()
                
                # Chain them together with | operator
                chain = prompt | llm | parser
                
                # Run the chain
                result = chain.invoke({"topic": topic})
            
            st.markdown("### Output:")
            st.write(result)
            
            with st.expander("🔍 How it works"):
                st.markdown("""
                ```python
                # Each component flows into the next
                chain = prompt | llm | parser
                
                # 1. prompt creates the message
                # 2. llm generates response
                # 3. parser extracts the text
                ```
                """)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Two Step Chain
# --------------------------------------------------------------------------------------
elif chain_example == "Two Step Chain":
    st.info("Chain with two sequential steps: Generate → Improve")
    
    st.markdown("**Step 1:** Generate a story opening")
    st.markdown("**Step 2:** Improve the writing")
    
    story_topic = st.text_input("Story topic:", placeholder="a robot learning to feel emotions")
    
    if st.button("🚀 Run Two-Step Chain"):
        if not story_topic.strip():
            st.warning("Please enter a story topic.")
            st.stop()
        
        try:
            llm = ChatOllama(model=model_name, temperature=0.7)
            parser = StrOutputParser()
            
            with st.spinner("Step 1: Generating story opening..."):
                # Chain 1: Generate initial story
                prompt1 = ChatPromptTemplate.from_template(
                    "Write a 3-sentence story opening about: {topic}"
                )
                chain1 = prompt1 | llm | parser
                story_draft = chain1.invoke({"topic": story_topic})
                
                st.markdown("#### Step 1 Output (Draft):")
                st.write(story_draft)
            
            with st.spinner("Step 2: Improving the writing..."):
                # Chain 2: Improve the story
                prompt2 = ChatPromptTemplate.from_template(
                    "Improve this story opening by making it more engaging and descriptive:\n\n{draft}"
                )
                chain2 = prompt2 | llm | parser
                story_improved = chain2.invoke({"draft": story_draft})
                
                st.markdown("#### Step 2 Output (Improved):")
                st.write(story_improved)
            
            with st.expander("🔍 How it works"):
                st.markdown("""
                ```python
                # Chain 1: Generate draft
                chain1 = prompt1 | llm | parser
                draft = chain1.invoke({"topic": topic})
                
                # Chain 2: Improve draft (uses output from chain1)
                chain2 = prompt2 | llm | parser
                improved = chain2.invoke({"draft": draft})
                ```
                
                **Note:** This runs two separate chains sequentially, 
                where the output of the first becomes input to the second.
                """)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Three Step Chain
# --------------------------------------------------------------------------------------
else:  # Three Step Chain
    st.info("Complex chain: Generate → Analyze → Summarize")
    
    st.markdown("**Step 1:** Generate a product description")
    st.markdown("**Step 2:** Analyze its marketing appeal")
    st.markdown("**Step 3:** Create a summary tweet")
    
    product = st.text_input("Product:", placeholder="eco-friendly water bottle")
    
    if st.button("🚀 Run Three-Step Chain"):
        if not product.strip():
            st.warning("Please enter a product.")
            st.stop()
        
        try:
            llm = ChatOllama(model=model_name, temperature=0.7)
            parser = StrOutputParser()
            
            # Step 1: Generate description
            with st.spinner("Step 1: Generating product description..."):
                prompt1 = ChatPromptTemplate.from_template(
                    "Write a compelling 2-paragraph product description for: {product}"
                )
                chain1 = prompt1 | llm | parser
                description = chain1.invoke({"product": product})
                
                st.markdown("#### Step 1: Product Description")
                st.write(description)
            
            # Step 2: Analyze appeal
            with st.spinner("Step 2: Analyzing marketing appeal..."):
                prompt2 = ChatPromptTemplate.from_template(
                    "Analyze the marketing appeal of this product description. "
                    "List 3 strong points and 2 areas for improvement:\n\n{description}"
                )
                chain2 = prompt2 | llm | parser
                analysis = chain2.invoke({"description": description})
                
                st.markdown("#### Step 2: Marketing Analysis")
                st.write(analysis)
            
            # Step 3: Create summary tweet
            with st.spinner("Step 3: Creating summary tweet..."):
                prompt3 = ChatPromptTemplate.from_template(
                    "Based on this product description and analysis, "
                    "write a compelling tweet (max 280 characters) to promote this product.\n\n"
                    "Description: {description}\n\n"
                    "Analysis: {analysis}"
                )
                chain3 = prompt3 | llm | parser
                tweet = chain3.invoke({
                    "description": description,
                    "analysis": analysis
                })
                
                st.markdown("#### Step 3: Summary Tweet")
                st.info(tweet)
            
            with st.expander("🔍 How it works"):
                st.markdown("""
                ```python
                # Each step builds on previous outputs
                
                # Step 1
                chain1 = prompt1 | llm | parser
                output1 = chain1.invoke({"product": product})
                
                # Step 2 (uses output1)
                chain2 = prompt2 | llm | parser
                output2 = chain2.invoke({"description": output1})
                
                # Step 3 (uses output1 and output2)
                chain3 = prompt3 | llm | parser
                output3 = chain3.invoke({
                    "description": output1,
                    "analysis": output2
                })
                ```
                
                This creates a pipeline where each step processes and 
                transforms data, passing results to the next step.
                """)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **LCEL (LangChain Expression Language)**: Use `|` to chain components
    - **Sequential Processing**: Each step feeds into the next
    - **Reusable Components**: Create once, use in multiple chains
    
    **Chain Structure:**
    ```python
    chain = prompt_template | llm | output_parser
    result = chain.invoke(input_data)
    ```
    
    **Why Use Chains:**
    - Break complex tasks into simple steps
    - Reusable and testable components
    - Clear data flow
    - Easy to modify and extend
    
    **Best Practices:**
    - Keep each step focused on one task
    - Pass only necessary data between steps
    - Handle errors at each step
    - Test chains incrementally
    """)

with st.expander("💡 Next Steps"):
    st.markdown("""
    Continue learning:
    - **05_Chat_History.py**: Add memory to chains
    - **10_Simple_RAG.py**: Combine retrieval with chains
    - **Advanced/01_Custom_Chain.py**: Build complex custom chains
    """)
