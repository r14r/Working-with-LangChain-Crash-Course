import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Routing Chain
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Routing Chain | Advanced",
    page_icon="🔀",
    layout="centered",
)

st.title("🔀 Dynamic Chain Routing")
st.caption("Route inputs to different chains based on content or classification.")

st.markdown("""
Routing chains direct inputs to specialized chains based on:
- Content classification
- Intent detection
- Topic categorization
- Complexity assessment

**What you'll learn:**
1. **RunnableBranch**: Conditional routing
2. **Dynamic chain selection**: Choose chains at runtime
3. **Multi-specialist architecture**: Different chains for different tasks
""")

# Configuration
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(key="routing_model", location="sidebar")

# --------------------------------------------------------------------------------------
# Define Specialized Chains
# --------------------------------------------------------------------------------------

def create_math_chain(model):
    """Chain specialized for math problems."""
    prompt = ChatPromptTemplate.from_template(
        """You are a math tutor. Solve this math problem step by step.

Problem: {input}

Solution:"""
    )
    llm = ChatOllama(model=model, temperature=0)
    return prompt | llm | StrOutputParser()

def create_creative_chain(model):
    """Chain specialized for creative writing."""
    prompt = ChatPromptTemplate.from_template(
        """You are a creative writer. Respond imaginatively and artistically.

Request: {input}

Creative response:"""
    )
    llm = ChatOllama(model=model, temperature=0.9)
    return prompt | llm | StrOutputParser()

def create_factual_chain(model):
    """Chain specialized for factual questions."""
    prompt = ChatPromptTemplate.from_template(
        """You are an encyclopedia. Provide factual, accurate information.

Question: {input}

Factual answer:"""
    )
    llm = ChatOllama(model=model, temperature=0.1)
    return prompt | llm | StrOutputParser()

def create_code_chain(model):
    """Chain specialized for coding questions."""
    prompt = ChatPromptTemplate.from_template(
        """You are a programming expert. Provide clear code examples and explanations.

Question: {input}

Code solution:"""
    )
    llm = ChatOllama(model=model, temperature=0.3)
    return prompt | llm | StrOutputParser()

# --------------------------------------------------------------------------------------
# Classification Function
# --------------------------------------------------------------------------------------

def classify_input(input_text: str, model: str) -> str:
    """Classify the input to determine which chain to use."""
    classifier_prompt = ChatPromptTemplate.from_template(
        """Classify this input into ONE category:
- math: mathematical problems or calculations
- creative: creative writing, stories, poems
- factual: factual questions, general knowledge
- code: programming, coding questions

Input: {input}

Category (respond with just one word):"""
    )
    
    llm = ChatOllama(model=model, temperature=0)
    chain = classifier_prompt | llm | StrOutputParser()
    
    category = chain.invoke({"input": input_text}).strip().lower()
    
    # Map to valid categories
    if "math" in category:
        return "math"
    elif "creative" in category:
        return "creative"
    elif "code" in category or "programming" in category:
        return "code"
    else:
        return "factual"

# --------------------------------------------------------------------------------------
# Routing Example Selection
# --------------------------------------------------------------------------------------
st.subheader("Choose Routing Method")

routing_method = st.radio(
    "Select routing approach:",
    ["Automatic Classification", "Manual Selection", "Branch-based Routing"]
)

# --------------------------------------------------------------------------------------
# Method 1: Automatic Classification
# --------------------------------------------------------------------------------------
if routing_method == "Automatic Classification":
    st.info("The system will automatically detect the input type and route to the appropriate chain.")
    
    user_input = st.text_area(
        "Enter your request:",
        placeholder="Ask anything...",
        height=100
    )
    
    if st.button("🚀 Process"):
        if not user_input.strip():
            st.warning("Please enter something.")
            st.stop()
        
        try:
            with st.spinner("Classifying input..."):
                category = classify_input(user_input, model_name)
            
            st.info(f"📂 Detected category: **{category.upper()}**")
            
            with st.spinner(f"Processing with {category} chain..."):
                # Route to appropriate chain
                if category == "math":
                    chain = create_math_chain(model_name)
                elif category == "creative":
                    chain = create_creative_chain(model_name)
                elif category == "code":
                    chain = create_code_chain(model_name)
                else:
                    chain = create_factual_chain(model_name)
                
                result = chain.invoke({"input": user_input})
            
            st.markdown("### 💬 Response:")
            st.write(result)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Method 2: Manual Selection
# --------------------------------------------------------------------------------------
elif routing_method == "Manual Selection":
    st.info("Manually choose which specialized chain to use.")
    
    chain_type = st.selectbox(
        "Select chain:",
        ["Math", "Creative", "Factual", "Code"]
    )
    
    user_input = st.text_area(
        "Enter your request:",
        placeholder="Enter text...",
        height=100
    )
    
    if st.button("🚀 Process"):
        if not user_input.strip():
            st.warning("Please enter something.")
            st.stop()
        
        try:
            with st.spinner(f"Processing with {chain_type} chain..."):
                if chain_type == "Math":
                    chain = create_math_chain(model_name)
                elif chain_type == "Creative":
                    chain = create_creative_chain(model_name)
                elif chain_type == "Code":
                    chain = create_code_chain(model_name)
                else:
                    chain = create_factual_chain(model_name)
                
                result = chain.invoke({"input": user_input})
            
            st.markdown("### 💬 Response:")
            st.write(result)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Method 3: Branch-based Routing
# --------------------------------------------------------------------------------------
else:  # Branch-based Routing
    st.info("Uses RunnableBranch for declarative routing logic.")
    
    user_input = st.text_area(
        "Enter your request:",
        placeholder="Ask anything...",
        height=100
    )
    
    if st.button("🚀 Process with Branch"):
        if not user_input.strip():
            st.warning("Please enter something.")
            st.stop()
        
        try:
            with st.spinner("Processing..."):
                # Classify once to avoid redundant calls
                category = classify_input(user_input, model_name)
                st.info(f"📂 Category: {category}")
                
                # Create branches with cached classification
                math_branch = (
                    lambda x: category == "math",
                    create_math_chain(model_name)
                )
                
                creative_branch = (
                    lambda x: category == "creative",
                    create_creative_chain(model_name)
                )
                
                code_branch = (
                    lambda x: category == "code",
                    create_code_chain(model_name)
                )
                
                # Default branch
                default_chain = create_factual_chain(model_name)
                
                # Create branching chain
                branch = RunnableBranch(
                    math_branch,
                    creative_branch,
                    code_branch,
                    default_chain
                )
                
                result = branch.invoke({"input": user_input})
            
            st.markdown("### 💬 Response:")
            st.write(result)
        
        except Exception as exc:
            st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 How Routing Works"):
    st.markdown("""
    **Routing Pattern:**
    
    ```python
    # 1. Classify input
    category = classify(input_text)
    
    # 2. Select chain
    if category == "math":
        chain = math_chain
    elif category == "creative":
        chain = creative_chain
    else:
        chain = default_chain
    
    # 3. Execute
    result = chain.invoke(input_text)
    ```
    
    **RunnableBranch:**
    ```python
    from langchain_core.runnables import RunnableBranch
    
    branch = RunnableBranch(
        (condition1, chain1),
        (condition2, chain2),
        default_chain
    )
    
    result = branch.invoke(input)
    ```
    
    **Benefits:**
    - Specialized handling
    - Better performance
    - Clearer logic
    - Easier testing
    """)

with st.expander("💡 Use Cases"):
    st.markdown("""
    **Customer Support:**
    - Technical → Technical support chain
    - Billing → Billing chain
    - General → FAQ chain
    
    **Content Processing:**
    - Code → Code analyzer
    - Natural language → NLP chain
    - Structured data → Data extractor
    
    **Multi-language:**
    - English → English chain
    - Spanish → Spanish chain
    - Auto-detect → Classify first
    
    **Complexity-based:**
    - Simple → Fast chain
    - Complex → Powerful chain
    - Unknown → Classification chain
    """)

with st.expander("🎯 Advanced Patterns"):
    st.markdown("""
    **Cascading Routing:**
    - First: Detect language
    - Then: Route by topic
    - Finally: Route by complexity
    
    **Confidence-based:**
    - High confidence → Specialized chain
    - Low confidence → General chain
    - Very low → Human escalation
    
    **Load Balancing:**
    - Route based on system load
    - Use faster models when busy
    - Queue complex requests
    
    **A/B Testing:**
    - Route % to chain A
    - Route % to chain B
    - Compare results
    """)
