import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from lib.helper_streamlit import select_model

# --------------------------------------------------------------------------------------
# App: Prompt Templates
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Prompt Templates | Beginner",
    page_icon="📝",
    layout="centered",
)

st.title("📝 Prompt Templates")
st.caption("Learn how to create reusable prompt templates in LangChain.")

st.markdown(
    """
Prompt templates help you create consistent, reusable prompts. Instead of 
writing the same prompt structure repeatedly, you define it once with 
variables that can be filled in.

**What you'll learn:**
1. Create a **ChatPromptTemplate**  
2. Use **variables** in your prompts  
3. Format prompts with different inputs
"""
)

# --------------------------------------------------------------------------------------
# Model Selection
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    model_name = select_model(
        key="template_model",
        location="sidebar",
        label="Choose your model"
    )

# --------------------------------------------------------------------------------------
# Template Examples
# --------------------------------------------------------------------------------------
st.subheader("1) Choose a Template")

template_choice = st.radio(
    "Select a prompt template to try:",
    [
        "Simple Question",
        "Translation",
        "Code Explanation",
        "Creative Writing"
    ]
)

# Define different templates
templates = {
    "Simple Question": ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer this question clearly and concisely:\n\n{question}"
    ),
    "Translation": ChatPromptTemplate.from_template(
        "Translate the following text from {source_language} to {target_language}:\n\n{text}"
    ),
    "Code Explanation": ChatPromptTemplate.from_template(
        "Explain this {language} code in simple terms:\n\n```{language}\n{code}\n```"
    ),
    "Creative Writing": ChatPromptTemplate.from_template(
        "Write a short {genre} story about {topic}. Keep it to {length} sentences."
    )
}

# --------------------------------------------------------------------------------------
# Input Fields Based on Template
# --------------------------------------------------------------------------------------
st.subheader("2) Fill in the Template")

if template_choice == "Simple Question":
    question = st.text_input("Question:", placeholder="What is machine learning?")
    inputs = {"question": question}
    
elif template_choice == "Translation":
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.text_input("From:", value="English")
    with col2:
        target_lang = st.text_input("To:", value="Spanish")
    text = st.text_area("Text:", placeholder="Hello, how are you?")
    inputs = {"source_language": source_lang, "target_language": target_lang, "text": text}
    
elif template_choice == "Code Explanation":
    language = st.selectbox("Language:", ["Python", "JavaScript", "Java", "Go"])
    code = st.text_area("Code:", placeholder='def hello():\n    print("Hello, World!")')
    inputs = {"language": language, "code": code}
    
else:  # Creative Writing
    genre = st.selectbox("Genre:", ["science fiction", "mystery", "romance", "horror"])
    topic = st.text_input("Topic:", placeholder="a time traveler")
    length = st.slider("Length (sentences):", 3, 10, 5)
    inputs = {"genre": genre, "topic": topic, "length": length}

# --------------------------------------------------------------------------------------
# Generate Response
# --------------------------------------------------------------------------------------
if st.button("🚀 Generate Response"):
    # Check if all inputs are filled
    if any(not str(v).strip() for v in inputs.values()):
        st.warning("Please fill in all fields.")
        st.stop()
    
    try:
        with st.spinner("Generating..."):
            # Get the selected template
            template = templates[template_choice]
            
            # Format the prompt with inputs
            formatted_prompt = template.format_messages(**inputs)
            
            # Show the formatted prompt
            with st.expander("📋 View Formatted Prompt"):
                st.code(formatted_prompt[0].content)
            
            # Initialize model and get response
            llm = ChatOllama(model=model_name, temperature=0.7)
            response = llm.invoke(formatted_prompt)
            
        st.markdown("### Response:")
        st.write(response.content)
        
    except Exception as exc:
        st.error(f"Error: {exc}")

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **ChatPromptTemplate**: Create reusable prompt structures
    - **Variables**: Use {variable_name} placeholders in prompts
    - **format_messages()**: Fill in the template with actual values
    
    **Benefits:**
    - Consistency across multiple requests
    - Easy to test and modify prompts
    - Cleaner, more maintainable code
    
    **Template Structure:**
    ```python
    template = ChatPromptTemplate.from_template(
        "Your prompt with {variable1} and {variable2}"
    )
    formatted = template.format_messages(
        variable1="value1",
        variable2="value2"
    )
    ```
    """)

with st.expander("💡 Next Steps"):
    st.markdown("""
    Continue your learning:
    - **03_Output_Parser.py**: Parse structured output from the AI
    - **04_Simple_Chain.py**: Combine templates with LLMs in chains
    - **05_Chat_History.py**: Add conversation memory
    """)
