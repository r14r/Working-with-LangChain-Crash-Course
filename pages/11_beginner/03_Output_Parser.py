import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Output Parsers
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Output Parsers | Beginner",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Output Parsers")
st.caption("Learn how to parse and structure AI responses in LangChain.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown(
        """
    Output parsers help you get structured data from AI responses. Instead of 
    just getting raw text, you can parse responses into specific formats like 
    JSON, lists, or custom objects.

    **What you'll learn:**
    1. **StrOutputParser**: Get clean string output  
    2. **JsonOutputParser**: Parse responses as JSON  
    3. **Pydantic Models**: Define structured schemas
    """
    )

    # --------------------------------------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(
            key="parser_model",
            location="sidebar",
            label="Choose your model"
        )

    # --------------------------------------------------------------------------------------
    # Pydantic Models for JSON Parsing
    # --------------------------------------------------------------------------------------
    class MovieReview(BaseModel):
        """Schema for a movie review."""
        title: str = Field(description="Movie title")
        rating: int = Field(description="Rating from 1 to 10")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")
        summary: str = Field(description="One-sentence summary")

    class Recipe(BaseModel):
        """Schema for a recipe."""
        name: str = Field(description="Recipe name")
        cuisine: str = Field(description="Type of cuisine")
        ingredients: List[str] = Field(description="List of ingredients")
        steps: List[str] = Field(description="Cooking steps")
        prep_time: str = Field(description="Preparation time")

    # --------------------------------------------------------------------------------------
    # Parser Selection
    # --------------------------------------------------------------------------------------
    st.subheader("1) Choose a Parser Type")

    parser_type = st.radio(
        "Select the type of output parsing:",
        ["String Parser", "JSON Parser - Movie Review", "JSON Parser - Recipe"]
    )

    # --------------------------------------------------------------------------------------
    # Input Based on Parser Type
    # --------------------------------------------------------------------------------------
    st.subheader("2) Enter Your Request")

    if parser_type == "String Parser":
        st.info("String parser returns clean text output.")
        user_input = st.text_area(
            "Ask anything:",
            placeholder="Write a haiku about programming",
            height=100
        )
    
    elif parser_type == "JSON Parser - Movie Review":
        st.info("This will return a structured movie review as JSON.")
        movie_name = st.text_input(
            "Movie name:",
            placeholder="The Matrix"
        )
        user_input = f"Write a detailed review of the movie '{movie_name}'"
    
    else:  # Recipe
        st.info("This will return a structured recipe as JSON.")
        dish_name = st.text_input(
            "Dish name:",
            placeholder="Spaghetti Carbonara"
        )
        user_input = f"Provide a recipe for {dish_name}"

    # --------------------------------------------------------------------------------------
    # Generate and Parse Response
    # --------------------------------------------------------------------------------------
    if st.button("🚀 Generate and Parse"):
        if not user_input or not user_input.strip():
            st.warning("Please provide input.")
            st.stop()
    
        try:
            with st.spinner("Processing..."):
                llm = ChatOllama(model=model_name, temperature=0.7)
            
                if parser_type == "String Parser":
                    # Simple string parser
                    parser = StrOutputParser()
                    template = ChatPromptTemplate.from_template("{input}")
                    chain = template | llm | parser
                    result = chain.invoke({"input": user_input})
                
                    st.markdown("### Parsed Output (String):")
                    st.write(result)
                
                elif parser_type == "JSON Parser - Movie Review":
                    # JSON parser with MovieReview schema
                    parser = JsonOutputParser(pydantic_object=MovieReview)
                    format_instructions = parser.get_format_instructions()
                
                    template = ChatPromptTemplate.from_template(
                        "{input}\n\n{format_instructions}"
                    )
                
                    chain = template | llm | parser
                    result = chain.invoke({
                        "input": user_input,
                        "format_instructions": format_instructions
                    })
                
                    st.markdown("### Parsed Output (JSON):")
                    st.json(result)
                
                    # Show structured access
                    with st.expander("📊 Accessing Structured Data"):
                        st.write(f"**Title:** {result['title']}")
                        st.write(f"**Rating:** {result['rating']}/10")
                        st.write(f"**Pros:** {', '.join(result['pros'])}")
                        st.write(f"**Cons:** {', '.join(result['cons'])}")
                        st.write(f"**Summary:** {result['summary']}")
                
                else:  # Recipe
                    # JSON parser with Recipe schema
                    parser = JsonOutputParser(pydantic_object=Recipe)
                    format_instructions = parser.get_format_instructions()
                
                    template = ChatPromptTemplate.from_template(
                        "{input}\n\n{format_instructions}"
                    )
                
                    chain = template | llm | parser
                    result = chain.invoke({
                        "input": user_input,
                        "format_instructions": format_instructions
                    })
                
                    st.markdown("### Parsed Output (JSON):")
                    st.json(result)
                
                    # Show structured display
                    with st.expander("🍳 Recipe Details"):
                        st.write(f"**Dish:** {result['name']}")
                        st.write(f"**Cuisine:** {result['cuisine']}")
                        st.write(f"**Prep Time:** {result['prep_time']}")
                    
                        st.markdown("**Ingredients:**")
                        for ingredient in result['ingredients']:
                            st.write(f"- {ingredient}")
                    
                        st.markdown("**Steps:**")
                        for i, step in enumerate(result['steps'], 1):
                            st.write(f"{i}. {step}")
        
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("If JSON parsing fails, try a different model or adjust your request.")

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 What you learned"):
        st.markdown("""
        **Key Concepts:**
    
        - **StrOutputParser**: Extracts clean string content from AI responses
        - **JsonOutputParser**: Parses responses into JSON format
        - **Pydantic Models**: Define schemas for structured data
        - **Format Instructions**: Tell the AI how to structure its output
    
        **Why Use Parsers:**
        - Get predictable, structured output
        - Easy to process and validate data
        - Better integration with your application
        - Type safety and validation
    
        **Code Pattern:**
        ```python
        # Define schema
        class MyData(BaseModel):
            field1: str
            field2: int
    
        # Create parser
        parser = JsonOutputParser(pydantic_object=MyData)
        instructions = parser.get_format_instructions()
    
        # Use in chain
        chain = template | llm | parser
        result = chain.invoke({"format_instructions": instructions})
        ```
        """)

    with st.expander("💡 Next Steps"):
        st.markdown("""
        Continue learning:
        - **04_Simple_Chain.py**: Combine prompts, LLMs, and parsers
        - **05_Chat_History.py**: Add conversation memory
        - **09_Structured_Output.py** (Advanced): Complex data structures
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
