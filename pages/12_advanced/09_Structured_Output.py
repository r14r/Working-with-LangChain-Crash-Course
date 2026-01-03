import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from lib.helper_streamlit import select_model
from lib.helper_streamlit.show_source import show_source

# --------------------------------------------------------------------------------------
# App: Structured Output with Validation
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Structured Output | Advanced",
    page_icon="📋",
    layout="centered",
)

st.title("📋 Structured Output & Validation")
st.caption("Extract and validate structured data with Pydantic models.")

# Create tabs
tab1, tab2 = st.tabs(["📱 App", "📄 Source Code"])

with tab1:

    st.markdown("""
    Use Pydantic models to define schemas with validation rules.
    Ensures AI outputs conform to expected structure and constraints.
    """)

    # Configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = select_model(key="struct_model", location="sidebar")

    # --------------------------------------------------------------------------------------
    # Define Pydantic Models
    # --------------------------------------------------------------------------------------

    class Person(BaseModel):
        """Model for person information."""
        name: str = Field(description="Full name")
        age: int = Field(description="Age in years", ge=0, le=150)
        email: Optional[str] = Field(default=None, description="Email address")
        occupation: str = Field(description="Job title or profession")
    
        @validator('age')
        def validate_age(cls, v):
            if v < 0 or v > 150:
                raise ValueError('Age must be between 0 and 150')
            return v

    class ProductReview(BaseModel):
        """Model for product reviews."""
        product_name: str = Field(description="Name of the product")
        rating: int = Field(description="Rating from 1 to 5", ge=1, le=5)
        sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")
        pros: List[str] = Field(description="List of pros")
        cons: List[str] = Field(description="List of cons")
        would_recommend: bool = Field(description="Would recommend to others")
    
        @validator('sentiment')
        def validate_sentiment(cls, v):
            allowed = ['positive', 'neutral', 'negative']
            if v.lower() not in allowed:
                raise ValueError(f'Sentiment must be one of: {allowed}')
            return v.lower()

    class MeetingNotes(BaseModel):
        """Model for meeting notes."""
        title: str = Field(description="Meeting title")
        date: str = Field(description="Date in YYYY-MM-DD format")
        attendees: List[str] = Field(description="List of attendees")
        key_points: List[str] = Field(description="Main discussion points")
        action_items: List[dict] = Field(description="Action items with assignee and deadline")
        next_meeting: Optional[str] = Field(default=None, description="Next meeting date")

    # --------------------------------------------------------------------------------------
    # Schema Selection
    # --------------------------------------------------------------------------------------
    st.subheader("1) Choose a Schema")

    schema_choice = st.radio(
        "Select a schema to use:",
        ["Person Info", "Product Review", "Meeting Notes"]
    )

    schema_map = {
        "Person Info": Person,
        "Product Review": ProductReview,
        "Meeting Notes": MeetingNotes
    }

    selected_schema = schema_map[schema_choice]

    # Show schema
    with st.expander("📝 View Schema"):
        st.json(selected_schema.schema())

    # --------------------------------------------------------------------------------------
    # Input Text
    # --------------------------------------------------------------------------------------
    st.subheader("2) Input Text to Extract")

    if schema_choice == "Person Info":
        default_text = "John Smith is a 35-year-old software engineer. His email is john.smith@example.com."
    elif schema_choice == "Product Review":
        default_text = """I bought the UltraPhone X and I'm really impressed. 
        The battery life is amazing and the camera quality is excellent. 
        However, it's quite expensive and the size is a bit too large for my hands. 
        Overall, I'd rate it 4 out of 5 stars and would recommend it to others."""
    else:  # Meeting Notes
        default_text = """Team sync meeting on 2024-01-15. 
        Attendees: Alice, Bob, Carol. 
        We discussed the Q1 roadmap and identified three priorities: improve performance, add new features, fix critical bugs. 
        Action items: Alice will create performance benchmarks by Friday, Bob will draft feature specifications by next Monday.
        Next meeting scheduled for 2024-01-22."""

    text_input = st.text_area(
        "Enter text:",
        value=default_text,
        height=150
    )

    # --------------------------------------------------------------------------------------
    # Extract and Validate
    # --------------------------------------------------------------------------------------
    if st.button("🚀 Extract & Validate"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            st.stop()
    
        try:
            with st.spinner("Extracting structured data..."):
                # Create parser
                parser = JsonOutputParser(pydantic_object=selected_schema)
                format_instructions = parser.get_format_instructions()
            
                # Create prompt
                prompt = ChatPromptTemplate.from_template(
                    """Extract structured information from the text below.

    Text:
    {text}

    {format_instructions}"""
                )
            
                # Create chain
                llm = ChatOllama(model=model_name, temperature=0)
                chain = prompt | llm | parser
            
                # Extract
                result = chain.invoke({
                    "text": text_input,
                    "format_instructions": format_instructions
                })
        
            st.success("✅ Successfully extracted and validated!")
        
            st.markdown("### 📊 Extracted Data:")
            st.json(result)
        
            # Validate against Pydantic model
            try:
                validated = selected_schema(**result)
                st.info("✅ Data passes all validation rules")
            
                # Show validated object
                with st.expander("🔍 Validated Object"):
                    st.write(validated)
        
            except Exception as validation_error:
                st.error(f"Validation failed: {validation_error}")
    
        except Exception as exc:
            st.error(f"Error: {exc}")

    # --------------------------------------------------------------------------------------
    # Learning Section
    # --------------------------------------------------------------------------------------
    with st.expander("📚 Pydantic Models & Validation"):
        st.markdown("""
        **Key Features:**
    
        - **Type Hints**: Enforce data types
        - **Field Constraints**: Min/max values, patterns
        - **Custom Validators**: Complex validation logic
        - **Optional Fields**: Nullable values
    
        **Example Model:**
        ```python
        from pydantic import BaseModel, Field, validator
    
        class Person(BaseModel):
            name: str = Field(description="Full name")
            age: int = Field(ge=0, le=150)
            email: Optional[str] = None
        
            @validator('age')
            def validate_age(cls, v):
                if v < 0 or v > 150:
                    raise ValueError('Invalid age')
                return v
        ```
    
        **Field Constraints:**
        - `ge`, `le`: Greater/less than or equal
        - `gt`, `lt`: Greater/less than
        - `min_length`, `max_length`: String/list length
        - `regex`: Pattern matching
    
        **Usage:**
        ```python
        parser = JsonOutputParser(pydantic_object=Person)
        instructions = parser.get_format_instructions()
    
        chain = prompt | llm | parser
        result = chain.invoke({"format_instructions": instructions})
    
        # Result is validated automatically
        ```
        """)

    with st.expander("💡 Benefits & Use Cases"):
        st.markdown("""
        **Benefits:**
        - ✅ Type safety
        - ✅ Automatic validation
        - ✅ Clear contracts
        - ✅ Better errors
        - ✅ IDE support
    
        **Use Cases:**
        - Data extraction from documents
        - Form filling from text
        - API response parsing
        - Database record creation
        - Structured logging
    
        **Best Practices:**
        - Define clear field descriptions
        - Use appropriate constraints
        - Handle optional fields
        - Provide default values
        - Document validation rules
        """)

    with st.expander("🎯 Advanced Patterns"):
        st.markdown("""
        **Nested Models:**
        ```python
        class Address(BaseModel):
            street: str
            city: str
    
        class Person(BaseModel):
            name: str
            address: Address
        ```
    
        **Lists and Enums:**
        ```python
        from enum import Enum
    
        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"
    
        class User(BaseModel):
            status: Status
            tags: List[str]
        ```
    
        **Custom Validators:**
        ```python
        @validator('email')
        def validate_email(cls, v):
            if '@' not in v:
                raise ValueError('Invalid email')
            return v.lower()
        ```
        """)

with tab2:
    st.markdown("### Source Code")
    show_source(__file__)
