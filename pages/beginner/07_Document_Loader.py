import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)

# --------------------------------------------------------------------------------------
# App: Document Loaders
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Loaders | Beginner",
    page_icon="📁",
    layout="centered",
)

st.title("📁 Document Loaders")
st.caption("Learn how to load various document formats into LangChain.")

st.markdown(
    """
Document loaders help you import content from different file formats into 
LangChain. Each loader is specialized for a specific file type.

**What you'll learn:**
1. Load text files, PDFs, and Markdown  
2. Understand document structure  
3. Access document metadata
"""
)

# --------------------------------------------------------------------------------------
# File Upload
# --------------------------------------------------------------------------------------
st.subheader("1) Upload a Document")

file_type = st.radio(
    "Select file type to upload:",
    ["Text (.txt)", "PDF (.pdf)", "Markdown (.md)"]
)

if file_type == "Text (.txt)":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    loader_class = TextLoader
elif file_type == "PDF (.pdf)":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    loader_class = PyPDFLoader
else:  # Markdown
    uploaded_file = st.file_uploader("Upload a Markdown file", type=["md"])
    loader_class = UnstructuredMarkdownLoader

# --------------------------------------------------------------------------------------
# Process Document
# --------------------------------------------------------------------------------------
if uploaded_file is not None:
    st.subheader("2) Document Content")
    
    try:
        with st.spinner("Loading document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load document
            loader = loader_class(tmp_path)
            documents = loader.load()
            
        st.success(f"✅ Loaded {len(documents)} document(s)")
        
        # Display documents
        for i, doc in enumerate(documents):
            with st.expander(f"📄 Document {i+1}" + (f" (Page {doc.metadata.get('page', 'N/A')})" if 'page' in doc.metadata else "")):
                
                # Content
                st.markdown("**Content:**")
                st.text_area(
                    "Document text",
                    value=doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""),
                    height=200,
                    key=f"content_{i}",
                    label_visibility="collapsed"
                )
                
                if len(doc.page_content) > 1000:
                    st.info(f"Showing first 1000 of {len(doc.page_content)} characters")
                
                # Metadata
                st.markdown("**Metadata:**")
                st.json(doc.metadata)
                
                # Statistics
                st.markdown("**Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(doc.page_content))
                with col2:
                    st.metric("Words", len(doc.page_content.split()))
                with col3:
                    st.metric("Lines", len(doc.page_content.split('\n')))
        
        # Summary
        st.subheader("3) Document Summary")
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages/Docs", len(documents))
        with col2:
            st.metric("Total Characters", f"{total_chars:,}")
        with col3:
            st.metric("Total Words", f"{total_words:,}")
        
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    except Exception as exc:
        st.error(f"Error loading document: {exc}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("👆 Upload a file to get started")
    
    # Show example
    with st.expander("📝 Create Sample Document"):
        sample_type = st.selectbox(
            "Sample type:",
            ["Text", "Markdown"]
        )
        
        if sample_type == "Text":
            sample_content = """Welcome to LangChain!

This is a sample text document that demonstrates document loading.

LangChain provides various document loaders for different file formats:
- TextLoader for plain text files
- PyPDFLoader for PDF documents
- UnstructuredMarkdownLoader for Markdown files
- And many more!

Each loader handles the specifics of parsing its file format and returns
documents in a standard format that LangChain can work with.

Try uploading your own file to see how it works!"""
        else:  # Markdown
            sample_content = """# Welcome to LangChain

## Document Loaders

This is a sample **Markdown** document.

### Key Features

- Easy to use
- Supports multiple formats
- Extracts metadata
- Works with `code blocks`

### Example Code

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("file.txt")
documents = loader.load()
```

Try uploading your own Markdown file!"""
        
        st.download_button(
            label=f"📥 Download Sample {sample_type}",
            data=sample_content,
            file_name=f"sample.{'txt' if sample_type == 'Text' else 'md'}",
            mime="text/plain"
        )

# --------------------------------------------------------------------------------------
# Learning Section
# --------------------------------------------------------------------------------------
with st.expander("📚 What you learned"):
    st.markdown("""
    **Key Concepts:**
    
    - **Document Loaders**: Import files into LangChain
    - **Document Object**: Contains `page_content` and `metadata`
    - **Metadata**: Additional information (page number, source, etc.)
    
    **How It Works:**
    ```python
    from langchain_community.document_loaders import TextLoader
    
    # Load a text file
    loader = TextLoader("document.txt")
    documents = loader.load()
    
    # Access content and metadata
    for doc in documents:
        print(doc.page_content)  # The text
        print(doc.metadata)      # File info
    ```
    
    **Common Loaders:**
    
    | Loader | File Type | Use Case |
    |--------|-----------|----------|
    | `TextLoader` | .txt | Plain text files |
    | `PyPDFLoader` | .pdf | PDF documents |
    | `UnstructuredMarkdownLoader` | .md | Markdown files |
    | `CSVLoader` | .csv | Tabular data |
    | `JSONLoader` | .json | JSON data |
    | `WebBaseLoader` | URL | Web pages |
    
    **Document Structure:**
    ```python
    document = {
        'page_content': 'The actual text...',
        'metadata': {
            'source': 'file.pdf',
            'page': 1,
            ...
        }
    }
    ```
    """)

with st.expander("💡 Next Steps"):
    st.markdown("""
    Now that you can load documents:
    - **08_Text_Splitter.py**: Split large documents into chunks
    - **09_Vector_Store.py**: Store documents for retrieval
    - **10_Simple_RAG.py**: Build a Q&A system
    
    **Advanced Loaders:**
    - **Word documents**: `Docx2txtLoader`
    - **HTML**: `UnstructuredHTMLLoader`
    - **URLs**: `WebBaseLoader`
    - **Directories**: `DirectoryLoader`
    """)

with st.expander("🔧 Tips & Best Practices"):
    st.markdown("""
    **Performance Tips:**
    - Large PDFs: Use page-by-page processing
    - Multiple files: Use `DirectoryLoader`
    - Web content: Cache results
    
    **Common Issues:**
    - **PDF text extraction**: Some PDFs are scanned images (use OCR)
    - **Encoding errors**: Specify encoding for text files
    - **Large files**: Split into smaller chunks
    
    **Metadata Usage:**
    - Track document source
    - Filter by page number
    - Sort by date
    - Group by category
    """)
