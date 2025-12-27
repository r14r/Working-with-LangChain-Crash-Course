import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lib.helper_streamlit import select_model

st.set_page_config(page_title="Self-RAG | Expert", page_icon="🔄", layout="centered")
st.title("🔄 Self-Correcting RAG")
st.caption("RAG system that evaluates and improves its own outputs.")

st.markdown("""
Self-RAG adds self-reflection:
1. Generate initial answer
2. Evaluate quality/relevance
3. Decide if retrieval/regeneration needed
4. Iteratively improve
""")

with st.sidebar:
    st.header("Configuration")
    model_name = select_model(key="self_rag_model", location="sidebar")
    embed_model = select_model(key="self_rag_embed", location="sidebar", default_models=["nomic-embed-text"])

# Load knowledge base
if "self_rag_kb" not in st.session_state:
    st.session_state.self_rag_kb = None

if st.button("📚 Load Knowledge Base"):
    docs = [
        Document(page_content="LangChain is a framework for LLM applications."),
        Document(page_content="RAG combines retrieval with generation."),
        Document(page_content="Self-RAG adds self-evaluation to RAG."),
    ]
    embeddings = OllamaEmbeddings(model=embed_model)
    st.session_state.self_rag_kb = DocArrayInMemorySearch.from_documents(docs, embeddings)
    st.success("Knowledge base ready!")

if st.session_state.self_rag_kb:
    question = st.text_input("Question:", placeholder="What is Self-RAG?")
    
    if st.button("🔍 Answer with Self-RAG"):
        if not question:
            st.warning("Enter a question.")
            st.stop()
        
        try:
            llm = ChatOllama(model=model_name, temperature=0)
            
            # Step 1: Initial retrieval
            with st.spinner("Step 1: Retrieving..."):
                docs = st.session_state.self_rag_kb.similarity_search(question, k=2)
                context = "\n".join([d.page_content for d in docs])
                st.info(f"Retrieved: {len(docs)} documents")
            
            # Step 2: Generate answer
            with st.spinner("Step 2: Generating answer..."):
                answer_prompt = ChatPromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                )
                chain = answer_prompt | llm | StrOutputParser()
                initial_answer = chain.invoke({"context": context, "question": question})
                
                st.markdown("### Initial Answer:")
                st.write(initial_answer)
            
            # Step 3: Self-evaluate
            with st.spinner("Step 3: Evaluating answer..."):
                eval_prompt = ChatPromptTemplate.from_template(
                    """Rate this answer's quality and relevance (1-10):
Question: {question}
Answer: {answer}

Respond with just a number."""
                )
                eval_chain = eval_prompt | llm | StrOutputParser()
                score_str = eval_chain.invoke({"question": question, "answer": initial_answer})
                
                try:
                    score = int(score_str.strip())
                except (ValueError, AttributeError):
                    score = 5
                
                st.metric("Quality Score", f"{score}/10")
            
            # Step 4: Refine if needed
            if score < 7:
                with st.spinner("Step 4: Refining answer..."):
                    refine_prompt = ChatPromptTemplate.from_template(
                        """Improve this answer to better address the question:
Question: {question}
Current: {answer}

Improved answer:"""
                    )
                    refine_chain = refine_prompt | llm | StrOutputParser()
                    refined_answer = refine_chain.invoke({
                        "question": question,
                        "answer": initial_answer
                    })
                    
                    st.markdown("### Refined Answer:")
                    st.success(refined_answer)
            else:
                st.success("✅ Answer is good!")
        
        except Exception as exc:
            st.error(f"Error: {exc}")

with st.expander("📚 Self-RAG Process"):
    st.markdown("""
    **Workflow:**
    1. Retrieve relevant docs
    2. Generate answer
    3. Self-evaluate quality
    4. Refine if needed
    5. Optionally retrieve more
    
    **Benefits:**
    - Higher quality outputs
    - Catches poor retrievals
    - Adaptive behavior
    - Self-improving
    """)
