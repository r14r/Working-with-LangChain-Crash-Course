import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

st.set_page_config(
    page_title="Learn LangChain | Chains",
    page_icon="🔗"
)

st.header("🔗 Chains")

st.write("""
Now that we have a good understanding of LLMs and Prompt Templates, we are ready to
introduce chains, the most important core component of LangChain.

In modern LangChain, the recommended way is to build chains using LCEL (Runnable)
composition instead of the legacy LLMChain / SequentialChain classes.
""")

st.subheader("Our first basic Chain")

st.code("""
llm = ChatOllama(model="gemma3:1b", temperature=0.9)

prompt = ChatPromptTemplate.from_template(\"\"\"
I want you to act as a movie creative. Can you come up with an alternative name for the movie {movie}?
The name should honor the film story as it is. Please limit your answer to the name only.
If you don't know the movie, answer: "I don't know this movie"
\"\"\")

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"movie": movie})
""")

# Shared objects (created once; Streamlit reruns are fine)
llm = ChatOllama(model="gemma3:1b", temperature=0.9)
parser = StrOutputParser()

basic_prompt = ChatPromptTemplate.from_template("""
I want you to act as a movie creative. Can you come up with an alternative name for the movie {movie}?
The name should honor the film story as it is. Please limit your answer to the name only.
If you don't know the movie, answer: "I don't know this movie"
""")

basic_chain = basic_prompt | llm | parser

with st.form("basic_chain"):
    movie_basic = st.text_input("Movie", placeholder="The Green Mile", key="movie_basic")
    execute_basic = st.form_submit_button("🚀 Execute")

    if execute_basic:
        if not movie_basic.strip():
            st.warning("Please enter a movie title.")
        else:
            with st.spinner("Processing your request..."):
                response_text = basic_chain.invoke({"movie": movie_basic})
            st.code(response_text)

st.write("""
This basic Chain is not very different from the Prompt Template approach, but let's move forward to
a more complex example where we can explore the advantages and simplicity that chains bring us.
""")

st.subheader("Sequential Chain")

st.write("""
What about if we want to use multiple LLM interactions and use the output of the first run as the
input for the second chain? This is a perfect scenario for a sequential pipeline.

In this example, we ask the LLM to generate an alternative movie title, then we generate a short
advertisement using that title.
""")

st.code("""
first_chain  = first_prompt  | llm | StrOutputParser()
second_chain = second_prompt | llm | StrOutputParser()

pipeline = (
    first_chain
    | RunnableLambda(lambda title: {"movie_title": title})
    | second_prompt
    | llm
    | StrOutputParser()
)

result = pipeline.invoke({"movie": movie})
""")

first_prompt = basic_prompt

second_prompt = ChatPromptTemplate.from_template("""
Can you write a short advertisement of this new movie including its title "{movie_title}"?
Please limit it to 20 words and return only the advertisement copy.
""")

first_chain = first_prompt | llm | parser
second_chain = second_prompt | llm | parser

# Compose a sequential pipeline:
# input {"movie": "..."} -> first_chain returns title string
# map title -> {"movie_title": title} -> second_chain produces trailer string
sequential_pipeline = (
    first_chain
    | RunnableLambda(lambda title: {"movie_title": title.strip()})
    | second_chain
)

with st.form("sequential_chain"):
    movie_seq = st.text_input("Movie", placeholder="The Green Mile", key="movie_seq")
    execute_seq = st.form_submit_button("🚀 Execute")

    if execute_seq:
        if not movie_seq.strip():
            st.warning("Please enter a movie title.")
        else:
            with st.spinner("Processing your request..."):
                movie_title = first_chain.invoke({"movie": movie_seq}).strip()
                trailer = sequential_pipeline.invoke({"movie": movie_seq}).strip()

            st.json({"movie_title": movie_title, "trailer": trailer})

st.info("Couldn't we just ask for the title and the description in the first chain?", icon="❓")

st.write("""
We could, but implementing it in steps offers several advantages:
- Better debugging and more control over the LLM responses
- Better responses due to more concise and specific prompts
- More flexibility if we want to dynamically assign steps to different chains (routing)
""")

st.subheader("To keep in mind:")

st.write("""
LangChain provides many chain patterns with different purposes. Studying them all can be
overwhelming, but using them in hands-on projects makes learning practical and efficient.
""")
