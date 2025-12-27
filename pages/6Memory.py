import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Learn LangChain | Memory",
    page_icon="💡"
)

st.header('💡 Memory')

st.write('''
LLMs are stateless by default, meaning that they have no built-in memory. But sometimes
we need memory to implement applications such like conversational systems, which may have
to remember previous information provided by the user. Fortunately, LangChain provides
several memory management solutions, suitable for different use cases.

We have seen beofre as vector stores are referred as long-term memory, instead, the methods
we will see in this secion are considered short-term memory, as they do not persist after
the interactions are complete.
''')

st.subheader('ConversationBufferMemory')

st.write('''
This is the simplest memory class and basically what it does, is to include previous messages
in the new LLM prompt. You can try to have a conversation with the chatbot, then ask questions
about the previous message and the LLM will be able to answer them.
''')

st.code('''
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOllama(model="gemma3:1b", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="history"
)

response = chain_with_history.invoke({"input": user_input}, config={"configurable": {"session_id": "session1"}})
''')

st.subheader('ConversationBufferWindowMemory')

st.write('''
Of course, the conversation can get long and including all the chat instory in the prompt can
become inefficient and expensive, because longest prompts result in a highest LLM token usage.
To optimize this behavior, LangChain provides three other types of memory. The
ConversationBufferWindowMemory let up decide how many messages in the chat history the system
has to remember, using a simple parameter:
''')

st.code('''
# In LCEL, you can limit history by slicing messages in the prompt
# or by managing the ChatMessageHistory manually

from langchain_core.messages import trim_messages

llm = ChatOllama(model="gemma3:1b", temperature=0.0)

# Trim to keep only last 2 messages
trimmer = trim_messages(max_tokens=2, strategy="last", token_counter=len)

chain_with_trimming = trimmer | prompt | llm
''')

st.subheader('ConversationTokenBufferMemory')

st.write('''
Very similar to the previous memory, the ConversationTokenBufferMemory type that let us be
more specific about the number of token we want to use in our prompts, and contains the content
stored to meet that limit. We have to pass the LLM as parameter, in order to calculate the 
number of tokens:
''')

st.code('''
# Token-based trimming can be done using trim_messages

from langchain_core.messages import trim_messages

llm = ChatOllama(model="gemma3:1b", temperature=0.0)

# Trim messages to fit within token limit
trimmer = trim_messages(max_tokens=50, strategy="last", token_counter=llm.get_num_tokens)
''')

st.subheader('ConversationSummaryMemory')

st.write('''
Finally, if we don't want just arbitrarily cut the memory based on a fixed lenght, we can
use the ConversationSummaryMemory, which still let us define a token limit, but passes as
memory a summary of the previous interactions. Thsi way we can still keep the short-term
memory under control while retaining the most importan information.
''')

st.code('''
# Summary memory can be implemented by periodically summarizing history
# and storing the summary instead of raw messages

from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="gemma3:1b", temperature=0.0)

summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this conversation: {history}"
)

# Periodically create summaries to compress history
''')

st.info("In the following example, we use modern LCEL with RunnableWithMessageHistory.\
 The chat history is maintained in session state.", icon="ℹ️")

memory_limit = st.selectbox(
    'Memory Strategy',
    ('Full History', 'Last 4 Messages', 'Last 2 Messages')
)

user_input = st.chat_input("Hey, how can I help you today?")

if user_input:
    # Initialize history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    
    if "memory_limit" not in st.session_state or st.session_state.memory_limit != memory_limit:
        st.session_state.memory_limit = memory_limit
        st.session_state.chat_history = ChatMessageHistory()
    
    # Build the chain
    llm = ChatOllama(model="gemma3:1b", temperature=0.0)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = chat_prompt | llm | StrOutputParser()
    
    # Get messages based on limit
    messages = st.session_state.chat_history.messages
    if memory_limit == 'Last 2 Messages':
        history_to_use = messages[-2:] if len(messages) > 2 else messages
    elif memory_limit == 'Last 4 Messages':
        history_to_use = messages[-4:] if len(messages) > 4 else messages
    else:
        history_to_use = messages
    
    # Invoke chain
    response = chain.invoke({"input": user_input, "history": history_to_use})
    
    # Store in history
    st.session_state.chat_history.add_user_message(user_input)
    st.session_state.chat_history.add_ai_message(response)
    
    st.write(response)
    
    # Show history
    st.json({"messages": [f"{m.type}: {m.content}" for m in st.session_state.chat_history.messages]})

st.divider()

st.write('A project by [Francesco Carlucci](https://francescocarlucci.com) - \
Need AI training / consulting? [Get in touch](mailto:info@francescocarlucci.com)')
