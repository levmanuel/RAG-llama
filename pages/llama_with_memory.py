import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

@st.cache_resource
def initialize_chain():
    llm = OllamaLLM(model="llama3.1")
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm

chain = initialize_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Effacer le chat"):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Que voulez-vous demander à LLaMA ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke({"history": st.session_state.history, "input": prompt})
        st.markdown(response)

    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=response))
    st.session_state.messages.append({"role": "assistant", "content": response})
