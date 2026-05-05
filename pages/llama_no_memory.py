import streamlit as st
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

@st.cache_resource
def initialize_llm():
    return OllamaLLM(model="llama3.1")

llm = initialize_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Effacer le chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Que voulez-vous demander à LLaMA ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = llm.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
