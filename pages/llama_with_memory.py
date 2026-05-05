import requests
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat LLaMA", page_icon="🦙")
st.title("Chat LLaMA via Ollama")


def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        return models if models else ["llama3.1"]
    except Exception:
        return ["llama3.1"]


models = get_ollama_models()
selected_model = st.selectbox("Modèle Ollama", models)


@st.cache_resource
def initialize_chain(model: str):
    llm = OllamaLLM(model=model)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm


chain = initialize_chain(selected_model)

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

if prompt := st.chat_input("Que voulez-vous demander ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke({"history": st.session_state.history, "input": prompt})
        st.markdown(response)

    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=response))
    st.session_state.messages.append({"role": "assistant", "content": response})
