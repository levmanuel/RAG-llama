import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

default_system_message = "Vous êtes un assistant IA utile et amical. Répondez aux questions de l'utilisateur de manière claire et concise."

if "system_message" not in st.session_state:
    st.session_state.system_message = default_system_message

new_system_message = st.text_area("Éditer le message système:", value=st.session_state.system_message, height=100)

if new_system_message != st.session_state.system_message:
    st.session_state.system_message = new_system_message
    st.session_state.messages = []
    st.session_state.history = []
    st.success("Message système mis à jour. L'historique a été réinitialisé.")

@st.cache_resource
def initialize_llm():
    return OllamaLLM(model="llama3.1")

llm = initialize_llm()

def build_chain(system_message):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm

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

    chain = build_chain(st.session_state.system_message)
    with st.chat_message("assistant"):
        response = chain.invoke({"history": st.session_state.history, "input": prompt})
        st.markdown(response)

    st.session_state.history.append(HumanMessage(content=prompt))
    st.session_state.history.append(AIMessage(content=response))
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Vérifier le message système actuel"):
    st.info(f"Message système actuel : {st.session_state.system_message}")
