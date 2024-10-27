import streamlit as st
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configuration de la page Streamlit
st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

# Initialisation du modèle LLaMA avec Ollama
@st.cache_resource
def initialize_llm():
    llm = Ollama(
        model="llama3.1",  # Correction du nom du modèle
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

llm = initialize_llm()

# Initialisation de l'historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bouton pour effacer le chat
if st.button("Effacer le chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Affichage de l'historique du chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie de l'utilisateur
if prompt := st.chat_input("Que voulez-vous demander à LLaMA ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = llm(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})