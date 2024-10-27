import streamlit as st
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configuration de la page Streamlit
st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

# Initialisation du modèle LLaMA avec Ollama et de la chaîne de conversation
@st.cache_resource
def initialize_chain():
    llm = Ollama(
        model="llama3.1",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chain = initialize_chain()

# Initialisation de l'historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Bouton pour effacer le chat
if st.button("Effacer le chat"):
    st.session_state.messages = []
    chain.memory.clear()
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
        response = chain.predict(input=prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})