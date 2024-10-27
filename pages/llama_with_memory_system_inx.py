import streamlit as st
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Configuration de la page Streamlit
st.set_page_config(page_title="Chat LLaMA 3.1", page_icon="🦙")
st.title("Chat avec LLaMA 3.1 via Ollama")

# Définir le message système par défaut
default_system_message = "Vous êtes un assistant IA utile et amical. Répondez aux questions de l'utilisateur de manière claire et concise."

# Initialiser le message système dans session_state s'il n'existe pas
if "system_message" not in st.session_state:
    st.session_state.system_message = default_system_message

# Zone de texte pour éditer le message système
new_system_message = st.text_area("Éditer le message système:", value=st.session_state.system_message, height=100)

# Vérifier si le message système a été modifié
if new_system_message != st.session_state.system_message:
    st.session_state.system_message = new_system_message
    st.session_state.chain_initialized = False
    st.success("Message système mis à jour. La chaîne sera réinitialisée avec le nouveau message.")

# Initialisation du modèle LLaMA avec Ollama et de la chaîne de conversation
def initialize_chain(system_message):
    llm = Ollama(
        model="llama3.1",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    
    # Créer le template de prompt avec le message système
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)
    return chain

# Initialiser ou réinitialiser la chaîne si nécessaire
if "chain_initialized" not in st.session_state or not st.session_state.chain_initialized:
    chain = initialize_chain(st.session_state.system_message)
    st.session_state.chain = chain
    st.session_state.chain_initialized = True
    st.info(f"Chaîne initialisée avec le message système : {st.session_state.system_message}")
else:
    chain = st.session_state.chain

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

# Bouton pour vérifier le message système actuel
if st.button("Vérifier le message système actuel"):
    st.info(f"Message système actuel : {st.session_state.system_message}")