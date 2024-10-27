import streamlit as st
import os
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama
from langchain.vectorstores import FAISS


# Configuration de la page Streamlit
st.set_page_config(page_title="RAG PDF Chat", page_icon="📚")
st.title("RAG Llama 3.1")

# Initialisation de la session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "db" not in st.session_state:
    st.session_state.db = None

# Fonction pour charger et traiter les PDFs
def process_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    
    # Utilisation de FAISS au lieu de Chroma
    db = FAISS.from_documents(texts, embeddings)
    
    return db

# Interface utilisateur
pdf_paths = st.text_area("Entrez les chemins des fichiers PDF (un par ligne):", value='/Users/mlev/Downloads/Le_guide_de_la_parentalite_2023_01.pdf')
if st.button("Charger les PDFs"):
    with st.spinner("Chargement et traitement des PDFs..."):
        pdf_list = [path.strip() for path in pdf_paths.split("\n") if path.strip()]
        st.session_state.db = process_pdfs(pdf_list)
        st.success("PDFs chargés et traités avec succès!")

# Initialisation du modèle et de la chaîne de conversation
if st.session_state.db and not st.session_state.conversation:
    llm = Ollama(model="llama3.1")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.db.as_retriever(),
        memory=memory
    )

# Zone de chat
if st.session_state.conversation:
    if prompt := st.chat_input("Posez votre question ici"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = st.session_state.conversation({"question": prompt})
            full_response = response['answer']
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bouton pour effacer le chat
if st.button("Effacer le chat"):
    st.session_state.messages = []
    st.session_state.conversation = None
    st.experimental_rerun()