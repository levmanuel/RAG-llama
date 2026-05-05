import requests
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG PDF Chat", page_icon="📚")
st.title("RAG LLaMA via Ollama")


def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        return models if models else ["llama3.1"]
    except Exception:
        return ["llama3.1"]


def process_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(texts, embeddings)


def build_rag_chain(db, model):
    llm = ChatOllama(model=model)
    retriever = db.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "En tenant compte de l'historique de la conversation et de la dernière question de l'utilisateur, "
                   "reformulez une question autonome compréhensible sans l'historique. "
                   "Ne répondez pas à la question, reformulez-la uniquement si nécessaire, sinon retournez-la telle quelle."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un assistant pour des tâches de questions-réponses. "
                   "Utilisez uniquement les éléments de contexte suivants pour répondre. "
                   "Si vous ne savez pas, dites-le simplement.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


# --- UI ---

models = get_ollama_models()
selected_model = st.selectbox("Modèle Ollama", models)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "db" not in st.session_state:
    st.session_state.db = None

if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None

# Rebuild chain if model changed after PDFs were already loaded
if st.session_state.db is not None and st.session_state.loaded_model != selected_model:
    st.session_state.chain = build_rag_chain(st.session_state.db, selected_model)
    st.session_state.loaded_model = selected_model
    st.info(f"Chaîne reconstruite avec le modèle : {selected_model}")

pdf_paths = st.text_area("Entrez les chemins des fichiers PDF (un par ligne):",
                         value="/Users/mlev/Downloads/Le_guide_de_la_parentalite_2023_01.pdf")

if st.button("Charger les PDFs"):
    with st.spinner("Chargement et traitement des PDFs..."):
        pdf_list = [path.strip() for path in pdf_paths.split("\n") if path.strip()]
        st.session_state.db = process_pdfs(pdf_list)
        st.session_state.chain = build_rag_chain(st.session_state.db, selected_model)
        st.session_state.loaded_model = selected_model
        st.session_state.messages = []
        st.session_state.history = []
        st.success("PDFs chargés et traités avec succès!")

if st.session_state.chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = st.session_state.chain.invoke({
                "input": prompt,
                "history": st.session_state.history,
            })
            answer = result["answer"].content if hasattr(result["answer"], "content") else result["answer"]
            st.markdown(answer)

        st.session_state.history.append(HumanMessage(content=prompt))
        st.session_state.history.append(AIMessage(content=answer))
        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Effacer le chat"):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()
