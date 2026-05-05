# RAG-Llama

Application web multi-pages construite avec **Streamlit** pour explorer différentes approches de chatbot avec **LLaMA 3.1** tournant localement via **Ollama**.

## Pages disponibles

| Page | Description |
|---|---|
| `llama_no_memory` | Chat simple sans mémoire de conversation |
| `llama_with_memory` | Chat avec historique de conversation |
| `llama_with_memory_system_inx` | Chat avec historique et message système personnalisable |
| `llama_RAG_PDF` | Questions-réponses sur vos fichiers PDF (RAG) |

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et le modèle LLaMA 3.1 téléchargé

```bash
ollama pull llama3.1
```

## Installation

```bash
git clone https://github.com/levmanuel/RAG-llama.git
cd RAG-llama

pip install streamlit langchain langchain-ollama langchain-community \
            langchain-huggingface langchain-text-splitters faiss-cpu \
            pypdf sentence-transformers
```

## Lancement

```bash
streamlit run main.py
```

L'application est accessible sur `http://localhost:8501`.

## Architecture technique

- **LLM** : LLaMA 3.1 via Ollama (`langchain-ollama`)
- **Mémoire** : gestion explicite de l'historique avec `HumanMessage` / `AIMessage`
- **Chaînes** : LCEL — composition `prompt | llm`
- **RAG** : `create_history_aware_retriever` + `create_retrieval_chain` + FAISS
- **Embeddings** : `HuggingFaceEmbeddings` (modèle local, sans clé API)
