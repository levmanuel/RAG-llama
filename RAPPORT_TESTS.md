# Rapport de tests fonctionnels — RAG-Llama

**Date :** 2026-05-07  
**Outil :** pytest 9.0.3 — Python 3.11.15  
**Résultat global : ✅ 17 / 17 tests passés (0,28 s)**

---

## Stratégie

Les pages Streamlit ne peuvent pas être testées comme des applications web classiques (pas de navigateur, pas de serveur). La stratégie adoptée consiste à :

1. **Mocker Streamlit** avant tout import de page — les appels `st.*` deviennent des no-ops.
2. **Mocker les dépendances LangChain** (`langchain_ollama`, `langchain_community`, etc.) afin qu'aucun modèle ni service externe ne soit requis.
3. **Charger les modules de page** via `importlib` pour pouvoir appeler leurs fonctions directement.
4. **Patcher au bon endroit** (`patch.object(module, "NomDeLaClasse")`) pour contrôler chaque appel unitairement.

> Aucun service externe (Ollama, modèles HuggingFace, fichiers PDF) n'est nécessaire pour exécuter la suite.

---

## Résultats détaillés

### `get_ollama_models` — 4 tests

Fonction commune aux 4 pages. Elle interroge l'API locale Ollama (`GET /api/tags`) et retourne la liste des modèles installés.

| # | Test | Résultat |
|---|------|----------|
| 1 | Retourne la liste des modèles quand Ollama répond | ✅ PASSED |
| 2 | Fallback `["llama3.1"]` si Ollama est inaccessible (`ConnectionError`) | ✅ PASSED |
| 3 | Fallback `["llama3.1"]` si la liste de modèles est vide | ✅ PASSED |
| 4 | Fallback `["llama3.1"]` sur toute exception générique (timeout…) | ✅ PASSED |

**Comportement vérifié :** la gestion d'erreur est robuste — l'application démarre toujours même sans Ollama.

---

### `process_pdfs` — 4 tests

Pipeline de traitement de la page RAG : chargement PDF → découpage → embeddings → base vectorielle FAISS.

| # | Test | Résultat |
|---|------|----------|
| 5 | Un seul PDF : `PyPDFLoader` appelé une fois avec le bon chemin | ✅ PASSED |
| 6 | Plusieurs PDFs : `PyPDFLoader` appelé autant de fois que de fichiers | ✅ PASSED |
| 7 | Paramètres de chunking corrects (`chunk_size=1000`, `chunk_overlap=200`) | ✅ PASSED |
| 8 | La valeur de retour est bien la base vectorielle FAISS | ✅ PASSED |

**Comportement vérifié :** le pipeline respecte l'ordre et les paramètres attendus. L'objet retourné est bien le vector store.

---

### `build_rag_chain` — 4 tests

Construction de la chaîne RAG conversationnelle (`create_history_aware_retriever` + `create_retrieval_chain`).

| # | Test | Résultat |
|---|------|----------|
| 9  | `ChatOllama` est instancié avec le nom de modèle sélectionné | ✅ PASSED |
| 10 | La fonction retourne la chaîne finale produite par `create_retrieval_chain` | ✅ PASSED |
| 11 | `create_history_aware_retriever` est bien appelé | ✅ PASSED |
| 12 | `create_retrieval_chain` reçoit exactement `(history_aware_retriever, qa_chain)` | ✅ PASSED |

**Comportement vérifié :** la composition de la chaîne est correcte. Le modèle sélectionné est bien propagé à `ChatOllama`.

---

### Pages de chat — 5 tests

Les trois pages chat (`llama_no_memory`, `llama_with_memory`, `llama_with_memory_system_inx`).

| # | Test | Page | Résultat |
|---|------|------|----------|
| 13 | `OllamaLLM` reçoit le bon nom de modèle | `llama_no_memory` | ✅ PASSED |
| 14 | `OllamaLLM` reçoit le bon nom de modèle | `llama_with_memory` | ✅ PASSED |
| 15 | `OllamaLLM` reçoit le bon nom de modèle | `llama_with_memory_system_inx` | ✅ PASSED |
| 16 | Le message système personnalisé est inclus dans le prompt | `llama_with_memory_system_inx` | ✅ PASSED |
| 17 | La chaîne est composée via LCEL (`prompt \| llm`) | `llama_with_memory` | ✅ PASSED |

**Comportement vérifié :** le modèle sélectionné est correctement transmis au LLM. Le message système est bien injecté dans le `ChatPromptTemplate`. La composition LCEL (`prompt | llm`) est utilisée.

---

## Couverture fonctionnelle

| Fonctionnalité | Couverte |
|---|---|
| Récupération des modèles Ollama | ✅ |
| Gestion d'erreur Ollama (inaccessible / vide / timeout) | ✅ |
| Chargement et découpage de PDFs | ✅ |
| Paramètres de chunking | ✅ |
| Création de la base vectorielle FAISS | ✅ |
| Construction de la chaîne RAG (history-aware) | ✅ |
| Propagation du modèle sélectionné (toutes les pages) | ✅ |
| Message système personnalisable | ✅ |
| Composition LCEL (`prompt \| llm`) | ✅ |
| Reconstruction automatique de la chaîne au changement de modèle | ⚠️ Non couverte (logique dans le flux Streamlit, non testable unitairement) |
| Interface utilisateur (boutons, chat input, affichage) | ⚠️ Hors scope (nécessiterait Selenium ou Playwright) |

---

## Lancer les tests

```bash
pip install pytest
python -m pytest tests/test_pages.py -v
```

```
17 passed in 0.28s
```
