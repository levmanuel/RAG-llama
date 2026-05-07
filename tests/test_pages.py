"""
Tests fonctionnels des pages Streamlit RAG-Llama.
Les dépendances externes (Streamlit, LangChain, Ollama) sont toutes mockées
via conftest.py — aucun service externe n'est requis.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

PAGES = Path(__file__).parent.parent / "pages"


def _load(filename: str):
    """Charge un module de page Streamlit via importlib."""
    path = PAGES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Chargement unique des modules (les mocks streamlit/langchain sont déjà en place)
rag = _load("llama_RAG_PDF.py")
nomem = _load("llama_no_memory.py")
mem = _load("llama_with_memory.py")
sys_m = _load("llama_with_memory_system_inx.py")


# ─────────────────────────────────────────────────────────────
# get_ollama_models  (logique identique dans les 4 pages)
# ─────────────────────────────────────────────────────────────

class TestGetOllamaModels:

    def _response(self, names):
        r = Mock()
        r.json.return_value = {"models": [{"name": n} for n in names]}
        return r

    def test_retourne_liste_des_modeles_disponibles(self):
        with patch.object(rag, "requests") as req:
            req.get.return_value = self._response(["llama3.1", "mistral", "gemma2"])
            result = rag.get_ollama_models()
        assert result == ["llama3.1", "mistral", "gemma2"]

    def test_fallback_si_ollama_inaccessible(self):
        with patch.object(rag, "requests") as req:
            req.get.side_effect = ConnectionError("refused")
            result = rag.get_ollama_models()
        assert result == ["llama3.1"]

    def test_fallback_si_liste_modeles_vide(self):
        with patch.object(rag, "requests") as req:
            req.get.return_value = self._response([])
            result = rag.get_ollama_models()
        assert result == ["llama3.1"]

    def test_fallback_si_exception_generique(self):
        with patch.object(rag, "requests") as req:
            req.get.side_effect = Exception("timeout")
            result = rag.get_ollama_models()
        assert result == ["llama3.1"]


# ─────────────────────────────────────────────────────────────
# process_pdfs
# ─────────────────────────────────────────────────────────────

class TestProcessPdfs:

    def _mocks(self):
        loader_inst = MagicMock()
        loader_inst.load.return_value = [MagicMock()]
        loader_cls = MagicMock(return_value=loader_inst)

        splitter_inst = MagicMock()
        splitter_inst.split_documents.return_value = [MagicMock()]
        splitter_cls = MagicMock(return_value=splitter_inst)

        faiss_cls = MagicMock()
        faiss_cls.from_documents.return_value = MagicMock()

        return loader_cls, splitter_cls, MagicMock(), faiss_cls

    def _patch(self, loader_cls, splitter_cls, emb, faiss_cls):
        return (
            patch.object(rag, "PyPDFLoader", loader_cls),
            patch.object(rag, "RecursiveCharacterTextSplitter", splitter_cls),
            patch.object(rag, "HuggingFaceEmbeddings", return_value=emb),
            patch.object(rag, "FAISS", faiss_cls),
        )

    def test_charge_un_seul_pdf(self):
        lc, sc, emb, fc = self._mocks()
        with self._patch(lc, sc, emb, fc)[0], self._patch(lc, sc, emb, fc)[1], \
             self._patch(lc, sc, emb, fc)[2], self._patch(lc, sc, emb, fc)[3]:
            rag.process_pdfs(["/tmp/doc.pdf"])
        lc.assert_called_once_with("/tmp/doc.pdf")
        lc.return_value.load.assert_called_once()

    def test_charge_plusieurs_pdfs(self):
        lc, sc, emb, fc = self._mocks()
        patches = self._patch(lc, sc, emb, fc)
        with patches[0], patches[1], patches[2], patches[3]:
            rag.process_pdfs(["/a.pdf", "/b.pdf", "/c.pdf"])
        assert lc.call_count == 3

    def test_chunking_avec_bons_parametres(self):
        lc, sc, emb, fc = self._mocks()
        patches = self._patch(lc, sc, emb, fc)
        with patches[0], patches[1], patches[2], patches[3]:
            rag.process_pdfs(["/tmp/doc.pdf"])
        sc.assert_called_once_with(chunk_size=1000, chunk_overlap=200)

    def test_retourne_base_vectorielle_faiss(self):
        lc, sc, emb, fc = self._mocks()
        expected = MagicMock()
        fc.from_documents.return_value = expected
        patches = self._patch(lc, sc, emb, fc)
        with patches[0], patches[1], patches[2], patches[3]:
            result = rag.process_pdfs(["/tmp/doc.pdf"])
        assert result is expected


# ─────────────────────────────────────────────────────────────
# build_rag_chain
# ─────────────────────────────────────────────────────────────

class TestBuildRagChain:

    def _base_patches(self):
        return (
            patch.object(rag, "ChatOllama"),
            patch.object(rag, "create_history_aware_retriever"),
            patch.object(rag, "create_stuff_documents_chain"),
            patch.object(rag, "create_retrieval_chain"),
        )

    def test_chatollama_reçoit_le_bon_modele(self):
        chat_cls = MagicMock()
        with patch.object(rag, "ChatOllama", chat_cls), \
             patch.object(rag, "create_history_aware_retriever"), \
             patch.object(rag, "create_stuff_documents_chain"), \
             patch.object(rag, "create_retrieval_chain"):
            rag.build_rag_chain(MagicMock(), "mistral:latest")
        chat_cls.assert_called_once_with(model="mistral:latest")

    def test_retourne_la_chaine_finale(self):
        expected = MagicMock()
        with patch.object(rag, "ChatOllama"), \
             patch.object(rag, "create_history_aware_retriever"), \
             patch.object(rag, "create_stuff_documents_chain"), \
             patch.object(rag, "create_retrieval_chain", return_value=expected):
            result = rag.build_rag_chain(MagicMock(), "llama3.1")
        assert result is expected

    def test_history_aware_retriever_est_appele(self):
        with patch.object(rag, "ChatOllama"), \
             patch.object(rag, "create_history_aware_retriever") as har, \
             patch.object(rag, "create_stuff_documents_chain"), \
             patch.object(rag, "create_retrieval_chain"):
            rag.build_rag_chain(MagicMock(), "llama3.1")
        har.assert_called_once()

    def test_retrieval_chain_reçoit_har_et_qa(self):
        har_result = MagicMock()
        qa_result = MagicMock()
        with patch.object(rag, "ChatOllama"), \
             patch.object(rag, "create_history_aware_retriever", return_value=har_result), \
             patch.object(rag, "create_stuff_documents_chain", return_value=qa_result), \
             patch.object(rag, "create_retrieval_chain") as rc:
            rag.build_rag_chain(MagicMock(), "llama3.1")
        rc.assert_called_once_with(har_result, qa_result)


# ─────────────────────────────────────────────────────────────
# Pages de chat : initialisation LLM / chaîne
# ─────────────────────────────────────────────────────────────

class TestChatPages:

    def test_no_memory_llm_reçoit_le_bon_modele(self):
        llm_cls = MagicMock()
        with patch.object(nomem, "OllamaLLM", llm_cls):
            nomem.initialize_llm("gemma2")
        llm_cls.assert_called_once_with(model="gemma2")

    def test_with_memory_llm_reçoit_le_bon_modele(self):
        llm_cls = MagicMock()
        with patch.object(mem, "OllamaLLM", llm_cls):
            mem.initialize_chain("mistral")
        llm_cls.assert_called_once_with(model="mistral")

    def test_system_message_llm_reçoit_le_bon_modele(self):
        llm_cls = MagicMock()
        with patch.object(sys_m, "OllamaLLM", llm_cls):
            sys_m.initialize_llm("llama3.2")
        llm_cls.assert_called_once_with(model="llama3.2")

    def test_system_message_inclus_dans_prompt(self):
        custom_msg = "Tu es un expert en droit fiscal."
        tpl_mock = MagicMock()
        tpl_mock.from_messages.return_value.__or__ = MagicMock(return_value=MagicMock())

        with patch.object(sys_m, "ChatPromptTemplate", tpl_mock):
            sys_m.build_chain(custom_msg)

        # Vérifie que from_messages a été appelé avec un tuple ("system", custom_msg)
        call_args = tpl_mock.from_messages.call_args[0][0]
        system_tuples = [m for m in call_args if isinstance(m, tuple) and m[0] == "system"]
        assert any(custom_msg in m[1] for m in system_tuples)

    def test_with_memory_chain_produit_pipe(self):
        """initialize_chain doit composer prompt | llm via LCEL."""
        mock_llm = MagicMock()
        mock_prompt = MagicMock()
        expected = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=expected)

        tpl_mock = MagicMock()
        tpl_mock.from_messages.return_value = mock_prompt

        with patch.object(mem, "OllamaLLM", return_value=mock_llm), \
             patch.object(mem, "ChatPromptTemplate", tpl_mock):
            result = mem.initialize_chain("llama3.1")

        mock_prompt.__or__.assert_called_once_with(mock_llm)
        assert result is expected
