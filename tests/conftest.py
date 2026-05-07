import sys
from unittest.mock import MagicMock


class SessionState:
    """Simule st.session_state avec accès attribut et opérateur 'in'."""

    def __contains__(self, key):
        return key in vars(self)

    def __getattr__(self, key):
        return None  # attribut non défini → None


# --- Mock Streamlit ---
_st = MagicMock()
_st.cache_resource = lambda f: f   # @st.cache_resource devient un no-op
_st.button.return_value = False
_st.chat_input.return_value = None
# text_area retourne la même valeur que le message système par défaut
_default_sys_msg = (
    "Vous êtes un assistant IA utile et amical. "
    "Répondez aux questions de l'utilisateur de manière claire et concise."
)
_st.text_area.return_value = _default_sys_msg
_st.selectbox.return_value = "llama3.1"
_st.session_state = SessionState()
sys.modules["streamlit"] = _st

# --- Mock LangChain et dépendances ---
for _mod in [
    "langchain",
    "langchain_ollama",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.vectorstores",
    "langchain_text_splitters",
    "langchain_huggingface",
    "langchain.chains",
    "langchain.chains.combine_documents",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.messages",
]:
    sys.modules.setdefault(_mod, MagicMock())
