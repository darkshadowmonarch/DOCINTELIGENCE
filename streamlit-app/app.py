"""
Neotericos - Clinical Evidence Search
Streamlit App with Built-in RAG System
"""

import streamlit as st
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# Local only (.env still works locally)
load_dotenv()

# ---------------------------------------------------
# KEY LOADING (WORKS LOCAL + STREAMLIT CLOUD)
# ---------------------------------------------------
def get_api_key():
    # Streamlit Cloud secrets
    if "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]

    # Local fallback (.env)
    return os.getenv("ANTHROPIC_API_KEY")


def ensure_env_key():
    """Some SDKs require environment variable"""
    key = get_api_key()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    return key

# ---------------------------------------------------

# Page config
st.set_page_config(
    page_title="Neotericos - Clinical Evidence Search",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = None


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def initialize_rag():
    try:
        # Ensure key available everywhere
        api_key = ensure_env_key()

        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY not found in Secrets or .env")
            st.info("Streamlit Cloud → Settings → Secrets")
            st.code('ANTHROPIC_API_KEY = "sk-xxxx"')
            return False

        # Load embedding model
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = load_embedding_model()

        # Initialize ChromaDB
        with st.spinner("Loading vector database..."):
            chromadb_path = Path("chromadb_storage")
            if not chromadb_path.exists():
                st.error("❌ chromadb_storage folder missing in repo root")
                return False

            st.session_state.chroma_client = chromadb.PersistentClient(
                path=str(chromadb_path)
            )

            st.session_state.collection = st.session_state.chroma_client.get_collection(
                name="clinical_evidence"
            )

            count = st.session_state.collection.count()
            st.success(f"✅ Loaded {count} document chunks")

        # Initialize Anthropic
        st.session_state.anthropic_client = Anthropic(api_key=api_key)
        st.success("✅ Anthropic API initialized")

        st.session_state.rag_initialized = True
        return True

    except Exception as e:
        st.error(f"❌ Error initializing RAG: {e}")
        return False


def search_documents(query: str, top_k: int = 5):
    query_embedding = st.session_state.embedding_model.encode([query]).tolist()[0]

    results = st.session_state.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = []
    if results['documents'] and len(results['documents']) > 0:
        for i in range(len(results['documents'][0])):
            documents.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })

    return documents


def generate_response(query: str, context_docs: list):
    context = "\n\n".join([
        f"Source: {doc['metadata'].get('source', 'Unknown')} (Page {doc['metadata'].get('page', 'N/A')})\n{doc['text']}"
        for doc in context_docs
    ])

    history_text = ""
    if len(st.session_state.messages) > 0:
        history_text = "\n\nPrevious conversation:\n"
        for msg in st.session_state.messages[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful medical AI assistant specializing in clinical evidence and research.

Use the following context from clinical papers to answer the question.
If you don't know the answer based on the context, say so - don't make up information.
Always cite your sources by mentioning the source document names.
{history_text}
Context from research papers:
{context}

User Question: {query}

Provide a detailed, evidence-based answer with citations:"""

    message = st.session_state.anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# ---------------- UI ----------------

st.title("🔬 Neotericos")
st.caption("Clinical Evidence & Research Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")

    api_key = get_api_key()
    if api_key:
        st.success(f"🔑 API Key Loaded (...{api_key[-8:]})")
    else:
        st.error("🔑 API Key Missing")
        st.caption("Add in Streamlit Cloud Secrets")

    st.divider()

    if not st.session_state.rag_initialized:
        if st.button("🚀 Initialize RAG System", type="primary"):
            if initialize_rag():
                st.rerun()
    else:
        st.success("✅ RAG System Ready")

        if st.session_state.collection:
            count = st.session_state.collection.count()
            st.metric("Document Chunks", f"{count:,}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
