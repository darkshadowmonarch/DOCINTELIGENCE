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

# Load environment variables from .env file
load_dotenv()

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
    """Load and cache the embedding model"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def initialize_rag():
    """Initialize RAG system components"""
    try:
        # Load embedding model
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = load_embedding_model()

        # Initialize ChromaDB
        with st.spinner("Loading vector database..."):
            chromadb_path = Path("chromadb_storage")
            if not chromadb_path.exists():
                st.error("❌ ChromaDB not found! Please copy chromadb_storage folder here.")
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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY not found!")
            st.info("💡 Make sure .env file exists with your API key")
            st.code("ANTHROPIC_API_KEY=your_key_here")

            # Show current directory for debugging
            st.caption(f"Current directory: {os.getcwd()}")
            st.caption(f".env exists: {Path('.env').exists()}")
            return False

        st.session_state.anthropic_client = Anthropic(api_key=api_key)
        st.success("✅ Anthropic API initialized")
        st.session_state.rag_initialized = True

        return True

    except Exception as e:
        st.error(f"❌ Error initializing RAG: {e}")
        return False


def search_documents(query: str, top_k: int = 5):
    """Search for relevant documents"""
    # Generate query embedding
    query_embedding = st.session_state.embedding_model.encode([query]).tolist()[0]

    # Search ChromaDB
    results = st.session_state.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Format results
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
    """Generate response using Claude"""
    # Build context
    context = "\n\n".join([
        f"Source: {doc['metadata'].get('source', 'Unknown')} (Page {doc['metadata'].get('page', 'N/A')})\n{doc['text']}"
        for doc in context_docs
    ])

    # Build prompt with chat history
    history_text = ""
    if len(st.session_state.messages) > 0:
        history_text = "\n\nPrevious conversation:\n"
        for msg in st.session_state.messages[-6:]:  # Last 3 exchanges
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

    # Call Claude
    message = st.session_state.anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# UI Layout
st.title("🔬 Neotericos")
st.caption("Clinical Evidence & Research Search")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")

    # Show API key status
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        st.success(f"🔑 API Key: ...{api_key[-8:]}")
    else:
        st.error("🔑 API Key: Not found")
        st.caption("Create .env file with:\nANTHROPIC_API_KEY=your_key")

    st.divider()

    if not st.session_state.rag_initialized:
        if st.button("🚀 Initialize RAG System", type="primary"):
            if initialize_rag():
                st.rerun()
    else:
        st.success("✅ RAG System Ready")

        # Show stats
        if st.session_state.collection:
            count = st.session_state.collection.count()
            st.metric("Document Chunks", f"{count:,}")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.header("ℹ️ About")
    st.info("""
    **Neotericos** is an AI-powered clinical evidence search system.

    Ask questions about:
    - Treatment guidelines
    - Clinical trials
    - Research findings
    - Medical evidence

    Powered by:
    - ChromaDB (vector store)
    - Claude Sonnet 4.5 (LLM)
    - Sentence Transformers (embeddings)
    """)

# Main chat area
if not st.session_state.rag_initialized:
    st.info("👈 Click 'Initialize RAG System' in the sidebar to get started")

    st.markdown("### 📚 Example Questions:")
    st.markdown("""
    - What are the latest treatment options for diabetes?
    - What is the evidence for metformin use in type 2 diabetes?
    - What are the guidelines for hypertension management?
    - Compare treatment options for rheumatoid arthritis
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📄 Sources ({len(message['sources'])})"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** (Page {source['page']})")

    # Chat input
    if prompt := st.chat_input("Ask about clinical evidence..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching clinical evidence..."):
                try:
                    # Search for relevant documents
                    relevant_docs = search_documents(prompt)

                    # Generate response
                    response = generate_response(prompt, relevant_docs)

                    # Display response
                    st.markdown(response)

                    # Show sources
                    if relevant_docs:
                        with st.expander(f"📄 Sources ({len(relevant_docs)})"):
                            for doc in relevant_docs:
                                st.markdown(f"- **{doc['metadata'].get('source', 'Unknown')}** (Page {doc['metadata'].get('page', 'N/A')})")

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": [{"source": doc['metadata'].get('source', 'Unknown'),
                                   "page": doc['metadata'].get('page', 'N/A')}
                                  for doc in relevant_docs]
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.divider()
st.caption("⚠️ Always verify critical medical information with primary sources")
