"""
AI Pharmacy - RAG Chat Assistant
---------------------------------
Upload a pharmacy/medical PDF (drug leaflet, formulary, guideline, etc.),
the app chunks + embeds it, stores it in a FAISS vector index, and lets you
ask questions answered by a Groq-hosted LLM grounded ONLY in that document.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub -> deploy on share.streamlit.io (Streamlit
    Community Cloud) -> add GROQ_API_KEY in the app's Secrets.
"""

import streamlit as st

from rag_utils import (
    extract_text_from_pdf,
    chunk_text,
    get_embedding_model,
    build_vector_store,
    build_qa_chain,
    answer_question,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Pharmacy Assistant",
    page_icon="💊",
    layout="wide",
)

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return get_embedding_model()


# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💊 AI Pharmacy")
    st.caption("RAG assistant over your own pharmacy documents")

    st.subheader("1. Groq_API_Key")
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    groq_api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
        "Add it as a Streamlit secret named GROQ_API_KEY when deploying so users don't have to paste it.",
    )

    st.subheader("2. Model")
    model_name = st.selectbox(
        "Groq model",
        GROQ_MODELS,
        index=0,
        help="Free-tier Groq models. If a model errors out, Groq's catalog may have "
        "changed — check https://console.groq.com/docs/models for the current list.",
    )

    st.subheader("3. Chunking")
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=50)
    top_k = st.slider("Chunks retrieved per question (k)", 2, 8, 4)

    st.divider()
    if st.button("🗑️ Clear document & chat", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.session_state.processed_file_name = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("💊 AI Pharmacy Assistant")
st.caption(
    "Upload a pharmacy/medical PDF (drug leaflet, formulary, guideline). "
    "Ask questions and get answers grounded strictly in that document."
)
st.warning(
    "⚠️ Educational tool only. Not a substitute for advice from a licensed pharmacist or doctor. "
    "Do not use this app to make real medication decisions.",
    icon="⚠️",
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file_name:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar before processing a document.")
    else:
        with st.status("Processing document...", expanded=True) as status:
            st.write("📄 Extracting text from PDF...")
            raw_text = extract_text_from_pdf(uploaded_file.read())

            if not raw_text.strip():
                status.update(label="No extractable text found", state="error")
                st.error(
                    "Couldn't extract any text from this PDF. It may be a scanned "
                    "image-only document — try a text-based PDF instead."
                )
            else:
                st.write(f"✂️ Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
                documents = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.write(f"Created **{len(documents)}** chunks.")

                st.write("🧠 Generating embeddings (local, free model)...")
                embedding_model = load_embedding_model()

                st.write("📦 Building FAISS vector index...")
                vector_store = build_vector_store(documents, embedding_model)

                st.write("🔗 Connecting to Groq LLM...")
                qa_chain = build_qa_chain(vector_store, groq_api_key, model_name, k=top_k)

                st.session_state.vector_store = vector_store
                st.session_state.qa_chain = qa_chain
                st.session_state.chat_history = []
                st.session_state.processed_file_name = uploaded_file.name

                status.update(label="Document ready ✅", state="complete")

if st.session_state.processed_file_name:
    st.success(f"📄 Loaded: **{st.session_state.processed_file_name}** — ask away below.")

st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("View source chunks used"):
                for i, doc in enumerate(turn["sources"], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:600])

question = st.chat_input("Ask a question about the uploaded document...")

if question:
    if st.session_state.qa_chain is None:
        st.error("Please upload a PDF and enter your Groq API key first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(st.session_state.qa_chain, question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"⚠️ Something went wrong calling Groq: `{e}`"
                    sources = []
                st.markdown(answer)
                if sources:
                    with st.expander("View source chunks used"):
                        for i, doc in enumerate(sources, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(doc.page_content[:600])

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
