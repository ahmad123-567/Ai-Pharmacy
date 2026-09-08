"""
rag_utils.py
------------
Core building blocks for the AI Pharmacy RAG app:
  1. Extract text from an uploaded PDF
  2. Split text into overlapping chunks
  3. Embed chunks with a local HuggingFace sentence-transformer (free, no API key)
  4. Store/query vectors with FAISS
  5. Answer questions with a Groq-hosted LLM, grounded only in retrieved chunks
"""

from __future__ import annotations

import io
from typing import List

from pypdf import PdfReader


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_classic.chains import create_retrieval_chain


# ---------------------------------------------------------------------------
# 1. PDF -> raw text
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an in-memory PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages_text)


# ---------------------------------------------------------------------------
# 2. Text -> chunks
# ---------------------------------------------------------------------------
def chunk_text(
    raw_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split raw text into overlapping chunks wrapped as LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]


# ---------------------------------------------------------------------------
# 3 & 4. Chunks -> embeddings -> FAISS vector store
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Local, free embedding model (downloaded once, cached by Streamlit)."""
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)


def build_vector_store(documents: List[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """Embed the chunks and build an in-memory FAISS index."""
    return FAISS.from_documents(documents, embedding_model)


# ---------------------------------------------------------------------------
# 5. Retrieval + Groq LLM answer chain
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are AI Pharmacy Assistant, a careful, factual assistant that answers \
questions ONLY using the context retrieved from the pharmacy/medical PDF documents provided below.

Rules you must follow:
1. Answer strictly from the given context. If the answer is not contained in the context, say \
clearly: "I could not find this information in the uploaded document." Do not guess or use outside \
knowledge about drugs, dosages, or interactions.
2. When you do answer, be precise about drug names, dosages, forms, and warnings exactly as stated \
in the context. Quote figures (e.g. dosage in mg) exactly as written.
3. Always end every answer with this exact disclaimer on its own line: \
"⚠️ This is informational only and not a substitute for advice from a licensed pharmacist or doctor."
4. Keep answers concise and use bullet points for lists (e.g. side effects, dosage instructions).

Context from the document:
{context}
"""


def get_llm(groq_api_key: str, model_name: str, temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )


def build_qa_chain(vector_store: FAISS, groq_api_key: str, model_name: str, k: int = 4):
    """Build a retrieval-augmented QA chain backed by Groq."""
    llm = get_llm(groq_api_key, model_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def answer_question(qa_chain, question: str) -> dict:
    """
    Run the chain and return a dict with:
      - 'answer': the generated answer text
      - 'sources': list of retrieved Document chunks used as context
    """
    result = qa_chain.invoke({"input": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("context", []),
    }
