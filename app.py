import os
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached model
# -----------------------------
@st.cache_resource(show_spinner="Loading open-source embedding model...")
def load_embedding_model():
    """Load the open-source Sentence Transformers embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(uploaded_file) -> List[Dict]:
    """Extract text page-by-page so retrieved chunks can show page numbers."""
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping character-based chunks."""
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_chunks(pages: List[Dict], filename: str) -> List[Dict]:
    """Chunk every PDF page while retaining source metadata."""
    all_chunks = []

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_data["page"],
                    "filename": filename,
                    "chunk_number": chunk_number,
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(chunks: List[Dict], model) -> faiss.Index:
    """Embed chunks and store normalized vectors in a FAISS index."""
    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(
    question: str,
    index: faiss.Index,
    chunks: List[Dict],
    model,
    top_k: int = TOP_K,
) -> List[Tuple[Dict, float]]:
    """Retrieve the most relevant chunks for the user's question."""
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, index.ntotal)

    if k == 0:
        return []

    scores, indices = index.search(question_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((chunks[int(idx)], float(score)))

    return results


# -----------------------------
# Groq generation
# -----------------------------
def get_groq_client() -> Groq:
    """Read GROQ_API_KEY from Streamlit secrets or environment variables."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    retrieved: List[Tuple[Dict, float]],
) -> str:
    """Generate an answer using only the retrieved document context."""
    if not retrieved:
        return "I could not find relevant information in the uploaded document."

    context_parts = []
    for i, (item, score) in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i} | {item['filename']} | Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful document-question-answering assistant.

Rules:
1. Answer using ONLY the supplied document context.
2. If the answer is not contained in the context, say:
   "I couldn't find that information in the uploaded document."
3. Do not invent facts, sources, page numbers, quotations, or citations.
4. Keep the answer clear and useful.
5. When possible, mention the relevant page number(s).
"""

    user_prompt = f"""Document context:

{context}

User question:
{question}

Answer the question from the document context only.
"""

    client = get_groq_client()

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return completion.choices[0].message.content.strip()


# -----------------------------
# Session-state helpers
# -----------------------------
def reset_rag_state():
    """Clear the current document/index from the Streamlit session."""
    for key in [
        "faiss_index",
        "chunks",
        "document_name",
        "page_count",
        "chunk_count",
        "messages",
    ]:
        st.session_state.pop(key, None)


# -----------------------------
# UI
# -----------------------------
st.title("📚 RAG PDF Chat")
st.caption(
    "Upload a PDF → extract text → create chunks → generate open-source "
    "embeddings → search with FAISS → answer with Groq."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="Number of relevant chunks sent to the language model.",
    )

    st.divider()
    st.write("**Embedding model:**")
    st.code(EMBEDDING_MODEL, language="text")

    st.write("**Generation model:**")
    st.code(GROQ_MODEL, language="text")

    st.info(
        "The embedding model is open source. FAISS is an open-source "
        "vector similarity-search library. Groq provides the LLM inference API."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="For scanned/image-only PDFs, OCR is required; this version reads selectable PDF text.",
)

if uploaded_file is not None:
    current_name = uploaded_file.name

    if st.session_state.get("document_name") != current_name:
        reset_rag_state()

        with st.status("Building your RAG knowledge base...", expanded=True) as status:
            st.write("1/4 Extracting PDF text...")
            pages = extract_pdf_text(uploaded_file)

            if not pages:
                status.update(
                    label="No readable text found",
                    state="error",
                )
                st.error(
                    "This PDF does not contain extractable text. "
                    "If it is a scanned PDF, add OCR before using it."
                )
                st.stop()

            st.write(f"2/4 Extracted text from {len(pages)} page(s).")
            chunks = build_chunks(pages, current_name)

            if not chunks:
                status.update(
                    label="No chunks created",
                    state="error",
                )
                st.error("No usable text chunks could be created from this PDF.")
                st.stop()

            st.write(f"3/4 Created {len(chunks)} overlapping chunks.")
            model = load_embedding_model()

            st.write("4/4 Creating embeddings and FAISS vector index...")
            index = create_faiss_index(chunks, model)

            st.session_state.faiss_index = index
            st.session_state.chunks = chunks
            st.session_state.document_name = current_name
            st.session_state.page_count = len(pages)
            st.session_state.chunk_count = len(chunks)
            st.session_state.messages = []

            status.update(
                label="RAG knowledge base ready!",
                state="complete",
            )

    if "faiss_index" in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", st.session_state.page_count)
        col2.metric("Chunks", st.session_state.chunk_count)
        col3.metric("Vector dimension", st.session_state.faiss_index.d)

        st.success(
            f"**{st.session_state.document_name}** is ready. Ask questions below."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("Retrieved sources"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**Page {source['page']} — "
                                f"{source['filename']}**  \n"
                                f"Similarity: `{source['score']:.3f}`"
                            )
                            st.caption(source["text"])

        question = st.chat_input("Ask a question about your PDF...")

        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    model = load_embedding_model()

                    with st.spinner("Searching the document..."):
                        retrieved = retrieve_chunks(
                            question,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            model,
                            top_k=top_k,
                        )

                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, retrieved)

                    st.markdown(answer)

                    source_items = [
                        {
                            "filename": item["filename"],
                            "page": item["page"],
                            "score": score,
                            "text": item["text"],
                        }
                        for item, score in retrieved
                    ]

                    if source_items:
                        with st.expander("Retrieved sources"):
                            for source in source_items:
                                st.markdown(
                                    f"**Page {source['page']} — "
                                    f"{source['filename']}**  \n"
                                    f"Similarity: `{source['score']:.3f}`"
                                )
                                st.caption(source["text"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_items,
                        }
                    )

                except Exception as exc:
                    error_message = (
                        "Something went wrong while processing your question. "
                        "Please check the app logs and your GROQ_API_KEY."
                    )
                    st.error(error_message)
                    st.exception(exc)
else:
    st.info("Upload a PDF to start.")

st.divider()
st.caption(
    "Note: FAISS storage in this version is session-based. Uploading a new PDF "
    "builds a new in-memory index. The PDF itself is not permanently stored by this app."
)
