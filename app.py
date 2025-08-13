import io
import streamlit as st

# PDF reading
try:
    from pypdf import PdfReader  # preferred
except Exception:
    from PyPDF2 import PdfReader  # fallback if you installed PyPDF2

# LangChain bits (community imports for embeddings/vectorstores/llm backends)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG PDF Chat • Ollama + FAISS", page_icon="📄")
st.title("📄 RAG Chatbot (PDF) — FAISS + LLaMA (Ollama)")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def build_faiss_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text found in the PDF. If it's scanned, run OCR to make text searchable.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

def build_qa(db, model_name="llama3.2"):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model=model_name)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Ollama model", ["llama3.2", "llama3.1:8b", "llama3.1:70b"], index=0)
    show_sources = st.checkbox("Show source chunks", value=True)
    st.markdown(
        "- Ensure Ollama is running (pull a model first).\n"
        "- First answer may take a few seconds while the model warms up."
    )

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.read()

with st.spinner("Extracting text & building FAISS index..."):
    try:
        text = extract_text_from_pdf_bytes(pdf_bytes)
        db = build_faiss_from_text(text)
    except Exception as e:
        st.error(f"Index build failed: {e}")
        st.stop()

st.success("Index ready. Ask your question below.")

question = st.text_input("Your question about the document:")
if st.button("Ask", disabled=not question):
    with st.spinner("Thinking..."):
        qa = build_qa(db, model_name=model_name)
        result = qa({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

    st.subheader("Answer")
    st.write(answer)

    if show_sources and sources:
        st.subheader("Sources")
        for i, doc in enumerate(sources, start=1):
            st.markdown(f"**Chunk {i}**")
            st.code((doc.page_content or "").strip()[:1000])
