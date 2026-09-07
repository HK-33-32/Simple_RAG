from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

NO_ANSWER = "В загруженных документах недостаточно информации для ответа."

_ROOT = Path(__file__).resolve().parent
PERSIST_DIR = os.getenv("CHROMA_DIR", str(_ROOT / "vector_db"))
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TOP_K = int(os.getenv("TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# separators по убыванию «крупности» границы (RecursiveCharacterTextSplitter).
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты помощник для ответов на вопросы по загруженным PDF.\n"
            "Отвечай исключительно на основании предоставленного контекста.\n"
            "Если предоставленного контекста недостаточно для уверенного ответа, "
            f'ответь точно этой фразой: "{NO_ANSWER}"\n'
            "Не используй собственные знания для дополнения ответа.\n"
            "Note: Please analyze only based on the retrieved content without introducing external knowledge.",
        ),
        (
            "human",
            "Retrieved document:\n\n{context}\n\nUser question: {question}",
        ),
    ]
)


def get_embeddings():
    """BGE через HuggingFaceEmbeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdfs(file_paths: list[str]) -> list[Document]:
    """Каждая страница PDF -> отдельный Document (PyPDFLoader)."""
    pages: list[Document] = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load())
    return pages


def load_files(file_paths: list[str]) -> list[Document]:
    """PDF по страницам; TXT/Markdown — одним документом."""
    docs: list[Document] = []
    for file_path in file_paths:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdfs([file_path]))
            continue
        if suffix in {".md", ".txt"}:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": 0},
                )
            )
    return docs


def split_pages(pages: list[Document]) -> list[Document]:
    """Рекурсивный чанкинг + metadata источника."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = text_splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        page_0 = chunk.metadata.get("page", 0)
        try:
            page_1 = int(page_0) + 1
        except (TypeError, ValueError):
            page_1 = 1
        chunk.metadata["filename"] = Path(str(source)).name
        chunk.metadata["page"] = page_1
        chunk.metadata["chunk_id"] = i
    return chunks


_chroma_clients: dict[str, object] = {}
_chroma_lock = threading.Lock()


def _chroma_client(persist_directory: str):
    """Один PersistentClient на каталог — повторное создание ломает Chroma 1.5 на Windows."""
    import chromadb

    path = str(Path(persist_directory).resolve())
    Path(path).mkdir(parents=True, exist_ok=True)
    with _chroma_lock:
        client = _chroma_clients.get(path)
        if client is not None:
            return client
        client = chromadb.PersistentClient(path=path)
        _chroma_clients[path] = client
        return client


def _delete_collection(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    persist = Path(persist_directory)
    if not persist.exists():
        return
    client = _chroma_client(persist_directory)
    names = {col.name for col in client.list_collections()}
    if collection_name not in names:
        return
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def _get_vectorstore(
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    from langchain_chroma import Chroma

    embeddings = embeddings or get_embeddings()
    client = _chroma_client(persist_directory)
    existing = {col.name for col in client.list_collections()}
    kwargs = {
        "client": client,
        "collection_name": collection_name,
        "embedding_function": embeddings,
    }
    if collection_name not in existing:
        kwargs["collection_metadata"] = {"hnsw:space": "cosine"}
    return Chroma(**kwargs)


def build_vectorstore(
    chunks: list[Document],
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """Пересобрать коллекцию Chroma из чанков (Chroma.from_documents)."""
    embeddings = embeddings or get_embeddings()
    _delete_collection(persist_directory, collection_name)
    store = _get_vectorstore(embeddings, persist_directory, collection_name)
    if chunks:
        store.add_documents(chunks)
    return store


def load_vectorstore(
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    return _get_vectorstore(embeddings, persist_directory, collection_name)


def search(vectorstore, question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Vector similarity search."""
    pairs = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [(doc, float(score)) for doc, score in pairs]


def filter_relevant(
    ranked: list[tuple[Document, float]],
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[tuple[Document, float]]:
    """Упрощённый retrieval assessor из CRAG: порог релевантности вместо LLM-grader и web search."""
    return [(doc, score) for doc, score in ranked if score >= threshold]


def format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{filename}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_llm(api_key: str | None = None, model: str | None = None):
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "Подключите модель Google AI Studio через интерфейс: меню модели -> Подключить API."
        )
    return ChatGoogleGenerativeAI(
        model=(model or GEMINI_MODEL).strip() or GEMINI_MODEL,
        temperature=0,
        google_api_key=key,
    )


def generate_answer(
    question: str,
    docs: list[Document],
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm(api_key=api_key, model=model)
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": format_docs(docs), "question": question}).strip()


@dataclass
class RAGResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved: list[tuple[Document, float]] = field(default_factory=list)
    relevant: list[tuple[Document, float]] = field(default_factory=list)


def ask(
    question: str,
    vectorstore,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    api_key: str | None = None,
    model: str | None = None,
) -> RAGResult:
    ranked = search(vectorstore, question, k=k)
    relevant = filter_relevant(ranked, threshold=threshold)
    if not relevant:
        return RAGResult(answer=NO_ANSWER, retrieved=ranked, relevant=[])
    docs = [doc for doc, _ in relevant]
    answer = generate_answer(question, docs, api_key=api_key, model=model)
    return RAGResult(answer=answer, retrieved=ranked, relevant=relevant)