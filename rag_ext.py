"""Доработки Simply RAG, которых нет в книге.

GPU для эмбеддингов, инкрементальная индексация, прогресс, снимок Chroma
без второго клиента, список/удаление файлов, источники в ответе.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    NO_ANSWER,
    PERSIST_DIR,
    RAGResult,
    RELEVANCE_THRESHOLD,
    TOP_K,
    _chroma_client,
    _delete_collection,
    _get_vectorstore,
    ask as _book_ask,
    load_files,
    split_pages,
)

ProgressCallback = Callable[[str, int, int], None]

_embedding_device: str | None = None


def embedding_device() -> str:
    """cuda, если доступен GPU-сборкой PyTorch, иначе cpu."""
    global _embedding_device
    if _embedding_device is None:
        try:
            import torch

            _embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _embedding_device = "cpu"
    return _embedding_device


def embedding_batch_size(device: str | None = None) -> int:
    override = os.getenv("EMBED_BATCH_SIZE")
    if override:
        return max(1, int(override))
    return 32 if (device or embedding_device()) == "cuda" else 8


def embedding_info() -> dict[str, Any]:
    device = embedding_device()
    info: dict[str, Any] = {
        "device": device,
        "model": EMBEDDING_MODEL,
        "batch_size": embedding_batch_size(device),
    }
    if device == "cuda":
        try:
            import torch

            info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def _hf_model_kwargs() -> dict:
    """Токен Hub или только локальный кэш, чтобы не ходить в сеть без необходимости."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return {"token": token}
    hub = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / f"models--{EMBEDDING_MODEL.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir() and any(snapshots.iterdir()):
        return {"local_files_only": True}
    return {}


def get_embeddings():
    """BGE-M3 на GPU при CUDA, иначе CPU."""
    from langchain_huggingface import HuggingFaceEmbeddings

    device = embedding_device()
    model_kwargs: dict[str, Any] = {**_hf_model_kwargs(), "device": device}
    if device == "cuda":
        import torch

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["model_kwargs"] = {"torch_dtype": dtype}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": embedding_batch_size(device),
        },
        model_kwargs=model_kwargs,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_names(persist_directory: str) -> set[str]:
    persist = Path(persist_directory)
    if not persist.exists():
        return set()
    try:
        client = _chroma_client(persist_directory)
        return {col.name for col in client.list_collections()}
    except Exception:
        return set()


def list_indexed_files(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Проиндексированные файлы из Chroma без загрузки embedding-модели."""
    try:
        if collection_name not in _collection_names(persist_directory):
            return []
        client = _chroma_client(persist_directory)
        collection = client.get_collection(collection_name)
        if collection.count() == 0:
            return []
        data = collection.get(include=["metadatas"])
    except Exception:
        return []
    files: dict[str, dict] = {}
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        name = str(meta.get("filename") or Path(str(meta.get("source", ""))).name or "unknown")
        info = files.setdefault(
            name,
            {
                "filename": name,
                "chunks": 0,
                "pages": set(),
                "file_hash": meta.get("file_hash"),
            },
        )
        info["chunks"] += 1
        if meta.get("file_hash"):
            info["file_hash"] = meta.get("file_hash")
        page = meta.get("page")
        if page is not None:
            try:
                info["pages"].add(int(page))
            except (TypeError, ValueError):
                pass
    result = []
    for info in files.values():
        result.append(
            {
                "filename": info["filename"],
                "chunks": info["chunks"],
                "pages": len(info["pages"]),
                "file_hash": info.get("file_hash"),
            }
        )
    return sorted(result, key=lambda item: item["filename"].lower())


def _delete_by_filename(
    persist_directory: str,
    filename: str,
    collection_name: str = COLLECTION_NAME,
) -> None:
    if collection_name not in _collection_names(persist_directory):
        return
    client = _chroma_client(persist_directory)
    collection = client.get_collection(collection_name)
    collection.delete(where={"filename": filename})


def clear_vectorstore(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    _delete_collection(persist_directory, collection_name)


def remove_indexed_file(
    filename: str,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Удалить чанки одного файла. Возвращает число удалённых фрагментов."""
    match = next(
        (item for item in list_indexed_files(persist_directory, collection_name) if item["filename"] == filename),
        None,
    )
    _delete_by_filename(persist_directory, filename, collection_name)
    return int(match["chunks"]) if match else 0


def load_vectorstore(
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    if not list_indexed_files(persist_directory, collection_name):
        return None
    return _get_vectorstore(embeddings or get_embeddings(), persist_directory, collection_name)


def store_snapshot(store) -> tuple[int, set[str]]:
    """Число чанков и имена файлов из уже открытого store — без нового клиента Chroma."""
    if store is None:
        return 0, set()
    collection = getattr(store, "_collection", None)
    if collection is None:
        return 0, set()
    count = int(collection.count())
    if count == 0:
        return 0, set()
    data = collection.get(include=["metadatas"])
    names: set[str] = set()
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        names.add(str(meta.get("filename") or Path(str(meta.get("source", ""))).name or "unknown"))
    return count, names


def _add_documents_batched(
    store,
    chunks,
    progress: ProgressCallback | None = None,
    batch_size: int | None = None,
) -> None:
    total = len(chunks)
    if total == 0:
        return
    size = max(1, batch_size or embedding_batch_size())
    if progress:
        progress("embed", 0, total)
    for start in range(0, total, size):
        batch = chunks[start : start + size]
        store.add_documents(batch)
        if progress:
            progress("embed", min(start + size, total), total)


@dataclass
class IngestResult:
    store: object
    n_pages: int
    n_chunks: int
    added_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.store
        yield self.n_pages
        yield self.n_chunks


def ingest_pdfs(
    file_paths: list[str],
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    replace: bool = False,
    progress: ProgressCallback | None = None,
) -> IngestResult:
    """Индексация с пропуском уже обработанных файлов и отчётом о прогрессе."""
    def report(stage: str, current: int = 0, total: int = 0) -> None:
        if progress:
            progress(stage, current, total)

    report("model")
    embeddings = embeddings or get_embeddings()
    if replace:
        _delete_collection(persist_directory, collection_name)

    indexed = {
        item["filename"]: item.get("file_hash")
        for item in list_indexed_files(persist_directory, collection_name)
    }

    to_load: list[str] = []
    skipped: list[str] = []
    hashes: dict[str, str] = {}
    for path in file_paths:
        name = Path(path).name
        digest = file_sha256(path)
        hashes[name] = digest
        if name in indexed:
            existing_hash = indexed.get(name)
            if existing_hash and existing_hash != digest:
                _delete_by_filename(persist_directory, name, collection_name)
                to_load.append(path)
            else:
                skipped.append(name)
            continue
        to_load.append(path)

    store = _get_vectorstore(embeddings, persist_directory, collection_name)
    if not to_load:
        return IngestResult(
            store=store,
            n_pages=0,
            n_chunks=0,
            added_files=[],
            skipped_files=skipped,
        )

    report("parse")
    pages = load_files(to_load)
    report("split", 0, len(pages))
    chunks = split_pages(pages)
    for chunk in chunks:
        name = chunk.metadata.get("filename")
        if name in hashes:
            chunk.metadata["file_hash"] = hashes[name]
    _add_documents_batched(store, chunks, progress=progress)
    return IngestResult(
        store=store,
        n_pages=len(pages),
        n_chunks=len(chunks),
        added_files=[Path(p).name for p in to_load],
        skipped_files=skipped,
    )


def sources_from_docs(docs) -> list[dict]:
    """Источники из metadata retriever, не из LLM."""
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []
    for doc in docs:
        filename = str(doc.metadata.get("filename", "unknown"))
        page = int(doc.metadata.get("page", 0) or 0)
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"filename": filename, "page": page})
    return sources


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["Источники:"]
    for item in sources:
        lines.append(f"• {item['filename']} — стр. {item['page']}")
    return "\n".join(lines)


def ask(
    question: str,
    vectorstore,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    api_key: str | None = None,
    model: str | None = None,
) -> RAGResult:
    result = _book_ask(
        question,
        vectorstore,
        k=k,
        threshold=threshold,
        api_key=api_key,
        model=model,
    )
    if result.answer == NO_ANSWER:
        return result
    result.sources = sources_from_docs([doc for doc, _ in result.relevant])
    return result