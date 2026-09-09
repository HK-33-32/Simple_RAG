from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    NO_ANSWER,
    OPENAI_MODEL,
    PERSIST_DIR,
    PROVIDERS,
    RAGResult,
    RELEVANCE_THRESHOLD,
    TOP_K,
    _chroma_client,
    _delete_collection,
    _get_vectorstore,
    ask as _book_ask,
    llm_configured,
    load_files,
    normalize_provider,
    split_pages,
)

ProgressCallback = Callable[[str, int, int], None]

STATUS_READY = "ready"
STATUS_INDEXING = "indexing"
STATUS_FAILED = "failed"
DOCUMENT_STATUSES = (STATUS_READY, STATUS_INDEXING, STATUS_FAILED)
MANIFEST_NAME = "documents.json"
_DELETE_ID_BATCH = 500

_embedding_device: str | None = None
_manifest_lock = threading.Lock()


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


def document_is_complete(record: dict | None, digest: str) -> bool:
    """Пропуск только если запись явно ready и хеш совпал."""
    if not record or not digest:
        return False
    return record.get("status") == STATUS_READY and record.get("file_hash") == digest


def begin_document_ingest(record: dict | None, filename: str, digest: str) -> dict:
    previous = None
    generation = 1
    if record:
        generation = int(record.get("generation") or 0) + 1
        if record.get("status") == STATUS_READY:
            previous = {
                "file_hash": record.get("file_hash"),
                "generation": record.get("generation") or 1,
                "chunks": int(record.get("chunks") or 0),
                "pages": int(record.get("pages") or 0),
            }
    return {
        "filename": filename,
        "file_hash": digest,
        "status": STATUS_INDEXING,
        "generation": generation,
        "chunks": 0,
        "pages": 0,
        "expected_chunks": 0,
        "error": None,
        "previous": previous,
    }


def commit_document_ingest(record: dict, *, chunks: int, pages: int) -> dict:
    return {
        "filename": record["filename"],
        "file_hash": record.get("file_hash"),
        "status": STATUS_READY,
        "generation": record.get("generation") or 1,
        "chunks": int(chunks),
        "pages": int(pages),
        "expected_chunks": int(chunks),
        "error": None,
        "previous": None,
    }


def rollback_document_ingest(record: dict, error: str) -> dict:
    previous = record.get("previous")
    if previous:
        return {
            "filename": record["filename"],
            "file_hash": previous.get("file_hash"),
            "status": STATUS_READY,
            "generation": previous.get("generation") or 1,
            "chunks": int(previous.get("chunks") or 0),
            "pages": int(previous.get("pages") or 0),
            "expected_chunks": int(previous.get("chunks") or 0),
            "error": None,
            "previous": None,
        }
    return {
        "filename": record["filename"],
        "file_hash": record.get("file_hash"),
        "status": STATUS_FAILED,
        "generation": record.get("generation") or 1,
        "chunks": 0,
        "pages": 0,
        "expected_chunks": 0,
        "error": error,
        "previous": None,
    }


def chunk_ids_to_drop(
    ids: list[str],
    metas: list[dict | None],
    *,
    drop_generation: int | None = None,
    keep_generation: int | None = None,
) -> list[str]:
    dropped: list[str] = []
    for chunk_id, meta in zip(ids, metas):
        generation = None if not meta else meta.get("generation")
        if drop_generation is not None:
            if generation == drop_generation:
                dropped.append(chunk_id)
        elif keep_generation is not None:
            if generation != keep_generation:
                dropped.append(chunk_id)
        else:
            dropped.append(chunk_id)
    return dropped


def _manifest_path(persist_directory: str) -> Path:
    return Path(persist_directory) / MANIFEST_NAME


def _empty_manifest() -> dict:
    return {"documents": {}}


def _load_manifest(persist_directory: str) -> dict:
    path = _manifest_path(persist_directory)
    if not path.is_file():
        return _empty_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_manifest()
    documents = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(documents, dict):
        return _empty_manifest()
    cleaned = {}
    for name, record in documents.items():
        if isinstance(name, str) and isinstance(record, dict):
            cleaned[name] = record
    return {"documents": cleaned}


def _save_manifest(data: dict, persist_directory: str) -> None:
    path = _manifest_path(persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _put_record(record: dict, persist_directory: str) -> None:
    with _manifest_lock:
        data = _load_manifest(persist_directory)
        data["documents"][record["filename"]] = record
        _save_manifest(data, persist_directory)


def _drop_record(filename: str, persist_directory: str) -> None:
    with _manifest_lock:
        data = _load_manifest(persist_directory)
        if filename in data["documents"]:
            del data["documents"][filename]
            _save_manifest(data, persist_directory)


def _clear_manifest(persist_directory: str) -> None:
    with _manifest_lock:
        _save_manifest(_empty_manifest(), persist_directory)


def _legacy_record_from_chroma(item: dict) -> dict:
    chunks = int(item.get("chunks") or 0)
    return {
        "filename": item["filename"],
        "file_hash": item.get("file_hash"),
        "status": STATUS_READY,
        "generation": int(item.get("generation") or 1),
        "chunks": chunks,
        "pages": int(item.get("pages") or 0),
        "expected_chunks": chunks,
        "error": None,
        "previous": None,
    }


def _public_file_info(record: dict, chroma: dict | None = None) -> dict:
    chroma = chroma or {}
    status = record.get("status") if record.get("status") in DOCUMENT_STATUSES else STATUS_READY
    live = record.get("previous") if status == STATUS_INDEXING and record.get("previous") else record
    chunks = int(live.get("chunks") or chroma.get("chunks") or 0)
    pages = int(live.get("pages") or chroma.get("pages") or 0)
    return {
        "filename": record["filename"],
        "chunks": chunks,
        "pages": pages,
        "file_hash": live.get("file_hash") or chroma.get("file_hash"),
        "status": status,
        "generation": live.get("generation") or chroma.get("generation"),
        "error": record.get("error"),
    }


def _collection_names(persist_directory: str) -> set[str]:
    persist = Path(persist_directory)
    if not persist.exists():
        return set()
    try:
        client = _chroma_client(persist_directory)
        return {col.name for col in client.list_collections()}
    except Exception:
        return set()


def _chroma_file_stats(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, dict]:
    """Чанки/страницы/хеш по файлам из Chroma без embedding-модели."""
    try:
        if collection_name not in _collection_names(persist_directory):
            return {}
        client = _chroma_client(persist_directory)
        collection = client.get_collection(collection_name)
        if collection.count() == 0:
            return {}
        data = collection.get(include=["metadatas"])
    except Exception:
        return {}
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
                "generation": meta.get("generation"),
            },
        )
        info["chunks"] += 1
        if meta.get("file_hash"):
            info["file_hash"] = meta.get("file_hash")
        if meta.get("generation") is not None:
            info["generation"] = meta.get("generation")
        page = meta.get("page")
        if page is not None:
            try:
                info["pages"].add(int(page))
            except (TypeError, ValueError):
                pass
    result: dict[str, dict] = {}
    for info in files.values():
        result[info["filename"]] = {
            "filename": info["filename"],
            "chunks": info["chunks"],
            "pages": len(info["pages"]),
            "file_hash": info.get("file_hash"),
            "generation": info.get("generation"),
        }
    return result


def list_indexed_files(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Файлы с статусом ready/indexing/failed."""
    chroma = _chroma_file_stats(persist_directory, collection_name)
    manifest = _load_manifest(persist_directory)["documents"]
    names = set(chroma) | set(manifest)
    result = []
    for name in names:
        record = manifest.get(name)
        stats = chroma.get(name)
        if record is None and stats is not None:
            record = _legacy_record_from_chroma(stats)
        if record is None:
            continue
        result.append(_public_file_info(record, stats))
    return sorted(result, key=lambda item: item["filename"].lower())


def _file_chunk_pairs(
    persist_directory: str,
    filename: str,
    collection_name: str = COLLECTION_NAME,
) -> list[tuple[str, dict | None]]:
    if collection_name not in _collection_names(persist_directory):
        return []
    client = _chroma_client(persist_directory)
    collection = client.get_collection(collection_name)
    data = collection.get(where={"filename": filename}, include=["metadatas"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    return list(zip(ids, metas))


def _delete_ids(
    persist_directory: str,
    ids: list[str],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not ids or collection_name not in _collection_names(persist_directory):
        return
    client = _chroma_client(persist_directory)
    collection = client.get_collection(collection_name)
    for start in range(0, len(ids), _DELETE_ID_BATCH):
        collection.delete(ids=ids[start : start + _DELETE_ID_BATCH])


def _delete_by_filename(
    persist_directory: str,
    filename: str,
    collection_name: str = COLLECTION_NAME,
    *,
    drop_generation: int | None = None,
    keep_generation: int | None = None,
) -> None:
    pairs = _file_chunk_pairs(persist_directory, filename, collection_name)
    if not pairs:
        return
    ids, metas = zip(*pairs)
    to_drop = chunk_ids_to_drop(
        list(ids),
        list(metas),
        drop_generation=drop_generation,
        keep_generation=keep_generation,
    )
    _delete_ids(persist_directory, to_drop, collection_name)


def clear_vectorstore(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    _delete_collection(persist_directory, collection_name)
    _clear_manifest(persist_directory)


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
    _drop_record(filename, persist_directory)
    return int(match["chunks"]) if match else 0


def load_vectorstore(
    embeddings=None,
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    if not any(item.get("status") == STATUS_READY for item in list_indexed_files(persist_directory, collection_name)):
        return None
    return _get_vectorstore(embeddings or get_embeddings(), persist_directory, collection_name)


def recover_incomplete_documents(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """После обрыва процесса: indexing → failed, либо откат к предыдущей ready-версии."""
    chroma = _chroma_file_stats(persist_directory, collection_name)
    with _manifest_lock:
        data = _load_manifest(persist_directory)
        changed = False
        for name, stats in chroma.items():
            if name not in data["documents"]:
                data["documents"][name] = _legacy_record_from_chroma(stats)
                changed = True
        recovered = []
        for name, record in list(data["documents"].items()):
            if record.get("status") != STATUS_INDEXING:
                continue
            generation = record.get("generation")
            if record.get("previous"):
                _delete_by_filename(
                    persist_directory,
                    name,
                    collection_name,
                    drop_generation=generation,
                )
            else:
                _delete_by_filename(persist_directory, name, collection_name)
            restored = rollback_document_ingest(record, "Индексация прервана")
            data["documents"][name] = restored
            recovered.append(restored)
            changed = True
        if changed:
            _save_manifest(data, persist_directory)
    return recovered


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
    filenames: dict[str, str] | None = None,
) -> IngestResult:
    """Индексация с пропуском только fully-ready файлов."""
    def report(stage: str, current: int = 0, total: int = 0) -> None:
        if progress:
            progress(stage, current, total)

    report("model")
    embeddings = embeddings or get_embeddings()
    if replace:
        _delete_collection(persist_directory, collection_name)
        _clear_manifest(persist_directory)

    chroma = _chroma_file_stats(persist_directory, collection_name)
    manifest = _load_manifest(persist_directory)["documents"]
    names = filenames or {}
    skipped: list[str] = []
    jobs: list[tuple[str, str, str, dict | None]] = []
    for path in file_paths:
        name = names.get(path) or Path(path).name
        digest = file_sha256(path)
        record = manifest.get(name)
        if record is None and name in chroma:
            record = _legacy_record_from_chroma(chroma[name])
        if document_is_complete(record, digest):
            skipped.append(name)
            continue
        jobs.append((path, name, digest, record))

    store = _get_vectorstore(embeddings, persist_directory, collection_name)
    if not jobs:
        return IngestResult(
            store=store,
            n_pages=0,
            n_chunks=0,
            added_files=[],
            skipped_files=skipped,
        )

    added: list[str] = []
    total_pages = 0
    total_chunks = 0
    for path, name, digest, record in jobs:
        pending = begin_document_ingest(record, name, digest)
        _put_record(pending, persist_directory)
        try:
            report("parse")
            pages = load_files([path])
            for page in pages:
                page.metadata["source"] = name
                page.metadata["filename"] = name
            report("split", 0, len(pages))
            chunks = split_pages(pages)
            for chunk in chunks:
                chunk.metadata["filename"] = name
                chunk.metadata["file_hash"] = digest
                chunk.metadata["generation"] = pending["generation"]
            pending["expected_chunks"] = len(chunks)
            pending["pages"] = len(pages)
            _put_record(pending, persist_directory)
            _add_documents_batched(store, chunks, progress=progress)
            if pending.get("previous"):
                _delete_by_filename(
                    persist_directory,
                    name,
                    collection_name,
                    keep_generation=pending["generation"],
                )
            ready = commit_document_ingest(pending, chunks=len(chunks), pages=len(pages))
            _put_record(ready, persist_directory)
            added.append(name)
            total_pages += len(pages)
            total_chunks += len(chunks)
        except Exception as exc:
            if pending.get("previous"):
                _delete_by_filename(
                    persist_directory,
                    name,
                    collection_name,
                    drop_generation=pending["generation"],
                )
            else:
                _delete_by_filename(persist_directory, name, collection_name)
            _put_record(rollback_document_ingest(pending, str(exc)), persist_directory)
            raise
    return IngestResult(
        store=store,
        n_pages=total_pages,
        n_chunks=total_chunks,
        added_files=added,
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
    provider: str | None = None,
    base_url: str | None = None,
) -> RAGResult:
    result = _book_ask(
        question,
        vectorstore,
        k=k,
        threshold=threshold,
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=base_url,
    )
    if result.answer == NO_ANSWER:
        return result
    result.sources = sources_from_docs([doc for doc, _ in result.relevant])
    return result