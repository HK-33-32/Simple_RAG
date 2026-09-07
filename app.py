"""
Endpoints:
    GET    /api/health
    GET    /api/files
    GET    /api/file?name=...
    GET    /api/raw?name=...
    DELETE /api/file?name=...
    POST   /api/ask
    POST   /api/upload
    GET    /api/models
    POST   /api/models/select
    POST   /api/models/connect
    DELETE /api/models?id=...

Run:  uvicorn app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re

mimetypes.add_type("text/javascript", ".mjs")
import threading
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_ext as rag

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
MODELS_PATH = BASE_DIR / "data" / "models.json"
STATIC_DIR = BASE_DIR / "static"
FILE_DIRS = (DOCS_DIR, UPLOADS_DIR)
INDEXABLE = {".pdf", ".md", ".txt"}
VIEWABLE = INDEXABLE | {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DOCS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Simply RAG")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_lock = threading.RLock()
_embeddings = None
_store = None
_indexing = False
_index_owner: str | None = None
_chunks_cached = 0
_indexed_names_cached: set[str] = set()
_index_status: dict = {
    "indexing": False,
    "filename": None,
    "stage": None,
    "current": 0,
    "total": 0,
    "error": None,
}


class AskRequest(BaseModel):
    question: str


class ModelSelectRequest(BaseModel):
    model_id: str


class ModelConnectRequest(BaseModel):
    provider: str
    model: str
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    device: str = "auto"


_models_lock = threading.RLock()
_models_registry: dict = {"active_id": None, "models": []}


def _public_entry(entry: dict) -> dict:
    return {
        "id": entry["id"],
        "selection_id": entry["id"],
        "name": entry.get("name") or entry.get("model") or "Модель",
        "source": entry.get("source") or "Google AI Studio",
        "format": entry.get("format") or "API",
        "size": "",
        "provider": entry.get("provider") or "google",
        "model": entry.get("model") or "",
        "available": bool((entry.get("api_key") or "").strip()),
    }


def _empty_public_model() -> dict:
    return {
        "provider": "google",
        "model": "",
        "name": "Выберите модель",
        "selection_id": "",
        "source": "Google AI Studio",
        "format": "API",
        "available": False,
    }


def _save_models_registry() -> None:
    MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.write_text(
        json.dumps(_models_registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_models_registry() -> None:
    global _models_registry
    if MODELS_PATH.is_file():
        try:
            data = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, list):
            cleaned = []
            for item in models:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                cleaned.append(item)
            active_id = data.get("active_id")
            if active_id not in {item["id"] for item in cleaned}:
                active_id = cleaned[0]["id"] if cleaned else None
            _models_registry = {"active_id": active_id, "models": cleaned}
            return
    _models_registry = {"active_id": None, "models": []}


def _active_entry() -> dict | None:
    active_id = _models_registry.get("active_id")
    for item in _models_registry.get("models") or []:
        if item.get("id") == active_id:
            return item
    return None


def _public_model() -> dict:
    entry = _active_entry()
    if entry is None:
        return _empty_public_model()
    return _public_entry(entry)


def _public_models() -> list[dict]:
    return [_public_entry(item) for item in _models_registry.get("models") or []]


_load_models_registry()


def _embeddings_cached():
    global _embeddings
    if _embeddings is None:
        _embeddings = rag.get_embeddings()
    return _embeddings


def get_store():
    global _store
    with _lock:
        if _store is None:
            _store = rag.load_vectorstore(embeddings=_embeddings_cached())
        return _store


def _apply_store_snapshot(store) -> None:
    global _chunks_cached, _indexed_names_cached
    _chunks_cached, _indexed_names_cached = rag.store_snapshot(store)


def _refresh_index_cache() -> None:
    global _chunks_cached, _indexed_names_cached
    try:
        files = rag.list_indexed_files()
    except Exception:
        return
    _indexed_names_cached = {item["filename"] for item in files}
    _chunks_cached = sum(item["chunks"] for item in files)


def _chunk_count() -> int:
    if _indexing:
        return _chunks_cached
    _refresh_index_cache()
    return _chunks_cached


def _indexed_names() -> set[str]:
    if _indexing:
        return set(_indexed_names_cached)
    _refresh_index_cache()
    return set(_indexed_names_cached)


def _set_index_status(**kwargs) -> None:
    _index_status.update(kwargs)


def _index_progress(stage: str, current: int, total: int) -> None:
    _set_index_status(stage=stage, current=current, total=total)


def _find_file(name: str) -> Path | None:
    if not name or Path(name).name != name or name in {".", ".."}:
        return None
    for directory in FILE_DIRS:
        path = directory / name
        if path.is_file():
            return path
    return None


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _coerce_page(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


@app.get("/api/health")
def health():
    model = _public_model()
    embed = rag.embedding_info()
    return {
        "status": "ok",
        "chunks_in_store": _chunk_count(),
        "indexing": _indexing,
        "index": {
            "filename": _index_status["filename"],
            "stage": _index_status["stage"],
            "current": _index_status["current"],
            "total": _index_status["total"],
            "error": _index_status["error"],
            "device": embed["device"],
        },
        "llm_loaded": bool((_active_entry() or {}).get("api_key")),
        "embedding_model": rag.EMBEDDING_MODEL,
        "embedding_device": embed["device"],
        "embedding_gpu": embed.get("gpu"),
        "active_model": model,
        "top_k": rag.TOP_K,
        "min_score": rag.RELEVANCE_THRESHOLD,
    }


@app.get("/api/models")
def list_models():
    with _models_lock:
        return {"active_model": _public_model(), "models": _public_models()}


@app.get("/api/models/discover")
def discover_models():
    with _models_lock:
        return {"models": _public_models()}


@app.post("/api/models/select")
def select_model(req: ModelSelectRequest):
    with _models_lock:
        match = next((item for item in _models_registry["models"] if item.get("id") == req.model_id), None)
        if match is None:
            raise HTTPException(404, "Модель не найдена")
        _models_registry["active_id"] = match["id"]
        _save_models_registry()
        public = _public_entry(match)
        return {"active_model": public, "model": public, "loaded": True}


@app.post("/api/models/connect")
def connect_model(req: ModelConnectRequest):
    provider = (req.provider or "").strip().lower()
    if provider in {"gemini"}:
        provider = "google"
    if provider != "google":
        raise HTTPException(400, "Подключите Google AI Studio.")
    api_key = (req.api_key or "").strip()
    if not api_key:
        raise HTTPException(400, "Укажите ключ Google AI Studio.")
    model_name = (req.model or rag.GEMINI_MODEL).strip() or rag.GEMINI_MODEL
    display_name = (req.name or model_name).strip() or model_name
    entry_id = f"google::{model_name}"
    entry = {
        "id": entry_id,
        "name": display_name,
        "provider": "google",
        "model": model_name,
        "source": "Google AI Studio",
        "format": "API",
        "api_key": api_key,
        "base_url": (req.base_url or "").strip(),
    }
    with _models_lock:
        existing = next((item for item in _models_registry["models"] if item.get("id") == entry_id), None)
        if existing is None:
            _models_registry["models"].append(entry)
        else:
            existing.update(entry)
        _models_registry["active_id"] = entry_id
        _save_models_registry()
        public = _public_entry(entry if existing is None else existing)
        return {"active_model": public, "model": public, "models": _public_models(), "loaded": True}


@app.delete("/api/models")
def delete_model(id: str):
    model_id = (id or "").strip()
    if not model_id:
        raise HTTPException(400, "Не указана модель")
    with _models_lock:
        before = len(_models_registry["models"])
        _models_registry["models"] = [item for item in _models_registry["models"] if item.get("id") != model_id]
        if len(_models_registry["models"]) == before:
            raise HTTPException(404, "Модель не найдена")
        if _models_registry.get("active_id") == model_id:
            _models_registry["active_id"] = (
                _models_registry["models"][0]["id"] if _models_registry["models"] else None
            )
        _save_models_registry()
        return {
            "deleted": model_id,
            "active_model": _public_model(),
            "models": _public_models(),
        }


@app.get("/api/files")
def list_files():
    indexed = _indexed_names()
    files = []
    seen: set[str] = set()
    for directory in FILE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIEWABLE:
                continue
            if path.name in seen:
                continue
            seen.add(path.name)
            files.append(
                {
                    "name": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size": path.stat().st_size,
                    "in_corpus": path.name in indexed,
                }
            )
    return files


@app.get("/api/file")
def get_file(name: str):
    path = _find_file(name)
    if path is None:
        raise HTTPException(404, f"Файл не найден: {name}")
    suffix = path.suffix.lower()
    return {
        "name": name,
        "type": suffix.lstrip("."),
        "text": None
        if suffix in IMAGE_TYPES | {".pdf"}
        else path.read_text(encoding="utf-8", errors="replace"),
        "url": f"/api/raw?name={quote(name)}",
    }


@app.get("/api/raw")
def raw_file(name: str):
    path = _find_file(name)
    if path is None:
        raise HTTPException(404, f"Файл не найден: {name}")
    from fastapi.responses import FileResponse

    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, media_type=media_type)


@app.delete("/api/file")
def delete_file(name: str):
    global _store
    if _indexing:
        raise HTTPException(409, "Дождитесь окончания индексации")
    path = _find_file(name)
    if path is None:
        raise HTTPException(404, f"Файл не найден: {name}")
    path.unlink()
    with _lock:
        removed_chunks = rag.remove_indexed_file(name)
        _store = None
        _refresh_index_cache()
    return {"deleted": name, "removed_chunks": removed_chunks}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "empty question")
    if _indexing:
        raise HTTPException(409, "Идёт индексация документа. Повторите вопрос через несколько секунд.")
    active = _active_entry()
    if not active or not (active.get("api_key") or "").strip():
        raise HTTPException(
            503,
            "Подключите модель Google AI Studio: меню модели -> Подключить API.",
        )
    try:
        with _lock:
            store = get_store()
            if store is None:
                return {
                    "question": question,
                    "answer": rag.NO_ANSWER,
                    "model": _public_model(),
                    "chunks": [],
                    "n_context": 0,
                }
            result = rag.ask(
                question,
                store,
                api_key=active.get("api_key"),
                model=active.get("model"),
            )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    chunks = []
    for index, (doc, score) in enumerate(result.relevant or result.retrieved, start=1):
        content = doc.page_content
        source_name = str(doc.metadata.get("filename") or Path(str(doc.metadata.get("source", ""))).name)
        page = _coerce_page(doc.metadata.get("page"))
        quote = _normalize_text(content)[:280]
        chunks.append(
            {
                "index": index,
                "source": source_name,
                "title": Path(source_name).stem or source_name,
                "score": round(float(score), 4),
                "snippet": " ".join(content.split())[:300],
                "chunk_text": content,
                "page": page,
                "quote": quote,
                "start_index": doc.metadata.get("start_index"),
            }
        )
    if result.answer == rag.NO_ANSWER:
        chunks = []
    return {
        "question": question,
        "answer": result.answer.strip(),
        "model": _public_model(),
        "chunks": chunks,
        "n_context": len(chunks),
    }


def _ingest_file(path: str, name: str):
    global _store
    with _lock:
        _store = None
        _set_index_status(filename=name, stage="model", current=0, total=0, error=None)
        result = rag.ingest_pdfs(
            [path],
            embeddings=_embeddings_cached(),
            progress=_index_progress,
        )
        _store = result.store
        _apply_store_snapshot(result.store)
        return result


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    global _indexing, _index_owner
    name = re.sub(r"[^\w.\- ]", "", file.filename or "upload")
    suffix = Path(name).suffix.lower()
    if suffix not in VIEWABLE:
        raise HTTPException(400, f"unsupported type: {suffix or '(none)'}")
    target = UPLOADS_DIR / name
    target.write_bytes(await file.read())
    kind = "image" if suffix in IMAGE_TYPES else "document"
    chunks_total = _chunks_cached if _indexing else _chunk_count()
    if suffix in INDEXABLE:
        owner = f"{name}:{id(target)}"
        _index_owner = owner
        _indexing = True
        _set_index_status(
            indexing=True,
            filename=name,
            stage="model",
            current=0,
            total=0,
            error=None,
        )
        try:
            await asyncio.to_thread(_ingest_file, str(target), name)
            chunks_total = _chunks_cached
        except Exception as exc:
            _set_index_status(error=str(exc))
            raise HTTPException(500, f"Не удалось проиндексировать документ: {exc}") from exc
        finally:
            if _index_owner == owner:
                _indexing = False
                _index_owner = None
                _set_index_status(indexing=False, stage=None)
    return {
        "kind": kind,
        "name": name,
        "chunks_total": chunks_total,
        "url": f"/api/raw?name={quote(name)}",
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
