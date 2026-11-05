"""Dashboard helpers for running ingestion and document deletion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from core.settings import load_settings
from core.runtime import build_pipeline as build_configured_pipeline
from core.runtime import runtime_settings
from core.trace import TraceContext
from ingestion.document_manager import DeleteResult, DocumentManager
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.pipeline import IngestionPipeline, IngestionResult, ProgressCallback
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.embedding.base_embedding import BaseEmbedding
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store import BaseVectorStore
from libs.vector_store.chroma_store import ChromaStore
from observability.logger import DEFAULT_TRACE_LOG_PATH, write_trace


@dataclass(frozen=True)
class UploadTarget:
    """Saved upload location used as the pipeline source path."""

    path: Path
    filename: str
    collection: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "filename": self.filename,
            "collection": self.collection,
        }


class IngestionService:
    """Coordinate dashboard uploads, ingestion runs, and document deletion."""

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        document_manager: DocumentManager,
        upload_root: str | Path,
        trace_log_path: str | Path = DEFAULT_TRACE_LOG_PATH,
    ) -> None:
        self.pipeline = pipeline
        self.document_manager = document_manager
        self.upload_root = Path(upload_root)
        self.trace_log_path = Path(trace_log_path)

    @classmethod
    def from_settings(
        cls,
        settings_path: str | Path = "config/settings.yaml",
        *,
        data_dir: str | Path = "data",
    ) -> "IngestionService":
        settings = load_settings(settings_path)
        data_root = Path(data_dir)
        db_dir = data_root / "db"

        bm25 = BM25Indexer(db_dir / "bm25")
        if bm25.index_path.exists():
            bm25.load()
        image_storage = ImageStorage(
            image_root=data_root / "images",
            db_path=db_dir / "image_index.db",
        )
        integrity = SQLiteIntegrityChecker(db_dir / "ingestion_history.db")

        pipeline = build_configured_pipeline(settings, data_root)
        manager = DocumentManager(
            chroma_store=ChromaStore(runtime_settings(settings, data_root)["vector_store"]),
            bm25_indexer=bm25,
            image_storage=image_storage,
            file_integrity=integrity,
        )
        return cls(
            pipeline=pipeline,
            document_manager=manager,
            upload_root=data_root / "uploads",
            trace_log_path=DEFAULT_TRACE_LOG_PATH,
        )

    def save_upload(
        self,
        filename: str,
        content: bytes,
        *,
        collection: str,
    ) -> UploadTarget:
        """Persist an uploaded PDF under data/uploads/{collection}/."""
        safe_collection = _safe_segment(collection, default="default")
        safe_filename = _safe_filename(filename)
        if not isinstance(content, bytes) or not content:
            raise ValueError("uploaded file content must be non-empty bytes")
        if Path(safe_filename).suffix.lower() != ".pdf":
            raise ValueError("only PDF uploads are supported")

        target_dir = self.upload_root / safe_collection
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = _dedupe_path(target_dir / safe_filename)
        target_path.write_bytes(content)
        return UploadTarget(
            path=target_path,
            filename=safe_filename,
            collection=safe_collection,
        )

    def ingest_file(
        self,
        source_path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Run the ingestion pipeline and return a dashboard-friendly payload."""
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string")
        clean_collection = collection.strip()
        trace = TraceContext(trace_type="ingestion")
        try:
            result = self.pipeline.run(
                source_path,
                collection=clean_collection,
                force=force,
                trace=trace,
                on_progress=on_progress,
            )
        except Exception as exc:
            trace.finish()
            payload = trace.to_dict()
            payload.update(
                {
                    "source_path": str(source_path),
                    "collection": clean_collection,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            write_trace(payload, self.trace_log_path)
            raise

        trace.finish()
        payload = trace.to_dict()
        payload.update(
            {
                "source_path": result.source_path,
                "collection": result.collection,
                "status": "success",
                "skipped": result.skipped,
                "chunk_count": result.chunk_count,
                "image_count": len(result.image_records),
            }
        )
        write_trace(payload, self.trace_log_path)
        return _ingestion_result_payload(result)

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        return [document.to_dict() for document in self.document_manager.list_documents(collection)]

    def delete_document(
        self,
        source_path: str,
        collection: str | None = None,
    ) -> dict[str, Any]:
        result = self.document_manager.delete_document(source_path, collection)
        return _delete_result_payload(result)


class LocalDeterministicEmbedding(BaseEmbedding):
    """Small offline embedding provider for dashboard-triggered ingestion."""

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(
                [
                    float(len(text)),
                    float(sum(ord(char) for char in text) % 997),
                    float(len(text.split())),
                ]
            )
        return vectors


class JsonVectorStore(BaseVectorStore):
    """Idempotent JSON-backed vector store for local dashboard ingestion."""

    def __init__(self, path: str | Path) -> None:
        super().__init__({})
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, records: list[Mapping[str, Any]], trace=None) -> None:
        self.validate_records(records)
        existing = self._load_records()
        for record in records:
            existing[str(record["id"])] = {
                "id": str(record["id"]),
                "text": str(record.get("text", "")),
                "vector": [float(value) for value in record["vector"]],
                "metadata": dict(record.get("metadata", {}))
                if isinstance(record.get("metadata"), Mapping)
                else {},
            }
        self.path.write_text(
            json.dumps(
                [existing[key] for key in sorted(existing)],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return list(self._load_records().values())[:top_k]

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records = self._load_records()
        return [records[item_id] for item_id in ids if item_id in records]

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Invalid vector store JSON: {self.path}")
        return {
            str(record["id"]): dict(record)
            for record in payload
            if isinstance(record, Mapping) and isinstance(record.get("id"), str)
        }


def _ingestion_result_payload(result: IngestionResult) -> dict[str, Any]:
    return {
        "file_hash": result.file_hash,
        "source_path": result.source_path,
        "collection": result.collection,
        "skipped": result.skipped,
        "chunk_count": result.chunk_count,
        "vector_count": len(result.vector_ids),
        "image_count": len(result.image_records),
        "bm25_index_path": str(result.bm25_index_path)
        if result.bm25_index_path is not None
        else None,
    }


def _delete_result_payload(result: DeleteResult) -> dict[str, Any]:
    return result.to_dict()


def _safe_filename(filename: str) -> str:
    raw_name = Path(str(filename or "upload.pdf")).name
    stem = Path(raw_name).stem
    suffix = Path(raw_name).suffix.lower() or ".pdf"
    safe_stem = _safe_segment(stem, default="upload")
    return f"{safe_stem}{suffix}"


def _safe_segment(value: str, *, default: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    return normalized or default


def _dedupe_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"unable to allocate upload path for {path.name}")


__all__ = [
    "IngestionService",
    "JsonVectorStore",
    "LocalDeterministicEmbedding",
    "UploadTarget",
]
