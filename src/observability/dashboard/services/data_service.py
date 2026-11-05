"""Data access helpers for dashboard data browsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.settings import load_settings
from core.runtime import runtime_settings
from ingestion.document_manager import DocumentDetail, DocumentInfo, DocumentManager
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


class DataService:
    """Wrap document and storage reads for Streamlit pages."""

    def __init__(self, document_manager: DocumentManager) -> None:
        self.document_manager = document_manager

    @classmethod
    def from_settings(
        cls,
        settings_path: str | Path = "config/settings.yaml",
        *,
        data_dir: str | Path = "data",
    ) -> "DataService":
        settings = load_settings(settings_path)
        data_root = Path(data_dir)
        bm25 = BM25Indexer(data_root / "db" / "bm25")
        if bm25.index_path.exists():
            bm25.load()
        image_storage = ImageStorage(
            image_root=data_root / "images",
            db_path=data_root / "db" / "image_index.db",
        )
        manager = DocumentManager(
            chroma_store=ChromaStore(runtime_settings(settings, data_root)["vector_store"]),
            bm25_indexer=bm25,
            image_storage=image_storage,
            file_integrity=SQLiteIntegrityChecker(
                data_root / "db" / "ingestion_history.db"
            ),
        )
        return cls(manager)

    def list_collections(self) -> list[str]:
        documents = self.document_manager.list_documents()
        return sorted({document.collection for document in documents})

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        return [
            _document_row(document)
            for document in self.document_manager.list_documents(collection)
        ]

    def get_document_detail(self, doc_id: str) -> dict[str, Any]:
        detail = self.document_manager.get_document_detail(doc_id)
        return _detail_payload(detail)

    def get_collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        return self.document_manager.get_collection_stats(collection).to_dict()


def _document_row(document: DocumentInfo) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "source_path": document.source_path,
        "collection": document.collection,
        "chunk_count": document.chunk_count,
        "image_count": document.image_count,
        "status": document.status,
        "updated_at": document.updated_at,
    }


def _detail_payload(detail: DocumentDetail) -> dict[str, Any]:
    payload = detail.to_dict()
    payload["chunk_count"] = len(payload["chunks"])
    payload["image_count"] = len(payload["images"])
    return payload


__all__ = ["DataService"]
