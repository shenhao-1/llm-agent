"""Cross-storage document lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.types import ChunkRecord
from ingestion.storage import BM25Indexer, ImageRecord, ImageStorage
from libs.loader.file_integrity import FileIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore


@dataclass(frozen=True)
class DocumentInfo:
    """Summary of one ingested document."""

    doc_id: str
    source_path: str
    collection: str
    doc_hash: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    status: str = "indexed"
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "collection": self.collection,
            "doc_hash": self.doc_hash,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "status": self.status,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class DocumentDetail:
    """Detailed document view with chunks and images."""

    info: DocumentInfo
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": self.info.to_dict(),
            "chunks": [dict(chunk) for chunk in self.chunks],
            "images": [dict(image) for image in self.images],
        }


@dataclass(frozen=True)
class DeleteResult:
    """Deletion counts across all coordinated stores."""

    source_path: str
    collection: str | None
    chroma_deleted: int = 0
    bm25_deleted: int = 0
    image_deleted: int = 0
    integrity_deleted: int = 0

    @property
    def deleted(self) -> bool:
        return any(
            count > 0
            for count in (
                self.chroma_deleted,
                self.bm25_deleted,
                self.image_deleted,
                self.integrity_deleted,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "chroma_deleted": self.chroma_deleted,
            "bm25_deleted": self.bm25_deleted,
            "image_deleted": self.image_deleted,
            "integrity_deleted": self.integrity_deleted,
            "deleted": self.deleted,
        }


@dataclass(frozen=True)
class CollectionStats:
    """Aggregated collection-level document statistics."""

    collection: str | None
    document_count: int
    chunk_count: int
    image_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
        }


class DocumentManager:
    """Coordinate document list/detail/delete operations across local stores."""

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityChecker,
    ) -> None:
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """Return ingested documents aggregated by source path."""
        _validate_optional_collection(collection)
        grouped = self._group_bm25_documents()
        for record in self.file_integrity.list_processed():
            if record.get("status") != "success":
                continue
            source_path = str(record.get("file_path", ""))
            if not source_path:
                continue
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            item = grouped.setdefault(
                source_path,
                {
                    "source_path": source_path,
                    "collection": str(metadata.get("collection", "default")),
                    "doc_hash": str(record.get("file_hash", "")) or None,
                    "chunks": [],
                    "status": record.get("status", "success"),
                    "updated_at": record.get("updated_at"),
                },
            )
            item["status"] = record.get("status", item.get("status", "success"))
            item["updated_at"] = record.get("updated_at")
            if item.get("doc_hash") is None:
                item["doc_hash"] = str(record.get("file_hash", "")) or None

        documents = [
            self._info_from_group(item)
            for item in grouped.values()
            if collection is None or item.get("collection") == collection
        ]
        return sorted(documents, key=lambda item: (item.collection, item.source_path))

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        """Return one document with chunk and image details."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        target = doc_id.strip()
        info = self._find_document(target)
        chunks = [
            _chunk_record_to_dict(record)
            for record in self._records_for_document(info.source_path, info.collection)
        ]
        images = [
            image.to_dict()
            for image in self._images_for_doc(info.doc_hash, info.collection)
        ]
        return DocumentDetail(info=info, chunks=chunks, images=images)

    def delete_document(
        self,
        source_path: str,
        collection: str | None = None,
    ) -> DeleteResult:
        """Delete a document from Chroma, BM25, image storage, and integrity history."""
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string")
        _validate_optional_collection(collection)
        normalized_source = source_path.strip()
        info = self._find_document(normalized_source, collection=collection)

        filters: dict[str, Any] = {"source_path": info.source_path}
        if collection is not None:
            filters["collection"] = collection
        chroma_deleted = self.chroma_store.delete_by_metadata(filters)
        bm25_deleted = self.bm25_indexer.remove_document(
            info.source_path,
            collection,
        )
        image_deleted = (
            self.image_storage.delete_by_document(info.doc_hash, collection)
            if info.doc_hash
            else 0
        )
        integrity_deleted = self.file_integrity.remove_record(info.source_path)
        if integrity_deleted == 0 and info.doc_hash:
            integrity_deleted = self.file_integrity.remove_record(info.doc_hash)

        return DeleteResult(
            source_path=info.source_path,
            collection=collection,
            chroma_deleted=chroma_deleted,
            bm25_deleted=bm25_deleted,
            image_deleted=image_deleted,
            integrity_deleted=integrity_deleted,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return aggregate document, chunk, and image counts."""
        documents = self.list_documents(collection)
        return CollectionStats(
            collection=collection,
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            image_count=sum(document.image_count for document in documents),
        )

    def _find_document(
        self,
        doc_id: str,
        collection: str | None = None,
    ) -> DocumentInfo:
        for document in self.list_documents(collection):
            if doc_id in {document.doc_id, document.source_path, document.doc_hash}:
                return document
        raise ValueError(f"document not found: {doc_id}")

    def _group_bm25_documents(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for record in self.bm25_indexer.documents.values():
            metadata = dict(record.metadata)
            source_path = metadata.get("source_path")
            if not isinstance(source_path, str) or not source_path.strip():
                continue
            item = grouped.setdefault(
                source_path,
                {
                    "source_path": source_path,
                    "collection": str(metadata.get("collection", "default")),
                    "doc_hash": metadata.get("doc_hash")
                    if isinstance(metadata.get("doc_hash"), str)
                    else None,
                    "chunks": [],
                    "status": "indexed",
                    "updated_at": None,
                },
            )
            item["chunks"].append(record)
            if item.get("doc_hash") is None and isinstance(metadata.get("doc_hash"), str):
                item["doc_hash"] = metadata["doc_hash"]
        return grouped

    def _info_from_group(self, item: dict[str, Any]) -> DocumentInfo:
        doc_hash = item.get("doc_hash") if isinstance(item.get("doc_hash"), str) else None
        collection = str(item.get("collection", "default"))
        return DocumentInfo(
            doc_id=doc_hash or str(item["source_path"]),
            source_path=str(item["source_path"]),
            collection=collection,
            doc_hash=doc_hash,
            chunk_count=len(item.get("chunks", [])),
            image_count=len(self._images_for_doc(doc_hash, collection)),
            status=str(item.get("status", "indexed")),
            updated_at=item.get("updated_at")
            if isinstance(item.get("updated_at"), str)
            else None,
        )

    def _records_for_document(
        self,
        source_path: str,
        collection: str,
    ) -> list[ChunkRecord]:
        return sorted(
            [
                record
                for record in self.bm25_indexer.documents.values()
                if record.metadata.get("source_path") == source_path
                and record.metadata.get("collection") == collection
            ],
            key=lambda record: int(record.metadata.get("chunk_index", 0)),
        )

    def _images_for_doc(
        self,
        doc_hash: str | None,
        collection: str | None,
    ) -> list[ImageRecord]:
        if not doc_hash:
            return []
        return self.image_storage.list_by_document(doc_hash, collection)


def _chunk_record_to_dict(record: ChunkRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "text": record.text,
        "metadata": dict(record.metadata),
        "has_dense_vector": record.dense_vector is not None,
        "has_sparse_vector": record.sparse_vector is not None,
    }


def _validate_optional_collection(collection: str | None) -> None:
    if collection is not None and (
        not isinstance(collection, str) or not collection.strip()
    ):
        raise ValueError("collection must be a non-empty string when provided")


__all__ = [
    "CollectionStats",
    "DeleteResult",
    "DocumentDetail",
    "DocumentInfo",
    "DocumentManager",
]
