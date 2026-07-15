"""Unit tests for dashboard data service."""

from __future__ import annotations

from ingestion.document_manager import CollectionStats, DocumentDetail, DocumentInfo
from observability.dashboard.services import DataService


class FakeDocumentManager:
    def __init__(self) -> None:
        self.documents = [
            DocumentInfo(
                doc_id="doc-a",
                source_path="docs/a.pdf",
                collection="docs",
                doc_hash="hash-a",
                chunk_count=2,
                image_count=1,
                status="success",
                updated_at="2026-06-03T00:00:00",
            ),
            DocumentInfo(
                doc_id="doc-b",
                source_path="notes/b.pdf",
                collection="notes",
                doc_hash="hash-b",
                chunk_count=1,
                image_count=0,
                status="success",
            ),
        ]

    def list_documents(self, collection: str | None = None):
        if collection is None:
            return list(self.documents)
        return [document for document in self.documents if document.collection == collection]

    def get_document_detail(self, doc_id: str):
        info = next(document for document in self.documents if document.doc_id == doc_id)
        return DocumentDetail(
            info=info,
            chunks=[
                {
                    "id": "chunk-a",
                    "text": "chunk text",
                    "metadata": {"source_path": info.source_path},
                }
            ],
            images=[
                {
                    "image_id": "img-a",
                    "file_path": "data/images/docs/img-a.png",
                    "collection": info.collection,
                    "doc_hash": info.doc_hash,
                }
            ]
            if info.image_count
            else [],
        )

    def get_collection_stats(self, collection: str | None = None):
        documents = self.list_documents(collection)
        return CollectionStats(
            collection=collection,
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            image_count=sum(document.image_count for document in documents),
        )


def test_data_service_lists_collections_and_documents() -> None:
    service = DataService(FakeDocumentManager())  # type: ignore[arg-type]

    assert service.list_collections() == ["docs", "notes"]
    assert service.list_documents("docs") == [
        {
            "doc_id": "doc-a",
            "source_path": "docs/a.pdf",
            "collection": "docs",
            "chunk_count": 2,
            "image_count": 1,
            "status": "success",
            "updated_at": "2026-06-03T00:00:00",
        }
    ]


def test_data_service_returns_detail_payload_counts() -> None:
    service = DataService(FakeDocumentManager())  # type: ignore[arg-type]

    detail = service.get_document_detail("doc-a")

    assert detail["info"]["source_path"] == "docs/a.pdf"
    assert detail["chunk_count"] == 1
    assert detail["image_count"] == 1
    assert detail["chunks"][0]["text"] == "chunk text"
    assert detail["images"][0]["image_id"] == "img-a"


def test_data_service_returns_collection_stats_dict() -> None:
    service = DataService(FakeDocumentManager())  # type: ignore[arg-type]

    assert service.get_collection_stats("notes") == {
        "collection": "notes",
        "document_count": 1,
        "chunk_count": 1,
        "image_count": 0,
    }
