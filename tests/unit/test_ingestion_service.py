"""Unit tests for dashboard ingestion service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ingestion.document_manager import DeleteResult, DocumentInfo
from ingestion.pipeline import IngestionResult
from observability.dashboard.services.ingestion_service import (
    IngestionService,
    JsonVectorStore,
)


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        source_path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        trace=None,
        on_progress=None,
    ) -> IngestionResult:
        self.calls.append(
            {
                "source_path": str(source_path),
                "collection": collection,
                "force": force,
            }
        )
        if on_progress is not None:
            on_progress("load", 2, 8)
            on_progress("store_images", 8, 8)
        return IngestionResult(
            file_hash="hash-a",
            source_path=str(source_path),
            collection=collection,
            vector_ids=["vec-a"],
            image_records=[],
        )


class FakeDocumentManager:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str | None]] = []

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        documents = [
            DocumentInfo(
                doc_id="hash-a",
                source_path="docs/a.pdf",
                collection="docs",
                doc_hash="hash-a",
                chunk_count=3,
                image_count=1,
                status="success",
            )
        ]
        if collection is None:
            return documents
        return [document for document in documents if document.collection == collection]

    def delete_document(
        self,
        source_path: str,
        collection: str | None = None,
    ) -> DeleteResult:
        self.deleted.append((source_path, collection))
        return DeleteResult(
            source_path=source_path,
            collection=collection,
            chroma_deleted=1,
            bm25_deleted=2,
        )


def test_save_upload_sanitizes_collection_and_filename(tmp_path: Path) -> None:
    service = _service(tmp_path)

    target = service.save_upload(
        "../My File.PDF",
        b"%PDF-1.4",
        collection="docs / unsafe",
    )

    assert target.filename == "My_File.pdf"
    assert target.collection == "docs_unsafe"
    assert target.path == tmp_path / "uploads" / "docs_unsafe" / "My_File.pdf"
    assert target.path.read_bytes() == b"%PDF-1.4"


def test_save_upload_deduplicates_existing_file(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.save_upload("paper.pdf", b"first", collection="docs")
    second = service.save_upload("paper.pdf", b"second", collection="docs")

    assert first.path.name == "paper.pdf"
    assert second.path.name == "paper_1.pdf"
    assert second.path.read_bytes() == b"second"


def test_ingest_file_forwards_progress_and_returns_summary(tmp_path: Path) -> None:
    service = _service(tmp_path)
    progress: list[tuple[str, int, int]] = []

    result = service.ingest_file(
        tmp_path / "a.pdf",
        collection="docs",
        force=True,
        on_progress=lambda stage, current, total: progress.append(
            (stage, current, total)
        ),
    )

    assert service.pipeline.calls == [  # type: ignore[attr-defined]
        {
            "source_path": str(tmp_path / "a.pdf"),
            "collection": "docs",
            "force": True,
        }
    ]
    assert progress == [("load", 2, 8), ("store_images", 8, 8)]
    assert result == {
        "file_hash": "hash-a",
        "source_path": str(tmp_path / "a.pdf"),
        "collection": "docs",
        "skipped": False,
        "chunk_count": 0,
        "vector_count": 1,
        "image_count": 0,
        "bm25_index_path": None,
    }
    trace_payload = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
    assert trace_payload["trace_type"] == "ingestion"
    assert trace_payload["status"] == "success"
    assert trace_payload["source_path"] == str(tmp_path / "a.pdf")
    assert trace_payload["collection"] == "docs"


def test_delete_document_delegates_to_document_manager(tmp_path: Path) -> None:
    service = _service(tmp_path)

    result = service.delete_document("docs/a.pdf", "docs")

    assert service.document_manager.deleted == [("docs/a.pdf", "docs")]  # type: ignore[attr-defined]
    assert result["deleted"] is True
    assert result["chroma_deleted"] == 1
    assert result["bm25_deleted"] == 2


def test_json_vector_store_upserts_records_idempotently(tmp_path: Path) -> None:
    store = JsonVectorStore(tmp_path / "vectors.json")
    first: list[Mapping[str, Any]] = [
        {"id": "b", "text": "old", "vector": [1.0], "metadata": {"v": 1}},
        {"id": "a", "text": "alpha", "vector": [2.0], "metadata": {}},
    ]
    second: list[Mapping[str, Any]] = [
        {"id": "b", "text": "new", "vector": [3.0], "metadata": {"v": 2}},
    ]

    store.upsert(first)
    store.upsert(second)

    assert [record["id"] for record in store.query([1.0], 10)] == ["a", "b"]
    assert store.get_by_ids(["b"]) == [
        {"id": "b", "text": "new", "vector": [3.0], "metadata": {"v": 2}}
    ]


def _service(tmp_path: Path) -> IngestionService:
    return IngestionService(
        pipeline=FakePipeline(),  # type: ignore[arg-type]
        document_manager=FakeDocumentManager(),  # type: ignore[arg-type]
        upload_root=tmp_path / "uploads",
        trace_log_path=tmp_path / "traces.jsonl",
    )
