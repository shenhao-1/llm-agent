"""Unit tests for cross-store document management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from core import Chunk
from ingestion import DocumentManager
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker


class FakeChromaStore:
    def __init__(self) -> None:
        self.deleted_filters: list[dict[str, Any]] = []
        self.delete_count = 0

    def delete_by_metadata(self, filters: Mapping[str, Any]) -> int:
        self.deleted_filters.append(dict(filters))
        return self.delete_count


def test_list_documents_returns_chunk_and_image_counts(tmp_path: Path) -> None:
    manager, _chroma, _checker = _manager(tmp_path)

    documents = manager.list_documents()

    assert [document.source_path for document in documents] == [
        "docs/guide.pdf",
        "notes/ref.pdf",
    ]
    guide = documents[0]
    assert guide.doc_id == "doc-guide"
    assert guide.collection == "docs"
    assert guide.chunk_count == 2
    assert guide.image_count == 1


def test_list_documents_filters_by_collection(tmp_path: Path) -> None:
    manager, _chroma, _checker = _manager(tmp_path)

    documents = manager.list_documents(collection="notes")

    assert [document.source_path for document in documents] == ["notes/ref.pdf"]
    assert documents[0].chunk_count == 1


def test_get_document_detail_returns_ordered_chunks_and_images(tmp_path: Path) -> None:
    manager, _chroma, _checker = _manager(tmp_path)

    detail = manager.get_document_detail("doc-guide")

    assert detail.info.source_path == "docs/guide.pdf"
    assert [chunk["id"] for chunk in detail.chunks] == ["guide-0", "guide-1"]
    assert detail.chunks[0]["has_sparse_vector"] is True
    assert [image["image_id"] for image in detail.images] == ["img-guide"]


def test_get_collection_stats_aggregates_document_chunk_and_image_counts(
    tmp_path: Path,
) -> None:
    manager, _chroma, _checker = _manager(tmp_path)

    all_stats = manager.get_collection_stats()
    docs_stats = manager.get_collection_stats("docs")

    assert all_stats.document_count == 2
    assert all_stats.chunk_count == 3
    assert all_stats.image_count == 2
    assert docs_stats.document_count == 1
    assert docs_stats.chunk_count == 2
    assert docs_stats.image_count == 1


def test_delete_document_coordinates_all_backing_stores(tmp_path: Path) -> None:
    manager, chroma, checker = _manager(tmp_path)
    chroma.delete_count = 2

    result = manager.delete_document("docs/guide.pdf", collection="docs")

    assert result.deleted is True
    assert result.chroma_deleted == 2
    assert result.bm25_deleted == 2
    assert result.image_deleted == 1
    assert result.integrity_deleted == 1
    assert chroma.deleted_filters == [
        {"source_path": "docs/guide.pdf", "collection": "docs"}
    ]
    assert [document.source_path for document in manager.list_documents()] == [
        "notes/ref.pdf"
    ]
    assert checker.get_record("doc-guide") is None


def test_missing_document_raises_clear_error(tmp_path: Path) -> None:
    manager, _chroma, _checker = _manager(tmp_path)

    with pytest.raises(ValueError, match="document not found"):
        manager.get_document_detail("missing")


def _manager(
    tmp_path: Path,
) -> tuple[DocumentManager, FakeChromaStore, SQLiteIntegrityChecker]:
    bm25 = BM25Indexer(tmp_path / "bm25").build(
        SparseEncoder().encode(
            [
                _chunk("guide alpha", "guide-0", "docs/guide.pdf", "docs", "doc-guide", 0),
                _chunk("guide beta", "guide-1", "docs/guide.pdf", "docs", "doc-guide", 1),
                _chunk("reference notes", "ref-0", "notes/ref.pdf", "notes", "doc-ref", 0),
            ]
        )
    )
    image_storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "image_index.db",
    )
    image_storage.save_bytes(
        image_id="img-guide",
        data=b"guide",
        collection="docs",
        doc_hash="doc-guide",
    )
    image_storage.save_bytes(
        image_id="img-ref",
        data=b"ref",
        collection="notes",
        doc_hash="doc-ref",
    )
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    checker.mark_success(
        "doc-guide",
        "docs/guide.pdf",
        metadata={"collection": "docs", "chunk_count": 2, "image_count": 1},
    )
    checker.mark_success(
        "doc-ref",
        "notes/ref.pdf",
        metadata={"collection": "notes", "chunk_count": 1, "image_count": 1},
    )
    chroma = FakeChromaStore()
    return (
        DocumentManager(
            chroma_store=chroma,  # type: ignore[arg-type]
            bm25_indexer=bm25,
            image_storage=image_storage,
            file_integrity=checker,
        ),
        chroma,
        checker,
    )


def _chunk(
    text: str,
    chunk_id: str,
    source_path: str,
    collection: str,
    doc_hash: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={
            "source_path": source_path,
            "collection": collection,
            "doc_hash": doc_hash,
            "chunk_index": chunk_index,
        },
        start_offset=0,
        end_offset=len(text),
        source_ref=doc_hash,
    )
