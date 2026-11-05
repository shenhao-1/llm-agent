"""Tests for the list_collections MCP tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.types import Chunk
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer

from mcp_server.tools.list_collections import (
    handle_list_collections,
    list_collections,
    tool_definition,
)


def test_list_collections_returns_sorted_collection_names_and_stats(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    (docs_dir / "notes").mkdir(parents=True)
    (docs_dir / "docs").mkdir(parents=True)
    (docs_dir / "docs" / "a.pdf").write_bytes(b"abc")
    (docs_dir / "docs" / "nested").mkdir()
    (docs_dir / "docs" / "nested" / "b.md").write_bytes(b"12345")
    (docs_dir / "notes" / "n.md").write_text("note", encoding="utf-8")

    result = list_collections(docs_dir)

    assert result["structuredContent"]["collection_names"] == ["docs", "notes"]
    assert result["structuredContent"]["count"] == 2
    assert result["structuredContent"]["collections"][0] == {
        "name": "docs",
        "path": str(docs_dir / "docs"),
        "document_count": 2,
        "total_bytes": 8,
    }
    assert "- docs (2 documents)" in result["content"][0]["text"]


def test_list_collections_missing_directory_returns_empty_result(tmp_path: Path) -> None:
    docs_dir = tmp_path / "missing"

    result = list_collections(docs_dir)

    assert result["structuredContent"]["collections"] == []
    assert result["structuredContent"]["collection_names"] == []
    assert "No collections found" in result["content"][0]["text"]


def test_list_collections_reads_index_when_documents_directory_is_absent(tmp_path: Path) -> None:
    chunk = Chunk(
        id="chunk-1",
        text="indexed document",
        metadata={"collection": "research", "source_path": "research/paper.pdf"},
    )
    BM25Indexer(tmp_path / "db" / "bm25").build(SparseEncoder().encode([chunk]))

    result = list_collections(tmp_path / "documents")

    assert result["structuredContent"]["collection_names"] == ["research"]
    assert result["structuredContent"]["collections"][0]["document_count"] == 1


def test_list_collections_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-dir"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="documents_dir"):
        list_collections(file_path)


def test_handle_list_collections_accepts_arguments_dict(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    (docs_dir / "alpha").mkdir(parents=True)

    result = handle_list_collections({"documents_dir": str(docs_dir)})

    assert result["structuredContent"]["collection_names"] == ["alpha"]


def test_tool_definition_schema() -> None:
    tool = tool_definition()

    assert tool.name == "list_collections"
    assert tool.input_schema["type"] == "object"
    assert tool.to_schema()["name"] == "list_collections"
