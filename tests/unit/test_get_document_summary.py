"""Tests for the get_document_summary MCP tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from mcp_server.tools.get_document_summary import (
    DocumentSummaryNotFound,
    get_document_summary,
    handle_get_document_summary,
    tool_definition,
)


def test_get_document_summary_returns_structured_metadata(tmp_path: Path) -> None:
    _write_vectors(
        tmp_path,
        [
            _record(
                "vec-1",
                doc_hash="doc-123",
                source_path="docs/guide.pdf",
                title="Azure Guide",
                summary="Configure Azure endpoints.",
                tags=["azure", "config"],
            ),
            _record(
                "vec-2",
                doc_hash="doc-123",
                source_path="docs/guide.pdf",
                title="Azure Guide",
                summary="Use API keys securely.",
                tags=["security", "azure"],
            ),
        ],
    )

    result = get_document_summary("doc-123", data_dir=tmp_path, offline=True)

    structured = result["structuredContent"]
    assert structured["doc_id"] == "doc-123"
    assert structured["source_path"] == "docs/guide.pdf"
    assert structured["title"] == "Azure Guide"
    assert structured["summary"] == "Configure Azure endpoints. Use API keys securely."
    assert structured["tags"] == ["azure", "config", "security"]
    assert structured["chunk_count"] == 2
    assert "## Azure Guide" in result["content"][0]["text"]


def test_get_document_summary_can_match_source_path_or_stem(tmp_path: Path) -> None:
    _write_vectors(
        tmp_path,
        [_record("vec-1", doc_hash="hash", source_path="docs/manual.pdf")],
    )

    by_path = get_document_summary("docs/manual.pdf", data_dir=tmp_path, offline=True)
    by_stem = get_document_summary("manual", data_dir=tmp_path, offline=True)

    assert by_path["structuredContent"]["doc_id"] == "hash"
    assert by_stem["structuredContent"]["doc_id"] == "hash"


def test_get_document_summary_missing_doc_raises_not_found(tmp_path: Path) -> None:
    _write_vectors(tmp_path, [_record("vec-1", doc_hash="known")])

    with pytest.raises(DocumentSummaryNotFound, match="Document not found"):
        get_document_summary("missing", data_dir=tmp_path, offline=True)


def test_handle_get_document_summary_maps_not_found_to_json_rpc_error(tmp_path: Path) -> None:
    _write_vectors(tmp_path, [])

    with pytest.raises(JsonRpcError) as exc:
        handle_get_document_summary({"doc_id": "missing", "data_dir": str(tmp_path), "offline": True})

    assert exc.value.code == INVALID_PARAMS
    assert exc.value.message == "Document not found: missing"


def test_get_document_summary_validates_doc_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="doc_id"):
        get_document_summary("", data_dir=tmp_path)


def test_tool_definition_schema() -> None:
    tool = tool_definition()

    assert tool.name == "get_document_summary"
    assert tool.input_schema["required"] == ["doc_id"]
    assert tool.to_schema()["name"] == "get_document_summary"


def _write_vectors(tmp_path: Path, records: list[dict[str, object]]) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "vectors.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _record(
    record_id: str,
    *,
    doc_hash: str,
    source_path: str = "docs/sample.pdf",
    title: str = "Sample",
    summary: str = "Sample summary.",
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "text": "chunk text",
        "vector": [1.0, 2.0, 3.0],
        "metadata": {
            "doc_hash": doc_hash,
            "source_path": source_path,
            "collection": "docs",
            "title": title,
            "summary": summary,
            "tags": tags or ["sample"],
        },
    }
