"""MCP tool for returning a document-level metadata summary."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping
from core.runtime import runtime_settings
from core.settings import load_settings
from libs.vector_store import VectorStoreFactory

from mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError, ToolDefinition


DEFAULT_DATA_DIR = Path("data")


class DocumentSummaryNotFound(LookupError):
    """Raised when no indexed metadata matches a requested document."""


def get_document_summary(
    doc_id: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    offline: bool = False,
    settings_path: str | Path = "config/settings.yaml",
) -> dict[str, Any]:
    """Return title, summary, and tags for a document from vector metadata."""
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")

    if offline:
        records = _load_vector_records(Path(data_dir) / "db" / "vectors.json")
    else:
        settings = runtime_settings(load_settings(settings_path), data_dir)
        store = VectorStoreFactory.create(settings)
        records = store.get_by_metadata()
    matching = [
        record
        for record in records
        if _matches_doc_id(record, doc_id.strip())
    ]
    if not matching:
        raise DocumentSummaryNotFound(f"Document not found: {doc_id}")

    structured = _build_structured_summary(doc_id.strip(), matching)
    return {
        "content": [{"type": "text", "text": _build_markdown(structured)}],
        "structuredContent": structured,
    }


def handle_get_document_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dict")
    try:
        return get_document_summary(
            doc_id=arguments.get("doc_id"),
            data_dir=arguments.get("data_dir", DEFAULT_DATA_DIR),
            offline=arguments.get("offline", False),
            settings_path=arguments.get("settings_path", "config/settings.yaml"),
        )
    except DocumentSummaryNotFound as exc:
        raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc


def tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="get_document_summary",
        description="Return title, summary, and tags for an indexed document.",
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document hash, source path, or source filename stem.",
                },
                "offline": {
                    "type": "boolean",
                    "default": False,
                    "description": "Read the legacy JSON demo index.",
                },
            },
            "required": ["doc_id"],
            "additionalProperties": False,
        },
        handler=handle_get_document_summary,
    )


def _load_vector_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DocumentSummaryNotFound(f"Document not found: vector index missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"invalid vector index format: {path}")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _matches_doc_id(record: Mapping[str, Any], doc_id: str) -> bool:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    source_path = metadata.get("source_path")
    candidates = {
        str(value)
        for value in [
            record.get("id"),
            metadata.get("doc_id"),
            metadata.get("doc_hash"),
            metadata.get("source_chunk_id"),
            source_path,
        ]
        if isinstance(value, str) and value.strip()
    }
    if isinstance(source_path, str) and source_path.strip():
        candidates.add(Path(source_path).stem)
        candidates.add(Path(source_path).name)
    return doc_id in candidates


def _build_structured_summary(
    requested_doc_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metadatas = [
        dict(record.get("metadata", {}))
        for record in records
        if isinstance(record.get("metadata", {}), Mapping)
    ]
    first_metadata = metadatas[0] if metadatas else {}
    source_path = _first_string(metadatas, "source_path") or "unknown"
    doc_hash = _first_string(metadatas, "doc_hash")
    title = (
        _first_string(metadatas, "document_title")
        or _first_string(metadatas, "title")
        or (Path(source_path).name if source_path != "unknown" else requested_doc_id)
    )
    summary = (
        _first_string(metadatas, "document_summary")
        or _combine_summaries(metadatas)
        or "No summary available."
    )
    return {
        "doc_id": doc_hash or requested_doc_id,
        "requested_doc_id": requested_doc_id,
        "source_path": source_path,
        "collection": first_metadata.get("collection"),
        "title": title,
        "summary": summary,
        "tags": _collect_tags(metadatas),
        "chunk_count": len(records),
    }


def _first_string(metadatas: list[dict[str, Any]], key: str) -> str | None:
    for metadata in metadatas:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _combine_summaries(metadatas: list[dict[str, Any]]) -> str | None:
    summaries: list[str] = []
    for metadata in metadatas:
        value = metadata.get("summary")
        if isinstance(value, str) and value.strip() and value.strip() not in summaries:
            summaries.append(value.strip())
        if len(summaries) == 3:
            break
    if not summaries:
        return None
    return " ".join(summaries)


def _collect_tags(metadatas: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for metadata in metadatas:
        value = metadata.get("tags", [])
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []
        for tag in values:
            if isinstance(tag, str) and tag.strip() and tag.strip() not in tags:
                tags.append(tag.strip())
    return tags


def _build_markdown(summary: Mapping[str, Any]) -> str:
    tags = summary.get("tags") or []
    tag_text = ", ".join(tags) if tags else "none"
    return "\n".join(
        [
            f"## {summary['title']}",
            "",
            f"- Document: {summary['doc_id']}",
            f"- Source: {summary['source_path']}",
            f"- Collection: {summary.get('collection') or '-'}",
            f"- Chunks: {summary['chunk_count']}",
            f"- Tags: {tag_text}",
            "",
            str(summary["summary"]),
        ]
    )


__all__ = [
    "DocumentSummaryNotFound",
    "get_document_summary",
    "handle_get_document_summary",
    "tool_definition",
]
