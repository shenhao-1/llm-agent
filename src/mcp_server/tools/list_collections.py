"""MCP tool for listing available document collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.protocol_handler import ToolDefinition


DEFAULT_DOCUMENTS_DIR = Path("data/documents")


def list_collections(documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR) -> dict[str, Any]:
    """Return collections discovered under data/documents/{collection}/."""
    root = Path(documents_dir)
    if not root.exists():
        return _result([], root)
    if not root.is_dir():
        raise ValueError(f"documents_dir must be a directory: {root}")

    collections = [
        _collection_info(collection_dir)
        for collection_dir in sorted(root.iterdir(), key=lambda path: path.name)
        if collection_dir.is_dir()
    ]
    return _result(collections, root)


def handle_list_collections(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dict")
    documents_dir = arguments.get("documents_dir", DEFAULT_DOCUMENTS_DIR)
    return list_collections(documents_dir)


def tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="list_collections",
        description="List available document collections under data/documents.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=handle_list_collections,
    )


def _collection_info(collection_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in collection_dir.rglob("*") if path.is_file())
    return {
        "name": collection_dir.name,
        "path": str(collection_dir),
        "document_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _result(collections: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    names = [collection["name"] for collection in collections]
    if collections:
        lines = ["Available collections:"]
        lines.extend(
            f"- {collection['name']} ({collection['document_count']} documents)"
            for collection in collections
        )
        text = "\n".join(lines)
    else:
        text = f"No collections found under {root}."

    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": {
            "collections": collections,
            "collection_names": names,
            "count": len(collections),
            "documents_dir": str(root),
        },
    }


__all__ = [
    "handle_list_collections",
    "list_collections",
    "tool_definition",
]
