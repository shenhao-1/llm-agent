"""MCP tool for listing available document collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from ingestion.storage import BM25Indexer

from mcp_server.protocol_handler import ToolDefinition


DEFAULT_DOCUMENTS_DIR = Path("data/documents")


def list_collections(documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR) -> dict[str, Any]:
    """Return collections discovered under data/documents/{collection}/."""
    root = Path(documents_dir)
    if root.exists() and not root.is_dir():
        raise ValueError(f"documents_dir must be a directory: {root}")

    collections_by_name: dict[str, dict[str, Any]] = {}
    if root.exists():
        for collection_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if collection_dir.is_dir():
                info = _collection_info(collection_dir)
                collections_by_name[info["name"]] = info
    index = BM25Indexer(root.parent / "db" / "bm25")
    if index.index_path.exists():
        index.load()
        counts: dict[str, set[str]] = {}
        for record in index.documents.values():
            name = str(record.metadata.get("collection", "default"))
            source = str(record.metadata.get("source_path", ""))
            counts.setdefault(name, set()).add(source)
        for name, sources in counts.items():
            collections_by_name[name] = {
                "name": name,
                "path": str(index.index_path),
                "document_count": len(sources),
                "total_bytes": sum(
                    Path(source).stat().st_size
                    for source in sources
                    if Path(source).is_file()
                ),
            }
    return _result([collections_by_name[name] for name in sorted(collections_by_name)], root)


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
