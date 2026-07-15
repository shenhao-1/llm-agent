"""MCP tool for querying the local knowledge hub."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from core.query_engine import (
    DenseRetriever,
    HybridSearch,
    QueryProcessor,
    Reranker,
    RRFFusion,
    SparseRetriever,
)
from core.response import ResponseBuilder
from core.settings import load_settings
from core.trace import TraceContext
from core.types import RetrievalResult
from ingestion.storage import BM25Indexer
from libs.embedding import BaseEmbedding
from libs.vector_store import BaseVectorStore
from mcp_server.protocol_handler import ToolDefinition
from observability.logger import DEFAULT_TRACE_LOG_PATH, write_trace


DEFAULT_SETTINGS = Path("config/settings.yaml")
DEFAULT_DATA_DIR = Path("data")


def query_knowledge_hub(
    query: str,
    *,
    top_k: int = 5,
    collection: str | None = None,
    search: HybridSearch | None = None,
    reranker: Reranker | None = None,
    response_builder: ResponseBuilder | None = None,
    settings_path: str | Path = DEFAULT_SETTINGS,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    trace: TraceContext | None = None,
    trace_log_path: str | Path | None = DEFAULT_TRACE_LOG_PATH,
) -> dict[str, Any]:
    """Run hybrid retrieval, optional rerank, and return an MCP tool result."""
    _validate_query_args(query, top_k, collection)
    builder = response_builder or ResponseBuilder()
    owns_trace = trace is None
    current_trace = trace or TraceContext(trace_type="query")
    results: list[RetrievalResult] = []
    status = "success"
    error: str | None = None

    try:
        searcher = search or _build_default_search(settings_path, data_dir)
        filters = {"collection": collection} if collection else None
        results = searcher.search(query, top_k=top_k, filters=filters, trace=current_trace)
        ranker = reranker
        if ranker is None:
            ranker = Reranker(settings=load_settings(settings_path))
        results = ranker.rerank(query, results, trace=current_trace)
    except FileNotFoundError:
        status = "empty"
        results = []
    except RuntimeError as exc:
        if "first" not in str(exc).lower() and "not found" not in str(exc).lower():
            status = "failed"
            error = str(exc)
            raise
        status = "empty"
        results = []
    finally:
        current_trace.finish()
        if owns_trace and trace_log_path is not None:
            payload = current_trace.to_dict()
            payload.update(
                {
                    "query": query,
                    "collection": collection,
                    "top_k": top_k,
                    "result_count": len(results),
                    "status": status,
                }
            )
            if error is not None:
                payload["error"] = error
            write_trace(payload, trace_log_path)

    return builder.build(results, query)


def handle_query_knowledge_hub(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dict")
    query = arguments.get("query")
    top_k = arguments.get("top_k", 5)
    collection = arguments.get("collection")
    data_dir = arguments.get("data_dir", DEFAULT_DATA_DIR)
    settings_path = arguments.get("settings_path", DEFAULT_SETTINGS)
    return query_knowledge_hub(
        query=query,
        top_k=top_k,
        collection=collection,
        settings_path=settings_path,
        data_dir=data_dir,
    )


def tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="query_knowledge_hub",
        description="Search indexed documents and return Markdown with structured citations.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "Maximum number of chunks to return.",
                },
                "collection": {
                    "type": "string",
                    "description": "Optional collection metadata filter.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handle_query_knowledge_hub,
    )


class LocalDeterministicEmbedding(BaseEmbedding):
    """Offline embedding provider compatible with scripts/ingest.py output."""

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        vectors: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(char) for char in text) % 997)
            token_count = float(len(text.split()))
            vectors.append([length, checksum, token_count])
        return vectors


class ReadOnlyJsonVectorStore(BaseVectorStore):
    """Read and query the JSON vector store produced by scripts/ingest.py."""

    def __init__(self, path: str | Path) -> None:
        super().__init__({})
        self.path = Path(path)

    def upsert(self, records: list[Mapping[str, Any]], trace=None) -> None:
        raise NotImplementedError("query_knowledge_hub vector store is read-only")

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        records = self._load_records()
        filtered = [
            record
            for record in records
            if _metadata_matches(record.get("metadata", {}), filters)
        ]
        ranked = sorted(
            filtered,
            key=lambda record: (
                -_similarity(vector, record.get("vector")),
                str(record.get("id", "")),
            ),
        )
        return [
            {
                "id": str(record["id"]),
                "text": str(record.get("text", "")),
                "metadata": dict(record.get("metadata", {})),
                "score": _similarity(vector, record.get("vector")),
            }
            for record in ranked[:top_k]
        ]

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records_by_id = {str(record["id"]): record for record in self._load_records()}
        return [
            {
                "id": item_id,
                "text": str(records_by_id[item_id].get("text", "")),
                "metadata": dict(records_by_id[item_id].get("metadata", {})),
                "vector": list(records_by_id[item_id].get("vector", [])),
            }
            for item_id in ids
            if item_id in records_by_id
        ]

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"vector index not found: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"invalid vector index format: {self.path}")
        return [
            dict(item)
            for item in payload
            if isinstance(item, Mapping)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("vector"), list)
        ]


def _build_default_search(
    settings_path: str | Path,
    data_dir: str | Path,
) -> HybridSearch:
    settings = load_settings(settings_path)
    db_dir = Path(data_dir) / "db"
    vector_store = ReadOnlyJsonVectorStore(db_dir / "vectors.json")
    bm25_indexer = BM25Indexer(db_dir / "bm25").load()
    return HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            embedding_client=LocalDeterministicEmbedding(),
            vector_store=vector_store,
        ),
        sparse_retriever=SparseRetriever(
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        ),
        fusion=RRFFusion(),
    )


def _validate_query_args(query: Any, top_k: Any, collection: Any) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if collection is not None and (
        not isinstance(collection, str) or not collection.strip()
    ):
        raise ValueError("collection must be a non-empty string when provided")


def _metadata_matches(metadata: Any, filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True
    if not isinstance(metadata, Mapping):
        return False
    for key, expected in filters.items():
        if key not in metadata or metadata[key] != expected:
            return False
    return True


def _similarity(query_vector: list[float], candidate_vector: Any) -> float:
    if not isinstance(candidate_vector, list) or not candidate_vector:
        return 0.0
    values = [
        float(value)
        for value in candidate_vector
        if isinstance(value, (int, float))
    ]
    if not values or len(values) != len(query_vector):
        return 0.0
    distance = math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(query_vector, values))
    )
    return 1.0 / (1.0 + distance)


__all__ = [
    "LocalDeterministicEmbedding",
    "ReadOnlyJsonVectorStore",
    "handle_query_knowledge_hub",
    "query_knowledge_hub",
    "tool_definition",
]
