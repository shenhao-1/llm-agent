"""Sparse BM25 retrieval orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from core.types import RetrievalResult
from ingestion.storage import BM25Indexer
from libs.vector_store import BaseVectorStore, VectorStoreFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


class SparseRetriever:
    """Query BM25 candidates and hydrate them with vector-store text/metadata."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if settings is None and (bm25_indexer is None or vector_store is None):
            raise ValueError(
                "settings are required unless both bm25_indexer and vector_store are provided"
            )

        self.settings = settings or {}
        self.bm25_indexer = bm25_indexer or BM25Indexer(
            self._extract_bm25_index_dir(self.settings)
        ).load()
        self.vector_store = vector_store or VectorStoreFactory.create(self.settings)

    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Return BM25-ranked results with hydrated text and metadata."""
        self._validate_keywords(keywords)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_text = " ".join(keywords)
        bm25_hits = self.bm25_indexer.query(query_text, top_k=top_k)
        chunk_ids = [str(hit["chunk_id"]) for hit in bm25_hits]
        records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        records_by_id = {
            str(record.get("id", record.get("chunk_id"))): record
            for record in records
            if isinstance(record, Mapping)
        }

        results: list[RetrievalResult] = []
        for hit in bm25_hits:
            chunk_id = str(hit["chunk_id"])
            record = records_by_id.get(chunk_id)
            if record is None:
                # A stale BM25 posting must not become an empty-text search hit.
                continue
            metadata = self._metadata_from(record, hit)
            text = record.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(hit.get("score", 0.0)),
                    text=text,
                    metadata=metadata,
                )
            )

        if trace is not None:
            trace.record_stage(
                "sparse_retrieval",
                method="sparse_retriever",
                details={
                    "keyword_count": len(keywords),
                    "top_k": top_k,
                    "result_count": len(results),
                },
            )

        return results

    @staticmethod
    def _metadata_from(
        record: Mapping[str, Any],
        hit: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata = hit.get("metadata", {})
        if isinstance(metadata, Mapping):
            merged = dict(metadata)
        else:
            merged = {}
        record_metadata = record.get("metadata", {})
        if isinstance(record_metadata, Mapping):
            merged.update(dict(record_metadata))
        return merged

    @staticmethod
    def _validate_keywords(keywords: Any) -> None:
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("keywords must be a non-empty list")
        for idx, keyword in enumerate(keywords):
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError(f"keywords[{idx}] must be a non-empty string")

    @staticmethod
    def _extract_bm25_index_dir(settings: "Settings | Mapping[str, Any]") -> Path:
        retrieval: Mapping[str, Any] = {}
        if hasattr(settings, "retrieval"):
            candidate = getattr(settings, "retrieval")
            if isinstance(candidate, Mapping):
                retrieval = candidate
        elif isinstance(settings, Mapping):
            candidate = settings.get("retrieval")
            if isinstance(candidate, Mapping):
                retrieval = candidate

        sparse = retrieval.get("sparse")
        if isinstance(sparse, Mapping) and sparse.get("index_dir"):
            return Path(str(sparse["index_dir"]))
        if retrieval.get("bm25_index_dir"):
            return Path(str(retrieval["bm25_index_dir"]))
        return Path("data/db/bm25")


__all__ = ["SparseRetriever"]
