"""Hybrid dense+sparse retrieval orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, TYPE_CHECKING

from core.types import RetrievalResult

from .dense_retriever import DenseRetriever
from .fusion import RRFFusion
from .query_processor import QueryProcessor
from .sparse_retriever import SparseRetriever

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


class HybridSearch:
    """Coordinate query processing, dense retrieval, sparse retrieval, and fusion."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFFusion | None = None,
    ) -> None:
        if settings is None and query_processor is None:
            raise ValueError(
                "settings are required unless query_processor, dense_retriever, and sparse_retriever are provided"
            )

        self.settings = settings or {}
        retrieval = getattr(self.settings, "retrieval", None)
        if retrieval is None and isinstance(self.settings, Mapping):
            retrieval = self.settings.get("retrieval", {})
        retrieval = retrieval if isinstance(retrieval, Mapping) else {}
        self.strategy = str(retrieval.get("strategy", "hybrid")).lower()
        if self.strategy not in {"hybrid", "dense", "sparse"}:
            raise ValueError("retrieval.strategy must be hybrid, dense, or sparse")
        if settings is None and (
            (self.strategy != "sparse" and dense_retriever is None)
            or (self.strategy != "dense" and sparse_retriever is None)
        ):
            raise ValueError("active retrieval routes must be provided when settings are absent")
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or (
            DenseRetriever(self.settings) if self.strategy != "sparse" else None
        )
        self.sparse_retriever = sparse_retriever or (
            SparseRetriever(self.settings) if self.strategy != "dense" else None
        )
        self.fusion = fusion or RRFFusion()

    def search(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Return hybrid retrieval results with graceful single-route fallback."""
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping when provided")

        processed = self.query_processor.process(query, filters=filters, trace=trace)
        candidate_k = max(top_k * 2, top_k)
        dense_results, sparse_results, errors = self._retrieve_routes(
            query=processed.normalized_query or processed.original_query,
            keywords=processed.keywords,
            top_k=candidate_k,
            filters=processed.filters,
            trace=trace,
        )

        fused = (
            self.fusion.fuse(dense_results, sparse_results, top_k=None, trace=trace)
            if self.strategy == "hybrid"
            else dense_results or sparse_results
        )
        filtered = self._apply_metadata_filters(fused, processed.filters)
        results = filtered[:top_k]

        if trace is not None:
            trace.record_stage(
                "hybrid_search",
                method="hybrid_search",
                details={
                    "top_k": top_k,
                    "strategy": self.strategy,
                    "keyword_count": len(processed.keywords),
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(results),
                    "fallback": bool(errors),
                    "errors": errors,
                },
            )

        return results

    def _retrieve_routes(
        self,
        *,
        query: str,
        keywords: list[str],
        top_k: int,
        filters: dict[str, Any],
        trace: "TraceContext | None",
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], dict[str, str]]:
        if self.strategy == "dense":
            assert self.dense_retriever is not None
            return self.dense_retriever.retrieve(query, top_k, filters, trace), [], {}
        if self.strategy == "sparse":
            assert self.sparse_retriever is not None
            return [], self.sparse_retriever.retrieve(keywords, top_k, trace), {}
        assert self.dense_retriever is not None and self.sparse_retriever is not None
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.dense_retriever.retrieve,
                query,
                top_k,
                filters,
                trace,
            )
            sparse_future = executor.submit(
                self.sparse_retriever.retrieve,
                keywords,
                top_k,
                trace,
            )

            try:
                dense_results = dense_future.result()
            except Exception as exc:
                dense_results = []
                errors["dense"] = str(exc)

            try:
                sparse_results = sparse_future.result()
            except Exception as exc:
                sparse_results = []
                errors["sparse"] = str(exc)

        if not dense_results and not sparse_results and errors:
            raise RuntimeError(f"both retrieval routes failed: {errors}")

        return dense_results, sparse_results, errors

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Apply exact-match metadata filters as a post-filter safety net."""
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        if filters is None:
            return list(candidates)
        if not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping when provided")
        if not filters:
            return list(candidates)

        filtered: list[RetrievalResult] = []
        for candidate in candidates:
            if not isinstance(candidate, RetrievalResult):
                raise ValueError("candidates must contain RetrievalResult items")
            if _matches_filters(candidate.metadata, filters):
                filtered.append(candidate)
        return filtered


def _matches_filters(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if key not in metadata:
            continue
        actual = metadata[key]
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False
    return True


__all__ = ["HybridSearch"]
