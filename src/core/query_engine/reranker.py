"""Core reranker orchestration with fallback behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Mapping, TYPE_CHECKING

from core.types import RetrievalResult
from libs.reranker import BaseReranker, RerankerFactory

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext


class Reranker:
    """Apply a reranker backend and safely fall back to fusion order."""

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        backend: BaseReranker | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if settings is None and backend is None:
            raise ValueError("settings or backend must be provided")

        self.settings = settings or {}
        self.backend = backend or RerankerFactory.create(self.settings)
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else self._extract_timeout(self.settings)
        )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Return reranked candidates, falling back to original order on failure."""
        self._validate_inputs(query, candidates)
        if not candidates:
            self._record_trace(trace, False, 0, None)
            return []

        try:
            raw_results = self._run_backend(query, candidates, trace)
            results = self._normalize_backend_results(candidates, raw_results)
            self._record_trace(trace, False, len(results), None)
            return results
        except Exception as exc:
            fallback = self._fallback_results(candidates, str(exc))
            self._record_trace(trace, True, len(fallback), str(exc))
            return fallback

    def _run_backend(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: "TraceContext | None",
    ) -> list[dict[str, Any]]:
        payload = [self._candidate_to_mapping(candidate) for candidate in candidates]
        if self.timeout_seconds is None:
            return self.backend.rerank(query, payload, trace=trace)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.backend.rerank, query, payload, trace)
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(f"reranker timed out after {self.timeout_seconds}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _normalize_backend_results(
        self,
        original: list[RetrievalResult],
        raw_results: list[Mapping[str, Any]],
    ) -> list[RetrievalResult]:
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError("reranker backend returned no results")

        original_by_id = {result.chunk_id: result for result in original}
        normalized: list[RetrievalResult] = []
        seen: set[str] = set()

        for item in raw_results:
            if not isinstance(item, Mapping):
                raise ValueError("reranker backend result must be a mapping")
            chunk_id = item.get("chunk_id", item.get("id"))
            if not isinstance(chunk_id, str) or chunk_id not in original_by_id:
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            source = original_by_id[chunk_id]
            metadata = dict(source.metadata)
            backend_metadata = item.get("metadata")
            if isinstance(backend_metadata, Mapping):
                metadata.update(dict(backend_metadata))
            if "rerank_score" in item:
                metadata["rerank_score"] = item["rerank_score"]

            score = item.get("score", source.score)
            if not isinstance(score, (int, float)):
                score = source.score
            text = item.get("text", source.text)
            if not isinstance(text, str):
                text = str(text)

            normalized.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=text,
                    metadata=metadata,
                )
            )

        if not normalized:
            raise ValueError("reranker backend returned no matching candidates")
        return normalized

    @staticmethod
    def _candidate_to_mapping(candidate: RetrievalResult) -> dict[str, Any]:
        return {
            "id": candidate.chunk_id,
            "text": candidate.text,
            "metadata": dict(candidate.metadata),
            "score": candidate.score,
        }

    @staticmethod
    def _fallback_results(
        candidates: list[RetrievalResult],
        reason: str,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                score=candidate.score,
                text=candidate.text,
                metadata={
                    **dict(candidate.metadata),
                    "rerank_fallback": True,
                    "rerank_error": reason,
                },
            )
            for candidate in candidates
        ]

    @staticmethod
    def _validate_inputs(query: str, candidates: Any) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, RetrievalResult):
                raise ValueError(f"candidates[{idx}] must be a RetrievalResult")

    @staticmethod
    def _record_trace(
        trace: "TraceContext | None",
        fallback: bool,
        result_count: int,
        error: str | None,
    ) -> None:
        if trace is None:
            return
        details: dict[str, Any] = {
            "fallback": fallback,
            "result_count": result_count,
        }
        if error:
            details["error"] = error
        trace.record_stage("rerank", method="core_reranker", details=details)

    @staticmethod
    def _extract_timeout(settings: "Settings | Mapping[str, Any]") -> float | None:
        rerank: Mapping[str, Any] = {}
        if hasattr(settings, "rerank"):
            candidate = getattr(settings, "rerank")
            if isinstance(candidate, Mapping):
                rerank = candidate
        elif isinstance(settings, Mapping):
            candidate = settings.get("rerank")
            if isinstance(candidate, Mapping):
                rerank = candidate

        timeout = rerank.get("timeout_seconds")
        if timeout is None:
            timeout = rerank.get("timeout")
        if timeout is None:
            return None
        if not isinstance(timeout, (int, float)):
            raise ValueError("rerank timeout must be a number")
        return float(timeout)


__all__ = ["Reranker"]
