"""Reciprocal Rank Fusion for dense and sparse retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from core.types import RetrievalResult

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass
class _FusionCandidate:
    result: RetrievalResult
    score: float = 0.0
    best_rank: int = 10**9


class RRFFusion:
    """Fuse dense and sparse rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        k: int = 60,
        *,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(dense_weight, (int, float)) or dense_weight <= 0:
            raise ValueError("dense_weight must be a positive number")
        if not isinstance(sparse_weight, (int, float)) or sparse_weight <= 0:
            raise ValueError("sparse_weight must be a positive number")

        self.k = k
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Return a deterministic RRF-ranked list."""
        self._validate_results(dense_results, "dense_results")
        self._validate_results(sparse_results, "sparse_results")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer when provided")

        candidates: dict[str, _FusionCandidate] = {}
        self._add_ranking(candidates, dense_results, self.dense_weight)
        self._add_ranking(candidates, sparse_results, self.sparse_weight)

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.best_rank,
                candidate.result.chunk_id,
            ),
        )
        fused = [
            RetrievalResult(
                chunk_id=candidate.result.chunk_id,
                score=candidate.score,
                text=candidate.result.text,
                metadata=dict(candidate.result.metadata),
            )
            for candidate in ranked[:top_k]
        ]

        if trace is not None:
            trace.record_stage(
                "fusion",
                method="rrf",
                details={
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "result_count": len(fused),
                    "k": self.k,
                },
            )

        return fused

    def _add_ranking(
        self,
        candidates: dict[str, _FusionCandidate],
        results: list[RetrievalResult],
        weight: float,
    ) -> None:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)

            score = weight / (self.k + rank)
            candidate = candidates.get(result.chunk_id)
            if candidate is None:
                candidates[result.chunk_id] = _FusionCandidate(
                    result=result,
                    score=score,
                    best_rank=rank,
                )
                continue

            candidate.score += score
            candidate.best_rank = min(candidate.best_rank, rank)
            candidate.result = self._merge_result(candidate.result, result)

    @staticmethod
    def _merge_result(
        existing: RetrievalResult,
        incoming: RetrievalResult,
    ) -> RetrievalResult:
        metadata = dict(existing.metadata)
        metadata.update(incoming.metadata)
        text = existing.text or incoming.text
        return RetrievalResult(
            chunk_id=existing.chunk_id,
            score=existing.score,
            text=text,
            metadata=metadata,
        )

    @staticmethod
    def _validate_results(results: Any, field_name: str) -> None:
        if not isinstance(results, list):
            raise ValueError(f"{field_name} must be a list")
        for idx, result in enumerate(results):
            if not isinstance(result, RetrievalResult):
                raise ValueError(f"{field_name}[{idx}] must be a RetrievalResult")


__all__ = ["RRFFusion"]
