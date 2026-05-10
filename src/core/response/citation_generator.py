"""Citation generation for MCP retrieval responses."""

from __future__ import annotations

from typing import Any

from core.types import RetrievalResult


class CitationGenerator:
    """Convert retrieval results into stable structured citations."""

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
        if not isinstance(retrieval_results, list):
            raise ValueError("retrieval_results must be a list")

        citations: list[dict[str, Any]] = []
        for index, result in enumerate(retrieval_results, start=1):
            if not isinstance(result, RetrievalResult):
                raise ValueError(f"retrieval_results[{index - 1}] must be a RetrievalResult")
            metadata = dict(result.metadata)
            citations.append(
                {
                    "index": index,
                    "source": str(metadata.get("source_path", "unknown")),
                    "page": metadata.get("page", metadata.get("page_num")),
                    "chunk_id": result.chunk_id,
                    "score": float(result.score),
                }
            )
        return citations


__all__ = ["CitationGenerator"]
