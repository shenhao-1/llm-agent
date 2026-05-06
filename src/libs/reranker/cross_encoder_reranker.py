"""Cross-encoder reranker provider."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Deterministic reranker that delegates pair scoring to an injected scorer."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        super().__init__(config)
        self.top_m = self._resolve_top_m()
        self.timeout_seconds = self._resolve_timeout_seconds()
        self._scorer = scorer or self._default_scorer

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, candidate in enumerate(candidates[: self.top_m]):
            try:
                score = float(self._scorer(query, str(candidate["text"])))
            except TimeoutError as exc:
                raise RuntimeError("cross_encoder reranker timed out") from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("cross_encoder reranker failed") from exc
            scored.append((score, index, dict(candidate)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [item[2] for item in scored]
        ranked.extend(dict(candidate) for candidate in candidates[self.top_m :])
        return ranked

    def _resolve_top_m(self) -> int:
        top_m = self.config.get("top_m", self.config.get("top_n", 10))
        if not isinstance(top_m, int) or top_m <= 0:
            raise ValueError("cross_encoder reranker top_m must be a positive integer")
        return top_m

    def _resolve_timeout_seconds(self) -> float:
        timeout = self.config.get("timeout_seconds", 10)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(
                "cross_encoder reranker timeout_seconds must be a positive number"
            )
        return float(timeout)

    @staticmethod
    def _default_scorer(query: str, text: str) -> float:
        raise RuntimeError("cross_encoder scorer is not configured")


__all__ = ["CrossEncoderReranker"]
