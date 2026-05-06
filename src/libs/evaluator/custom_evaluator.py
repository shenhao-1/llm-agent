"""Lightweight built-in evaluator metrics."""

from __future__ import annotations

from .base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute lightweight retrieval metrics without external dependencies."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace=None,
    ) -> dict[str, float]:
        self.validate_query(query)
        self.validate_id_list(retrieved_ids, "retrieved_ids")
        self.validate_id_list(golden_ids, "golden_ids")

        if not golden_ids:
            raise ValueError("golden_ids must be a non-empty list")

        retrieved_set = set(retrieved_ids)
        golden_set = set(golden_ids)
        hit = 1.0 if retrieved_set & golden_set else 0.0

        reciprocal_rank = 0.0
        for index, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in golden_set:
                reciprocal_rank = 1.0 / index
                break

        precision_at_k = (
            len(retrieved_set & golden_set) / len(retrieved_ids) if retrieved_ids else 0.0
        )
        recall_at_k = len(retrieved_set & golden_set) / len(golden_set)

        return {
            "hit_rate": hit,
            "mrr": reciprocal_rank,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
        }
