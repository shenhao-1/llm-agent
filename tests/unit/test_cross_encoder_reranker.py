"""Unit tests for cross-encoder reranker provider."""

from __future__ import annotations

import pytest

from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


@pytest.fixture(autouse=True)
def reset_factory_registry():
    original = dict(RerankerFactory._providers)
    RerankerFactory._providers = {"none": RerankerFactory._providers.get("none")}
    yield
    RerankerFactory._providers = original


@pytest.mark.unit
def test_factory_routes_cross_encoder_provider() -> None:
    reranker = RerankerFactory.create({"rerank": {"provider": "cross_encoder", "top_m": 2}})

    assert isinstance(reranker, CrossEncoderReranker)


@pytest.mark.unit
def test_cross_encoder_reranker_sorts_top_m_candidates_by_score() -> None:
    scores = {"alpha": 0.2, "beta": 0.9, "gamma": 0.5}

    reranker = CrossEncoderReranker(
        {"provider": "cross_encoder", "top_m": 2},
        scorer=lambda query, text: scores[text],
    )

    results = reranker.rerank(
        "find beta",
        [
            {"id": "chunk-1", "text": "alpha"},
            {"id": "chunk-2", "text": "beta"},
            {"id": "chunk-3", "text": "gamma"},
        ],
    )

    assert [item["id"] for item in results] == ["chunk-2", "chunk-1", "chunk-3"]


@pytest.mark.unit
def test_cross_encoder_reranker_surfaces_timeout() -> None:
    def timed_out(query: str, text: str) -> float:
        raise TimeoutError("too slow")

    reranker = CrossEncoderReranker(
        {"provider": "cross_encoder", "top_m": 1},
        scorer=timed_out,
    )

    with pytest.raises(RuntimeError, match="cross_encoder reranker timed out"):
        reranker.rerank("find alpha", [{"id": "chunk-1", "text": "alpha"}])


@pytest.mark.unit
def test_cross_encoder_reranker_surfaces_failure() -> None:
    def broken(query: str, text: str) -> float:
        raise ValueError("bad model")

    reranker = CrossEncoderReranker(
        {"provider": "cross_encoder", "top_m": 1},
        scorer=broken,
    )

    with pytest.raises(RuntimeError, match="cross_encoder reranker failed"):
        reranker.rerank("find alpha", [{"id": "chunk-1", "text": "alpha"}])
