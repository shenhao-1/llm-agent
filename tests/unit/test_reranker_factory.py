"""Unit tests for reranker factory routing."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """Simple fake reranker used for routing tests."""

    def rerank(
        self,
        query: str,
        candidates: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return [dict(candidate) for candidate in reversed(candidates)]


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests while preserving none fallback."""
    original = dict(RerankerFactory._providers)
    RerankerFactory._providers = {"none": NoneReranker}
    yield
    RerankerFactory._providers = original


@pytest.mark.unit
def test_create_returns_none_reranker_for_none_provider() -> None:
    settings = {"rerank": {"provider": "none", "top_n": 5}}

    instance = RerankerFactory.create(settings)

    assert isinstance(instance, NoneReranker)
    assert instance.config["top_n"] == 5


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    RerankerFactory.register_provider("fake", FakeReranker)
    settings = Settings(
        llm={"provider": "dummy"},
        embedding={"provider": "dummy"},
        vector_store={"provider": "dummy"},
        retrieval={"top_k": 5},
        rerank={"provider": "fake"},
        evaluation={"provider": "custom"},
        observability={"level": "INFO"},
    )

    instance = RerankerFactory.create(settings)

    assert isinstance(instance, FakeReranker)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"rerank": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown reranker provider"):
        RerankerFactory.create(settings)


@pytest.mark.unit
def test_create_raises_on_missing_rerank_section() -> None:
    with pytest.raises(ValueError, match="Missing or invalid settings section: rerank"):
        RerankerFactory.create({})


@pytest.mark.unit
def test_create_rejects_non_mapping_settings() -> None:
    with pytest.raises(TypeError, match="settings must be Settings or Mapping"):
        RerankerFactory.create(["bad"])  # type: ignore[arg-type]


@pytest.mark.unit
def test_create_defaults_blank_provider_to_none_reranker() -> None:
    instance = RerankerFactory.create({"rerank": {"provider": "   "}})

    assert isinstance(instance, NoneReranker)


@pytest.mark.unit
def test_register_provider_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="Provider name must be a non-empty string"):
        RerankerFactory.register_provider("", FakeReranker)


@pytest.mark.unit
def test_register_provider_rejects_non_reranker_class() -> None:
    with pytest.raises(TypeError, match="provider_cls must inherit from BaseReranker"):
        RerankerFactory.register_provider("bad", object)  # type: ignore[arg-type]


@pytest.mark.unit
def test_none_reranker_preserves_input_order() -> None:
    reranker = NoneReranker()
    candidates = [
        {"id": "chunk-1", "text": "alpha"},
        {"id": "chunk-2", "text": "beta"},
        {"id": "chunk-3", "text": "gamma"},
    ]

    results = reranker.rerank("find alpha", candidates)

    assert [item["id"] for item in results] == ["chunk-1", "chunk-2", "chunk-3"]


@pytest.mark.unit
def test_validate_candidates_rejects_missing_text() -> None:
    with pytest.raises(ValueError, match=r"candidates\[0\]\.text"):
        NoneReranker().rerank("query", [{"id": "chunk-1"}])


@pytest.mark.unit
def test_validate_candidates_rejects_non_mapping_candidate() -> None:
    with pytest.raises(ValueError, match=r"candidates\[0\] must be a mapping"):
        NoneReranker().rerank("query", ["bad"])  # type: ignore[list-item]


@pytest.mark.unit
def test_validate_candidates_rejects_non_mapping_metadata() -> None:
    with pytest.raises(ValueError, match=r"candidates\[0\]\.metadata"):
        NoneReranker().rerank(
            "query",
            [{"id": "chunk-1", "text": "alpha", "metadata": "bad"}],
        )


@pytest.mark.unit
def test_validate_query_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        NoneReranker().rerank("   ", [{"id": "chunk-1", "text": "alpha"}])
