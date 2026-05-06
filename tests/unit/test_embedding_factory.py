"""Unit tests for embedding factory routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Simple fake provider used for routing tests."""

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.validate_texts(texts)
        return [[float(len(text))] for text in texts]


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests."""
    original = dict(EmbeddingFactory._providers)
    EmbeddingFactory._providers.clear()
    yield
    EmbeddingFactory._providers = original


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_mapping_settings() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    settings = {"embedding": {"provider": "fake", "model": "fake-embedding"}}

    instance = EmbeddingFactory.create(settings)

    assert isinstance(instance, FakeEmbedding)
    assert instance.config["model"] == "fake-embedding"
    assert instance.embed(["abc", "hello"]) == [[3.0], [5.0]]


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    EmbeddingFactory.register_provider("fake", FakeEmbedding)
    settings = Settings(
        llm={"provider": "dummy"},
        embedding={"provider": "fake"},
        vector_store={"provider": "dummy"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"level": "INFO"},
    )

    instance = EmbeddingFactory.create(settings)

    assert isinstance(instance, FakeEmbedding)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"embedding": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.create(settings)
