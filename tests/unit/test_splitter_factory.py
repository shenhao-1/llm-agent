"""Unit tests for splitter factory routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Simple fake splitter used for routing tests."""

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        self.validate_text(text)
        return [part.strip() for part in text.split("|") if part.strip()]


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests."""
    original = dict(SplitterFactory._providers)
    SplitterFactory._providers.clear()
    yield
    SplitterFactory._providers = original


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_mapping_settings() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)
    settings = {"splitter": {"provider": "fake", "chunk_size": 128}}

    instance = SplitterFactory.create(settings)

    assert isinstance(instance, FakeSplitter)
    assert instance.config["chunk_size"] == 128
    assert instance.split_text("a|b|c") == ["a", "b", "c"]


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    SplitterFactory.register_provider("fake", FakeSplitter)
    settings = Settings(
        llm={"provider": "dummy"},
        embedding={"provider": "dummy"},
        vector_store={"provider": "dummy"},
        retrieval={"top_k": 5, "splitter": {"provider": "fake"}},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"level": "INFO"},
    )

    instance = SplitterFactory.create(settings)

    assert isinstance(instance, FakeSplitter)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"splitter": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown splitter provider"):
        SplitterFactory.create(settings)
