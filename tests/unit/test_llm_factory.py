"""Unit tests for LLM factory routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """Simple fake provider used for routing tests."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.validate_messages(messages)
        return "fake-response"


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep factory registry isolated between tests."""
    original = dict(LLMFactory._providers)
    LLMFactory._providers.clear()
    yield
    LLMFactory._providers = original


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_mapping_settings() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    settings = {"llm": {"provider": "fake", "model": "fake-model"}}

    instance = LLMFactory.create(settings)

    assert isinstance(instance, FakeLLM)
    assert instance.config["model"] == "fake-model"
    assert instance.chat([{"role": "user", "content": "hi"}]) == "fake-response"


@pytest.mark.unit
def test_create_routes_to_registered_provider_with_settings_dataclass() -> None:
    LLMFactory.register_provider("fake", FakeLLM)
    settings = Settings(
        llm={"provider": "fake"},
        embedding={"provider": "dummy"},
        vector_store={"provider": "dummy"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"level": "INFO"},
    )

    instance = LLMFactory.create(settings)

    assert isinstance(instance, FakeLLM)


@pytest.mark.unit
def test_create_raises_on_unknown_provider() -> None:
    settings = {"llm": {"provider": "not-registered"}}

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(settings)
