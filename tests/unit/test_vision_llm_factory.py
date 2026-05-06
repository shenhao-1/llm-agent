"""Unit tests for Vision LLM factory routing."""

from __future__ import annotations

from typing import Any

import pytest

from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        self.validate_text(text)
        self.validate_image_input(image_path)
        return "vision-response"


@pytest.fixture(autouse=True)
def reset_factory_registries() -> Any:
    original_llm = dict(LLMFactory._providers)
    original_vision = dict(LLMFactory._vision_providers)
    LLMFactory._providers.clear()
    LLMFactory._vision_providers.clear()
    yield
    LLMFactory._providers = original_llm
    LLMFactory._vision_providers = original_vision


@pytest.mark.unit
def test_register_and_create_vision_provider() -> None:
    LLMFactory.register_vision_provider("fake-vision", FakeVisionLLM)

    instance = LLMFactory.create_vision_llm(
        {"llm": {"provider": "fake-vision", "model": "vision-model"}}
    )

    assert isinstance(instance, FakeVisionLLM)
    assert instance.chat_with_image("describe image", b"raw-image") == "vision-response"


@pytest.mark.unit
def test_create_vision_llm_raises_on_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown vision LLM provider"):
        LLMFactory.create_vision_llm({"llm": {"provider": "missing"}})


@pytest.mark.unit
def test_base_vision_llm_validates_image_input() -> None:
    with pytest.raises(ValueError, match="image_path must be a file path string or raw bytes"):
        FakeVisionLLM().chat_with_image("describe image", 123)  # type: ignore[arg-type]
