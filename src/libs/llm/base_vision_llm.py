"""Base contract for all vision-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class BaseVisionLLM(ABC):
    """Abstract vision LLM contract used by multimodal ingestion steps."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        """Return model output for a text prompt with an attached image."""

    @staticmethod
    def validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

    @staticmethod
    def validate_image_input(image_path: str | bytes) -> None:
        if isinstance(image_path, bytes):
            if not image_path:
                raise ValueError("image bytes must not be empty")
            return
        if isinstance(image_path, str):
            if not image_path.strip():
                raise ValueError("image_path must be a non-empty string")
            return
        raise ValueError("image_path must be a file path string or raw bytes")

    @staticmethod
    def load_image_bytes(image_path: str | bytes) -> bytes:
        BaseVisionLLM.validate_image_input(image_path)
        if isinstance(image_path, bytes):
            return image_path
        return Path(image_path).read_bytes()
