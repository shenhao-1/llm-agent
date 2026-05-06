"""Base contract for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseLLM(ABC):
    """Abstract LLM contract used by core and factory layers."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return model output for a chat-style prompt."""

    @staticmethod
    def validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate basic chat message shape used across providers."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{idx}] must be a mapping")

            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"messages[{idx}].role must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError(f"messages[{idx}].content must be a string")
