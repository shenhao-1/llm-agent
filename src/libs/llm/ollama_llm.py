"""Ollama chat completion provider."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama provider using the local HTTP chat endpoint."""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model = self._require_string("model")
        self.base_url = str(self.config.get("base_url", self.default_base_url)).strip()
        if not self.base_url:
            raise ValueError("ollama config missing base_url")
        timeout = self.config.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("ollama timeout must be a positive number")
        self.timeout = float(timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.validate_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if "options" in self.config and self.config["options"] is not None:
            payload["options"] = self.config["options"]

        request = Request(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = self._extract_error_detail(exc.read())
            raise RuntimeError(f"ollama request failed (HTTP {exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"ollama request failed (URLError): {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(
                f"ollama request failed ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ollama returned invalid JSON (JSONDecodeError)") from exc

        if not isinstance(response_payload, dict):
            raise RuntimeError("ollama returned invalid response structure")

        message = response_payload.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("ollama returned invalid response structure")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("ollama returned invalid response structure")
        return content

    @staticmethod
    def _extract_error_detail(raw_bytes: bytes) -> str:
        raw_text = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_text:
            return "empty response body"
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return error
        return raw_text

    def _require_string(self, key: str) -> str:
        value = str(self.config.get(key, "")).strip()
        if not value:
            raise ValueError(f"ollama config missing {key}")
        return value


__all__ = ["OllamaLLM"]
