"""OpenAI-compatible chat completion providers."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base_llm import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Shared HTTP client logic for OpenAI-style chat completion APIs."""

    provider_name = "openai-compatible"
    default_base_url = ""
    api_key_env_var: str | None = None

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model = self._require_string("model")
        self.base_url = self._resolve_base_url()
        self.api_key = self._resolve_api_key()
        self.timeout = self._resolve_timeout()

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.validate_messages(messages)
        response = self._post_json(
            url=self._build_url(),
            headers=self._build_headers(),
            payload=self._build_payload(messages),
        )
        return self._extract_text(response)

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update(self._optional_payload_fields())
        return payload

    def _optional_payload_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_tokens", "response_format"):
            if key in self.config and self.config[key] is not None:
                payload[key] = self.config[key]
        return payload

    def _resolve_base_url(self) -> str:
        base_url = str(self.config.get("base_url", self.default_base_url)).strip()
        if not base_url:
            raise ValueError(f"{self.provider_name} config missing base_url")
        return base_url

    def _resolve_api_key(self) -> str:
        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key and self.api_key_env_var:
            api_key = os.getenv(self.api_key_env_var, "").strip()
        if not api_key:
            raise ValueError(f"{self.provider_name} config missing api_key")
        return api_key

    def _resolve_timeout(self) -> float:
        timeout = self.config.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"{self.provider_name} timeout must be a positive number")
        return float(timeout)

    def _require_string(self, key: str) -> str:
        value = str(self.config.get(key, "")).strip()
        if not value:
            raise ValueError(f"{self.provider_name} config missing {key}")
        return value

    def _post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = self._extract_error_detail(exc.read())
            raise RuntimeError(
                f"{self.provider_name} request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"{self.provider_name} request failed (URLError): {exc.reason}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"{self.provider_name} request failed ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            payload_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.provider_name} returned invalid JSON (JSONDecodeError)"
            ) from exc

        if not isinstance(payload_dict, dict):
            raise RuntimeError(f"{self.provider_name} returned invalid response structure")
        return payload_dict

    def _extract_text(self, response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"{self.provider_name} returned invalid response structure")

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise RuntimeError(f"{self.provider_name} returned invalid response structure")

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError(f"{self.provider_name} returned invalid response structure")

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        raise RuntimeError(f"{self.provider_name} returned invalid response structure")

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
            if isinstance(error, Mapping):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message
        return raw_text


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI chat completion provider."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env_var = "OPENAI_API_KEY"


__all__ = ["OpenAICompatibleLLM", "OpenAILLM"]
