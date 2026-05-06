"""OpenAI-compatible embedding providers."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base_embedding import BaseEmbedding


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Shared HTTP client logic for OpenAI-style embedding APIs."""

    provider_name = "openai-compatible"
    default_base_url = ""
    api_key_env_var: str | None = None

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model = self._require_string("model")
        self.base_url = self._resolve_base_url()
        self.api_key = self._resolve_api_key()
        self.timeout = self._resolve_timeout()
        self.max_input_length = self._resolve_max_input_length()

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        normalized_texts = self._normalize_texts(texts)
        payload = {"model": self.model, "input": normalized_texts}
        payload.update(self._optional_payload_fields())

        response = self._post_json(
            url=self._build_url(),
            headers=self._build_headers(),
            payload=payload,
        )
        return self._extract_vectors(response)

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _optional_payload_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if "dimensions" in self.config and self.config["dimensions"] is not None:
            payload["dimensions"] = self.config["dimensions"]
        if "encoding_format" in self.config and self.config["encoding_format"] is not None:
            payload["encoding_format"] = self.config["encoding_format"]
        return payload

    def _normalize_texts(self, texts: list[str]) -> list[str]:
        normalized: list[str] = []
        for idx, text in enumerate(texts):
            if len(text) > self.max_input_length:
                raise ValueError(
                    f"{self.provider_name} texts[{idx}] exceeds max_input_length={self.max_input_length}"
                )
            normalized.append(text)
        return normalized

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

    def _resolve_max_input_length(self) -> int:
        max_input_length = self.config.get("max_input_length", 8192)
        if not isinstance(max_input_length, int) or max_input_length <= 0:
            raise ValueError(
                f"{self.provider_name} max_input_length must be a positive integer"
            )
        return max_input_length

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

    def _extract_vectors(self, response: Mapping[str, Any]) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"{self.provider_name} returned invalid response structure")

        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise RuntimeError(f"{self.provider_name} returned invalid response structure")
            embedding = item.get("embedding")
            self.validate_vector(embedding)
            vectors.append([float(value) for value in embedding])
        return vectors

    @staticmethod
    def validate_vector(vector: Any) -> None:
        if not isinstance(vector, list) or not vector:
            raise ValueError("embedding vector must be a non-empty list")
        for idx, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise ValueError(f"embedding vector[{idx}] must be a number")

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


class OpenAIEmbedding(OpenAICompatibleEmbedding):
    """OpenAI embedding provider."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env_var = "OPENAI_API_KEY"


__all__ = ["OpenAICompatibleEmbedding", "OpenAIEmbedding"]
