"""Azure Vision LLM provider."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image

from .base_vision_llm import BaseVisionLLM


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI vision provider using deployment-scoped chat completions."""

    provider_name = "azure"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key = self._require_string("api_key")
        self.azure_endpoint = self._require_string("azure_endpoint").rstrip("/")
        self.deployment_name = self._require_string("deployment_name")
        self.api_version = str(self.config.get("api_version", "2024-10-21")).strip()
        if not self.api_version:
            raise ValueError("azure vision config missing api_version")
        timeout = self.config.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("azure vision timeout must be a positive number")
        self.timeout = float(timeout)
        max_image_size = self.config.get("max_image_size", 2048)
        if not isinstance(max_image_size, int) or max_image_size <= 0:
            raise ValueError("azure vision max_image_size must be a positive integer")
        self.max_image_size = max_image_size

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> str:
        self.validate_text(text)
        image_bytes = self._prepare_image_bytes(image_path)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    + base64.b64encode(image_bytes).decode("utf-8")
                                )
                            },
                        },
                    ],
                }
            ]
        }

        request = Request(
            url=self._build_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = self._extract_error_detail(exc.read())
            raise RuntimeError(
                f"azure vision request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"azure vision request failed (URLError): {exc.reason}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"azure vision request failed ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            payload_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "azure vision returned invalid JSON (JSONDecodeError)"
            ) from exc

        return self._extract_text(payload_dict)

    def _build_url(self) -> str:
        deployment = quote(self.deployment_name, safe="")
        query = urlencode({"api-version": self.api_version})
        return (
            f"{self.azure_endpoint}/openai/deployments/"
            f"{deployment}/chat/completions?{query}"
        )

    def _prepare_image_bytes(self, image_path: str | bytes) -> bytes:
        raw = self.load_image_bytes(image_path)
        image = Image.open(io.BytesIO(raw))
        if max(image.size) > self.max_image_size:
            image.thumbnail((self.max_image_size, self.max_image_size))
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def _extract_text(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("azure vision returned invalid response structure")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise RuntimeError("azure vision returned invalid response structure")
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("azure vision returned invalid response structure")
        content = message.get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError("azure vision returned invalid response structure")

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
                code = error.get("code")
                if isinstance(message, str) and isinstance(code, str):
                    return f"{code}: {message}"
                if isinstance(message, str):
                    return message
        return raw_text

    def _require_string(self, key: str) -> str:
        value = str(self.config.get(key, "")).strip()
        if not value:
            raise ValueError(f"azure vision config missing {key}")
        return value


__all__ = ["AzureVisionLLM"]
