"""Unit tests for Azure Vision LLM provider."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image

from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.llm_factory import LLMFactory


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "large.png"
    image = Image.new("RGB", (3000, 1200), color=(255, 0, 0))
    image.save(path, format="PNG")
    return path


@pytest.mark.unit
def test_factory_routes_azure_vision_provider() -> None:
    llm = LLMFactory.create_vision_llm(
        {
            "llm": {
                "provider": "azure",
                "azure_endpoint": "https://example.openai.azure.com",
                "deployment_name": "gpt-4o",
                "api_version": "2024-10-21",
                "api_key": "secret-key",
            }
        }
    )

    assert isinstance(llm, AzureVisionLLM)


@pytest.mark.unit
def test_azure_vision_llm_posts_multimodal_payload(
    monkeypatch: pytest.MonkeyPatch, image_file: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"choices": [{"message": {"content": "detected chart"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    llm = AzureVisionLLM(
        {
            "provider": "azure",
            "azure_endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o",
            "api_version": "2024-10-21",
            "api_key": "secret-key",
            "max_image_size": 2048,
        }
    )

    result = llm.chat_with_image("describe image", str(image_file))

    assert result == "detected chart"
    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
    )
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["api-key"] == "secret-key"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe image"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.unit
def test_azure_vision_llm_accepts_raw_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    image = Image.new("RGB", (32, 32), color=(0, 255, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")

    llm = AzureVisionLLM(
        {
            "provider": "azure",
            "azure_endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o",
            "api_version": "2024-10-21",
            "api_key": "secret-key",
        }
    )

    assert llm.chat_with_image("describe image", output.getvalue()) == "ok"


@pytest.mark.unit
def test_azure_vision_llm_surfaces_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise HTTPError(
            url=request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error": {"code": "Unauthorized", "message": "bad api key"}}'
            ),
        )

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    llm = AzureVisionLLM(
        {
            "provider": "azure",
            "azure_endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o",
            "api_version": "2024-10-21",
            "api_key": "bad-key",
        }
    )

    image = Image.new("RGB", (32, 32), color=(0, 255, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")

    with pytest.raises(
        RuntimeError,
        match=r"azure vision request failed \(HTTP 401\): Unauthorized: bad api key",
    ):
        llm.chat_with_image("describe image", output.getvalue())


@pytest.mark.unit
def test_azure_vision_llm_surfaces_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise URLError("timed out")

    monkeypatch.setattr("libs.llm.azure_vision_llm.urlopen", fake_urlopen)

    llm = AzureVisionLLM(
        {
            "provider": "azure",
            "azure_endpoint": "https://example.openai.azure.com",
            "deployment_name": "gpt-4o",
            "api_version": "2024-10-21",
            "api_key": "secret-key",
        }
    )

    image = Image.new("RGB", (32, 32), color=(0, 255, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")

    with pytest.raises(
        RuntimeError,
        match=r"azure vision request failed \(URLError\): timed out",
    ):
        llm.chat_with_image("describe image", output.getvalue())
