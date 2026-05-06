"""Smoke tests for OpenAI and Azure embedding providers."""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError

import pytest

from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def reset_factory_registry() -> Any:
    """Keep embedding factory registry isolated between tests."""
    original = dict(EmbeddingFactory._providers)
    EmbeddingFactory._providers.clear()
    yield
    EmbeddingFactory._providers = original


@pytest.mark.unit
def test_factory_routes_openai_embedding_provider() -> None:
    embedding = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "test-key",
            }
        }
    )

    assert isinstance(embedding, OpenAIEmbedding)


@pytest.mark.unit
def test_factory_routes_azure_embedding_provider() -> None:
    embedding = EmbeddingFactory.create(
        {
            "embedding": {
                "provider": "azure",
                "deployment_name": "text-embedding-ada-002",
                "azure_endpoint": "https://example.openai.azure.com",
                "api_key": "azure-key",
                "api_version": "2024-10-21",
            }
        }
    )

    assert isinstance(embedding, AzureEmbedding)


@pytest.mark.unit
def test_openai_embedding_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return DummyResponse(
            {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)

    embedding = OpenAIEmbedding(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "secret-key",
            "dimensions": 2,
        }
    )

    vectors = embedding.embed(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["body"] == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
        "dimensions": 2,
    }
    assert captured["timeout"] == 30.0
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


@pytest.mark.unit
def test_azure_embedding_uses_deployment_scoped_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return DummyResponse({"data": [{"embedding": [0.5, 0.6, 0.7]}]})

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)

    embedding = AzureEmbedding(
        {
            "provider": "azure",
            "deployment_name": "text-embedding-ada-002",
            "azure_endpoint": "https://example.openai.azure.com",
            "api_key": "azure-secret",
            "api_version": "2024-10-21",
        }
    )

    vectors = embedding.embed(["hello azure"])

    assert vectors == [[0.5, 0.6, 0.7]]
    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21"
    )
    assert captured["body"] == {"model": "text-embedding-ada-002", "input": ["hello azure"]}
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["api-key"] == "azure-secret"


@pytest.mark.unit
def test_openai_embedding_rejects_overlong_input() -> None:
    embedding = OpenAIEmbedding(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "secret-key",
            "max_input_length": 5,
        }
    )

    with pytest.raises(ValueError, match=r"openai texts\[0\] exceeds max_input_length=5"):
        embedding.embed(["too-long"])


@pytest.mark.unit
def test_azure_embedding_surfaces_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float) -> DummyResponse:
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error": {"message": "rate limited"}}'),
        )

    monkeypatch.setattr("libs.embedding.openai_embedding.urlopen", fake_urlopen)

    embedding = AzureEmbedding(
        {
            "provider": "azure",
            "deployment_name": "text-embedding-ada-002",
            "azure_endpoint": "https://example.openai.azure.com",
            "api_key": "azure-secret",
            "api_version": "2024-10-21",
        }
    )

    with pytest.raises(RuntimeError, match=r"azure request failed \(HTTP 429\): rate limited"):
        embedding.embed(["hello azure"])
