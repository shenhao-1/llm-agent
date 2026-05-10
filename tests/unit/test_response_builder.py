"""Tests for MCP response construction and query tool output."""

from __future__ import annotations

import pytest

from core.response import CitationGenerator, ResponseBuilder
from core.types import RetrievalResult
from mcp_server.tools.query_knowledge_hub import query_knowledge_hub


def test_response_builder_returns_markdown_and_structured_citations() -> None:
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.91,
            text="Azure setup requires endpoint and API key.",
            metadata={"source_path": "docs/azure.pdf", "page": 2},
        )
    ]

    response = ResponseBuilder().build(results, "How to configure Azure?")

    assert response["content"][0]["type"] == "text"
    assert "[1]" in response["content"][0]["text"]
    assert "Azure setup" in response["content"][0]["text"]
    assert response["structuredContent"]["citations"] == [
        {
            "index": 1,
            "source": "docs/azure.pdf",
            "page": 2,
            "chunk_id": "chunk-1",
            "score": 0.91,
        }
    ]
    assert response["structuredContent"]["image_count"] == 0


def test_response_builder_returns_friendly_message_for_no_results() -> None:
    response = ResponseBuilder().build([], "missing topic")

    assert "No relevant documents found" in response["content"][0]["text"]
    assert response["structuredContent"]["citations"] == []
    assert response["structuredContent"]["result_count"] == 0
    assert response["structuredContent"]["image_count"] == 0


def test_response_builder_appends_image_content(tmp_path) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"png-bytes")
    response = ResponseBuilder().build(
        [
            RetrievalResult(
                chunk_id="chunk-img",
                score=0.8,
                text="diagram result",
                metadata={
                    "source_path": "docs/diagram.pdf",
                    "image_refs": ["img-1"],
                    "images": [{"id": "img-1", "path": str(image_path)}],
                },
            )
        ],
        "diagram",
    )

    assert response["content"][0]["type"] == "text"
    assert response["content"][1]["type"] == "image"
    assert response["content"][1]["mimeType"] == "image/png"
    assert response["content"][1]["data"]
    assert response["structuredContent"]["image_count"] == 1


def test_citation_generator_validates_retrieval_results() -> None:
    with pytest.raises(ValueError, match="RetrievalResult"):
        CitationGenerator().generate([{"bad": "shape"}])  # type: ignore[list-item]


def test_query_knowledge_hub_uses_search_rerank_and_response_builder() -> None:
    search = FakeSearch(
        [
            RetrievalResult(
                chunk_id="chunk-2",
                score=0.5,
                text="second result",
                metadata={"source_path": "docs/second.pdf", "page": 4},
            ),
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="first result",
                metadata={"source_path": "docs/first.pdf", "page": 1},
            ),
        ]
    )
    reranker = FakeReranker()

    response = query_knowledge_hub(
        "rank these",
        top_k=2,
        collection="docs",
        search=search,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
    )

    assert search.calls == [
        {"query": "rank these", "top_k": 2, "filters": {"collection": "docs"}}
    ]
    assert reranker.calls == [["chunk-2", "chunk-1"]]
    assert response["structuredContent"]["citations"][0]["chunk_id"] == "chunk-1"
    assert "[1]" in response["content"][0]["text"]


def test_query_knowledge_hub_validates_arguments() -> None:
    with pytest.raises(ValueError, match="query"):
        query_knowledge_hub("", search=FakeSearch([]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        query_knowledge_hub("query", top_k=0, search=FakeSearch([]))  # type: ignore[arg-type]


class FakeSearch:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, top_k: int, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.results[:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, candidates: list[RetrievalResult]):
        self.calls.append([candidate.chunk_id for candidate in candidates])
        return list(reversed(candidates))
