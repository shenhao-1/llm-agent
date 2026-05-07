"""Unit tests for metadata enrichment contract."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core import Chunk
from core.trace import TraceContext
from ingestion.transform import MetadataEnricher
from libs.llm.base_llm import BaseLLM


class JsonLLM(BaseLLM):
    """Fake LLM returning rich JSON metadata."""

    def __init__(self, response: str, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.validate_messages(messages)
        self.calls.append(messages)
        return self.response


class FailingLLM(BaseLLM):
    """Fake LLM simulating provider failure."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("metadata provider failed")


class ExplodingMetadataEnricher(MetadataEnricher):
    """Enricher that simulates a per-chunk unexpected failure."""

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        if "explode" in text:
            raise RuntimeError("bad metadata chunk")
        return super()._rule_based_enrich(text)


def test_rule_mode_adds_non_empty_title_summary_and_tags() -> None:
    chunk = _chunk(
        "Hybrid search combines BM25 lexical matching with dense embedding retrieval."
    )

    enriched = MetadataEnricher().transform([chunk])[0]

    assert enriched.metadata["title"]
    assert enriched.metadata["summary"]
    assert enriched.metadata["tags"]
    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert "hybrid" in enriched.metadata["tags"]
    assert enriched.text == chunk.text


def test_rule_mode_preserves_existing_metadata() -> None:
    chunk = _chunk("A short chunk.", metadata={"source_path": "docs/a.md", "chunk_index": 3})

    enriched = MetadataEnricher().transform([chunk])[0]

    assert enriched.metadata["source_path"] == "docs/a.md"
    assert enriched.metadata["chunk_index"] == 3
    assert enriched.metadata["title"] == "A short chunk"


def test_llm_mode_overrides_rule_metadata_with_valid_json() -> None:
    llm = JsonLLM(
        '{"title": "Hybrid Retrieval", "summary": "Combines sparse and dense search.", '
        '"tags": ["hybrid-search", "bm25", "embeddings"]}'
    )

    enriched = MetadataEnricher(
        {"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=llm,
    ).transform([_chunk("Hybrid search combines BM25 and embeddings.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "llm"
    assert enriched.metadata["title"] == "Hybrid Retrieval"
    assert enriched.metadata["summary"] == "Combines sparse and dense search."
    assert enriched.metadata["tags"] == ["hybrid-search", "bm25", "embeddings"]
    assert "Hybrid search combines" in llm.calls[0][1]["content"]


def test_llm_response_can_include_surrounding_text() -> None:
    llm = JsonLLM(
        'Here is the metadata:\n{"title": "RAG", "summary": "Retrieval augmented generation.", '
        '"tags": ["rag"]}\nDone.'
    )

    enriched = MetadataEnricher(
        {"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=llm,
    ).transform([_chunk("RAG connects retrieval and generation.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "llm"
    assert enriched.metadata["title"] == "RAG"


def test_invalid_llm_metadata_falls_back_to_rule_result() -> None:
    llm = JsonLLM('{"title": "", "summary": "bad", "tags": []}')

    enriched = MetadataEnricher(
        {"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=llm,
    ).transform([_chunk("Useful fallback text about retrieval.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert enriched.metadata["title"] == "Useful fallback text about retrieval"
    assert "title" in enriched.metadata["metadata_fallback_reason"]


def test_llm_failure_falls_back_without_raising() -> None:
    enriched = MetadataEnricher(
        {"ingestion": {"metadata_enricher": {"use_llm": True}}},
        llm=FailingLLM(),
    ).transform([_chunk("Fallback should survive provider errors.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert "metadata provider failed" in enriched.metadata["metadata_fallback_reason"]


def test_use_llm_false_does_not_call_passed_llm() -> None:
    llm = JsonLLM('{"title": "unused", "summary": "unused", "tags": ["unused"]}')

    enriched = MetadataEnricher(
        {"ingestion": {"metadata_enricher": {"use_llm": False}}},
        llm=llm,
    ).transform([_chunk("Rule only text.")])[0]

    assert enriched.metadata["metadata_enriched_by"] == "rule"
    assert llm.calls == []


def test_single_chunk_exception_preserves_original_and_continues() -> None:
    chunks = [_chunk("explode"), _chunk("The second chunk survives.", chunk_id="chunk-2")]

    enriched = ExplodingMetadataEnricher().transform(chunks)

    assert enriched[0].metadata["metadata_enriched_by"] == "fallback"
    assert enriched[0].metadata["metadata_enrich_error"] == "bad metadata chunk"
    assert enriched[0].metadata["title"]
    assert enriched[1].metadata["metadata_enriched_by"] == "rule"


def test_trace_records_metadata_enricher_stage() -> None:
    trace = TraceContext()

    MetadataEnricher().transform([_chunk("Trace me.")], trace=trace)

    assert trace.stages[-1]["name"] == "transform"
    assert trace.stages[-1]["method"] == "metadata_enricher"
    assert trace.stages[-1]["details"]["chunk_count"] == 1


def test_transform_rejects_non_chunk_entries() -> None:
    with pytest.raises(ValueError, match=r"chunks\[0\]"):
        MetadataEnricher().transform(["bad"])


def _chunk(
    text: str,
    chunk_id: str = "chunk-1",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=metadata or {"source_path": "docs/sample.md"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )
