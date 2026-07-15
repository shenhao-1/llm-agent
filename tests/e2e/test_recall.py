"""End-to-end recall regression tests over the golden test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core import Chunk
from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, RRFFusion, SparseRetriever
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from libs.embedding import BaseEmbedding
from libs.evaluator import CustomEvaluator
from libs.vector_store import BaseVectorStore
from observability.evaluation import EvalRunner, load_golden_test_set


GOLDEN_TEST_SET = Path("tests/fixtures/golden_test_set.json")
MIN_HIT_RATE = 1.0
MIN_RECALL_AT_K = 1.0
MIN_MRR = 0.75


def test_golden_set_recall_stays_above_threshold(tmp_path: Path) -> None:
    cases = load_golden_test_set(GOLDEN_TEST_SET)
    chunks = _golden_chunks()
    vector_store = InMemoryVectorStore(chunks)
    bm25 = BM25Indexer(tmp_path / "bm25").build(
        SparseEncoder().encode(chunks),
        persist=False,
    )
    search = HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            embedding_client=KeywordEmbedding(),
            vector_store=vector_store,
        ),
        sparse_retriever=SparseRetriever(
            bm25_indexer=bm25,
            vector_store=vector_store,
        ),
        fusion=RRFFusion(k=20),
    )
    runner = EvalRunner(
        {"retrieval": {"top_k": 5}},
        hybrid_search=search,
        evaluator=CustomEvaluator(),
    )

    report = runner.run(GOLDEN_TEST_SET)

    assert len(report.results) == len(cases)
    assert report.metrics["hit_rate"] >= MIN_HIT_RATE
    assert report.metrics["recall_at_k"] >= MIN_RECALL_AT_K
    assert report.metrics["mrr"] >= MIN_MRR
    for result in report.results:
        assert set(result.expected_chunk_ids).issubset(set(result.retrieved_ids))


class KeywordEmbedding(BaseEmbedding):
    """Deterministic lexical embedding for offline recall regression."""

    vocabulary = [
        "azure",
        "openai",
        "endpoint",
        "bm25",
        "sparse",
        "retrieval",
        "dashboard",
        "evaluation",
        "image",
        "caption",
        "ingestion",
    ]

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        return [_vectorize(text) for text in texts]


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, chunks: list[Chunk]) -> None:
        super().__init__({})
        self.records = {
            chunk.id: {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
                "vector": _vectorize(chunk.text),
            }
            for chunk in chunks
        }

    def upsert(self, records: list[Mapping[str, Any]], trace=None) -> None:
        self.validate_records(records)
        for record in records:
            self.records[str(record["id"])] = dict(record)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        candidates = [
            record
            for record in self.records.values()
            if _metadata_matches(record.get("metadata", {}), filters)
        ]
        ranked = sorted(
            candidates,
            key=lambda record: (
                -_dot(vector, list(record.get("vector", []))),
                str(record["id"]),
            ),
        )
        return [
            {
                "id": str(record["id"]),
                "text": str(record["text"]),
                "metadata": dict(record.get("metadata", {})),
                "score": _dot(vector, list(record.get("vector", []))),
            }
            for record in ranked[:top_k]
        ]

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        return [
            {
                "id": chunk_id,
                "text": str(self.records[chunk_id]["text"]),
                "metadata": dict(self.records[chunk_id].get("metadata", {})),
                "vector": list(self.records[chunk_id].get("vector", [])),
            }
            for chunk_id in ids
            if chunk_id in self.records
        ]


def _golden_chunks() -> list[Chunk]:
    return [
        _chunk(
            "chunk-azure-config",
            "Azure OpenAI endpoint configuration uses deployment names, api versions, "
            "resource endpoints, and secure keys.",
            "docs/azure_config.pdf",
        ),
        _chunk(
            "chunk-bm25-overview",
            "BM25 sparse retrieval is used for lexical keyword recall before hybrid "
            "fusion combines dense and sparse candidates.",
            "docs/bm25_overview.pdf",
        ),
        _chunk(
            "chunk-dashboard-eval",
            "The Dashboard evaluation panel runs the golden test set, selects custom "
            "or ragas backends, and shows hit rate and mrr metrics.",
            "docs/dashboard_evaluation.pdf",
        ),
        _chunk(
            "chunk-image-captioning",
            "The ingestion pipeline generates image captions so visual content can be "
            "embedded, indexed, retrieved, and cited with surrounding text.",
            "docs/image_captioning.pdf",
        ),
        _chunk(
            "chunk-distractor",
            "General project notes about unrelated logging and configuration defaults.",
            "docs/general_notes.pdf",
        ),
    ]


def _chunk(chunk_id: str, text: str, source_path: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "collection": "docs"},
        start_offset=0,
        end_offset=len(text),
        source_ref=source_path,
    )


def _vectorize(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if term in lowered else 0.0
        for term in KeywordEmbedding.vocabulary
    ]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _metadata_matches(
    metadata: Any,
    filters: Mapping[str, Any] | None,
) -> bool:
    if not filters:
        return True
    if not isinstance(metadata, Mapping):
        return False
    return all(metadata.get(key) == value for key, value in filters.items())
