"""BM25 inverted-index construction and filesystem persistence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
from typing import Any

from core.types import ChunkRecord
from ingestion.embedding.sparse_encoder import (
    SparseEncoder,
    SparseEncodingResult,
)


DEFAULT_INDEX_DIR = Path("data/db/bm25")
INDEX_FILENAME = "index.json"


class BM25Indexer:
    """Build, persist, and query a BM25 index from sparse term statistics."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: SparseEncoder | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.index_dir = Path(index_dir)
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or SparseEncoder()

        self.index: dict[str, dict[str, Any]] = {}
        self.document_frequency: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}
        self.documents: dict[str, ChunkRecord] = {}
        self.doc_count = 0
        self.avg_doc_length = 0.0

    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    def build(
        self,
        sparse_result: SparseEncodingResult,
        *,
        persist: bool = True,
    ) -> "BM25Indexer":
        """Replace the current index with records from one sparse encoding result."""
        self._validate_sparse_result(sparse_result)
        self.documents = {record.id: record for record in sparse_result.records}
        self.doc_lengths = self._extract_doc_lengths(sparse_result)
        self._recompute_index()
        if persist:
            self.save()
        return self

    def rebuild(
        self,
        sparse_result: SparseEncodingResult,
        *,
        persist: bool = True,
    ) -> "BM25Indexer":
        """Explicit alias for replacing all existing index content."""
        return self.build(sparse_result, persist=persist)

    def update(
        self,
        sparse_result: SparseEncodingResult,
        *,
        persist: bool = True,
    ) -> "BM25Indexer":
        """Add or replace chunk records, then recompute corpus-level BM25 stats."""
        self._validate_sparse_result(sparse_result)
        incoming_lengths = self._extract_doc_lengths(sparse_result)
        for record in sparse_result.records:
            self.documents[record.id] = record
            self.doc_lengths[record.id] = incoming_lengths[record.id]

        self._recompute_index()
        if persist:
            self.save()
        return self

    def save(self) -> Path:
        """Persist the current index to disk as deterministic JSON."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.index_path

    def load(self) -> "BM25Indexer":
        """Load index state from this instance's index directory."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {self.index_path}")

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._load_payload(payload)
        return self

    @classmethod
    def from_disk(cls, index_dir: str | Path = DEFAULT_INDEX_DIR) -> "BM25Indexer":
        """Create an indexer and load persisted index state from disk."""
        return cls(index_dir=index_dir).load()

    def query(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        """Return BM25-ranked chunk ids for a text query."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.doc_count == 0 or self.avg_doc_length <= 0:
            return []

        scores: dict[str, float] = defaultdict(float)
        query_terms = set(self.tokenizer.tokenize(query))
        for term in sorted(query_terms):
            entry = self.index.get(term)
            if not entry:
                continue

            idf = float(entry["idf"])
            for posting in entry["postings"]:
                chunk_id = str(posting["chunk_id"])
                tf = float(posting["tf"])
                doc_length = int(posting["doc_length"])
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                if denominator <= 0:
                    continue
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {
                "chunk_id": chunk_id,
                "score": score,
                "metadata": dict(self.documents[chunk_id].metadata),
            }
            for chunk_id, score in ranked[:top_k]
        ]

    def idf(self, term: str) -> float:
        """Return the stored IDF for a normalized term, or 0 for unknown terms."""
        entry = self.index.get(term.lower())
        return float(entry["idf"]) if entry else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the full index state."""
        return {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "document_frequency": dict(sorted(self.document_frequency.items())),
            "doc_lengths": dict(sorted(self.doc_lengths.items())),
            "documents": {
                chunk_id: record.to_dict()
                for chunk_id, record in sorted(self.documents.items())
            },
            "index": self.index,
        }

    def _load_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("BM25 index payload must be a mapping")

        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        self.doc_count = int(payload.get("doc_count", 0))
        self.avg_doc_length = float(payload.get("avg_doc_length", 0.0))
        self.document_frequency = {
            str(term): int(count)
            for term, count in payload.get("document_frequency", {}).items()
        }
        self.doc_lengths = {
            str(chunk_id): int(length)
            for chunk_id, length in payload.get("doc_lengths", {}).items()
        }
        self.documents = {
            str(chunk_id): ChunkRecord.from_dict(record)
            for chunk_id, record in payload.get("documents", {}).items()
        }
        self.index = {
            str(term): {
                "idf": float(entry["idf"]),
                "postings": [
                    {
                        "chunk_id": str(posting["chunk_id"]),
                        "tf": float(posting["tf"]),
                        "doc_length": int(posting["doc_length"]),
                    }
                    for posting in entry.get("postings", [])
                ],
            }
            for term, entry in payload.get("index", {}).items()
        }

    def _recompute_index(self) -> None:
        self.doc_count = len(self.documents)
        total_doc_length = sum(self.doc_lengths.get(chunk_id, 0) for chunk_id in self.documents)
        self.avg_doc_length = (
            total_doc_length / self.doc_count if self.doc_count else 0.0
        )

        document_frequency: dict[str, int] = {}
        postings_by_term: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk_id, record in sorted(self.documents.items()):
            sparse_vector = record.sparse_vector or {}
            doc_length = int(self.doc_lengths.get(chunk_id, sum(sparse_vector.values())))
            self.doc_lengths[chunk_id] = doc_length

            for term, tf in sorted(sparse_vector.items()):
                document_frequency[term] = document_frequency.get(term, 0) + 1
                postings_by_term[term].append(
                    {
                        "chunk_id": chunk_id,
                        "tf": float(tf),
                        "doc_length": doc_length,
                    }
                )

        self.document_frequency = dict(sorted(document_frequency.items()))
        self.index = {
            term: {
                "idf": self._calculate_idf(self.doc_count, self.document_frequency[term]),
                "postings": sorted(
                    postings,
                    key=lambda posting: posting["chunk_id"],
                ),
            }
            for term, postings in sorted(postings_by_term.items())
        }

    def _extract_doc_lengths(
        self,
        sparse_result: SparseEncodingResult,
    ) -> dict[str, int]:
        stats_lengths = sparse_result.corpus_stats.get("doc_lengths", {})
        lengths: dict[str, int] = {}
        for record in sparse_result.records:
            if record.id in stats_lengths:
                lengths[record.id] = int(stats_lengths[record.id])
            else:
                lengths[record.id] = int(sum((record.sparse_vector or {}).values()))
        return lengths

    @staticmethod
    def _calculate_idf(doc_count: int, document_frequency: int) -> float:
        if doc_count <= 0 or document_frequency <= 0:
            return 0.0
        return math.log((doc_count - document_frequency + 0.5) / (document_frequency + 0.5))

    @staticmethod
    def _validate_sparse_result(sparse_result: SparseEncodingResult) -> None:
        if not isinstance(sparse_result, SparseEncodingResult):
            raise ValueError("sparse_result must be a SparseEncodingResult")


__all__ = ["BM25Indexer", "DEFAULT_INDEX_DIR", "INDEX_FILENAME"]
