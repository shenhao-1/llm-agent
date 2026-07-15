"""Online retrieval CLI for the local hybrid search pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.query_engine import (  # noqa: E402
    DenseRetriever,
    HybridSearch,
    QueryProcessor,
    Reranker,
    RRFFusion,
    SparseRetriever,
)
from core.settings import SettingsValidationError, load_settings  # noqa: E402
from core.trace import TraceContext  # noqa: E402
from ingestion.storage import BM25Indexer  # noqa: E402
from libs.embedding import BaseEmbedding  # noqa: E402
from libs.vector_store import BaseVectorStore  # noqa: E402


DEFAULT_DATA_DIR = Path("data")
DEFAULT_SETTINGS = Path("config/settings.yaml")


class QuerySetupError(RuntimeError):
    """Raised when local retrieval artifacts are missing or unreadable."""


class LocalDeterministicEmbedding(BaseEmbedding):
    """Offline embedding provider compatible with scripts/ingest.py output."""

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        vectors: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(char) for char in text) % 997)
            token_count = float(len(text.split()))
            vectors.append([length, checksum, token_count])
        return vectors


class ReadOnlyJsonVectorStore(BaseVectorStore):
    """Read and query the JSON vector store produced by scripts/ingest.py."""

    def __init__(self, path: str | Path) -> None:
        super().__init__({})
        self.path = Path(path)

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace=None,
    ) -> None:
        raise NotImplementedError("query CLI vector store is read-only")

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        records = self._load_records()
        filtered = [
            record
            for record in records
            if _metadata_matches(record.get("metadata", {}), filters)
        ]
        ranked = sorted(
            filtered,
            key=lambda record: (
                -_similarity(vector, record.get("vector")),
                str(record.get("id", "")),
            ),
        )
        return [
            {
                "id": str(record["id"]),
                "text": str(record.get("text", "")),
                "metadata": dict(record.get("metadata", {})),
                "score": _similarity(vector, record.get("vector")),
            }
            for record in ranked[:top_k]
        ]

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records_by_id = {str(record["id"]): record for record in self._load_records()}
        return [
            {
                "id": item_id,
                "text": str(records_by_id[item_id].get("text", "")),
                "metadata": dict(records_by_id[item_id].get("metadata", {})),
                "vector": list(records_by_id[item_id].get("vector", [])),
            }
            for item_id in ids
            if item_id in records_by_id
        ]

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise QuerySetupError(
                f"vector index not found: {self.path}. Run scripts/ingest.py first."
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise QuerySetupError(f"invalid vector index format: {self.path}")

        records: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            record_id = item.get("id")
            vector = item.get("vector")
            if not isinstance(record_id, str) or not isinstance(vector, list):
                continue
            records.append(dict(item))
        return records


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        trace = TraceContext(trace_type="query") if args.verbose else None
        filters = _build_filters(args)
        search = _build_search(settings, Path(args.data_dir))
        results = search.search(
            args.query,
            top_k=args.top_k,
            filters=filters,
            trace=trace,
        )
        if not args.no_rerank:
            results = Reranker(settings=settings).rerank(
                args.query,
                results,
                trace=trace,
            )
    except QuerySetupError as exc:
        print(str(exc))
        print("No indexed data found. Please run scripts/ingest.py before querying.")
        return 1
    except (SettingsValidationError, OSError, ValueError, RuntimeError) as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 2

    if trace is not None:
        trace.finish()
    _print_results(results, top_k=args.top_k)
    if args.verbose and trace is not None:
        _print_trace(trace)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the local RAG index.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return. Defaults to 10.",
    )
    parser.add_argument(
        "--collection",
        help="Optional collection metadata filter.",
    )
    parser.add_argument(
        "--doc-type",
        help="Optional document type metadata filter, for example pdf.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print trace stages for query debugging.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranker stage.",
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS),
        help="Path to YAML settings.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root directory containing db/vectors.json and db/bm25/index.json.",
    )
    return parser


def _build_search(settings: Any, data_dir: Path) -> HybridSearch:
    db_dir = data_dir / "db"
    vector_store = ReadOnlyJsonVectorStore(db_dir / "vectors.json")
    bm25_indexer = _load_bm25_indexer(db_dir / "bm25")
    return HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=DenseRetriever(
            embedding_client=LocalDeterministicEmbedding(),
            vector_store=vector_store,
        ),
        sparse_retriever=SparseRetriever(
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        ),
        fusion=RRFFusion(),
    )


def _load_bm25_indexer(index_dir: Path) -> BM25Indexer:
    try:
        return BM25Indexer(index_dir).load()
    except FileNotFoundError as exc:
        raise QuerySetupError(
            f"BM25 index not found: {index_dir / 'index.json'}. Run scripts/ingest.py first."
        ) from exc


def _build_filters(args: argparse.Namespace) -> dict[str, str]:
    filters: dict[str, str] = {}
    if args.collection:
        filters["collection"] = args.collection
    if args.doc_type:
        filters["doc_type"] = args.doc_type
    return filters


def _print_results(results: list[Any], *, top_k: int) -> None:
    if not results:
        print("No relevant documents found. Please run scripts/ingest.py first if the index is empty.")
        return

    print(f"Top {min(top_k, len(results))} results")
    for index, result in enumerate(results, start=1):
        metadata = result.metadata
        source = metadata.get("source_path", "unknown")
        page = metadata.get("page", metadata.get("page_num", metadata.get("page_count", "-")))
        chunk_index = metadata.get("chunk_index", "-")
        print(f"\n[{index}] score={result.score:.4f} id={result.chunk_id}")
        print(f"source={source} page={page} chunk={chunk_index}")
        print(_summarize(result.text))


def _print_trace(trace: TraceContext) -> None:
    print("\nTrace stages")
    for index, stage in enumerate(trace.stages, start=1):
        method = stage.get("method", "-")
        details = stage.get("details", {})
        print(f"{index}. {stage['name']} method={method} details={details}")


def _summarize(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _metadata_matches(
    metadata: Any,
    filters: Mapping[str, Any] | None,
) -> bool:
    if not filters:
        return True
    if not isinstance(metadata, Mapping):
        return False
    for key, expected in filters.items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False
    return True


def _similarity(query_vector: list[float], candidate_vector: Any) -> float:
    if not isinstance(candidate_vector, list) or not candidate_vector:
        return 0.0
    values = [
        float(value)
        for value in candidate_vector
        if isinstance(value, (int, float))
    ]
    if not values or len(values) != len(query_vector):
        return 0.0
    distance = math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(query_vector, values))
    )
    return 1.0 / (1.0 + distance)


if __name__ == "__main__":
    raise SystemExit(main())
