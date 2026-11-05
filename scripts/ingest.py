"""Offline data ingestion CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.settings import SettingsValidationError, load_settings  # noqa: E402
from core.runtime import build_pipeline as build_configured_pipeline  # noqa: E402
from ingestion import IngestionPipeline, IngestionPipelineError  # noqa: E402
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder  # noqa: E402
from ingestion.storage import (  # noqa: E402
    BM25Indexer,
    ImageStorage,
    VectorUpserter,
)
from libs.embedding.base_embedding import BaseEmbedding  # noqa: E402
from libs.loader import SQLiteIntegrityChecker  # noqa: E402
from libs.vector_store import BaseVectorStore  # noqa: E402


class LocalDeterministicEmbedding(BaseEmbedding):
    """Small offline embedding provider used by the CLI."""

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        vectors: list[list[float]] = []
        for text in texts:
            length = float(len(text))
            checksum = float(sum(ord(char) for char in text) % 997)
            token_count = float(len(text.split()))
            vectors.append([length, checksum, token_count])
        return vectors


class JsonVectorStore(BaseVectorStore):
    """Idempotent JSON-backed vector store for offline ingestion."""

    def __init__(self, path: str | Path) -> None:
        super().__init__({})
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, records: list[Mapping[str, Any]], trace=None) -> None:
        self.validate_records(records)
        existing = self._load_records()
        for record in records:
            existing[str(record["id"])] = {
                "id": str(record["id"]),
                "text": str(record.get("text", "")),
                "vector": [float(value) for value in record["vector"]],
                "metadata": dict(record.get("metadata", {}))
                if isinstance(record.get("metadata"), Mapping)
                else {},
            }
        self.path.write_text(
            json.dumps(
                [existing[key] for key in sorted(existing)],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace=None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return list(self._load_records().values())[:top_k]

    def get_by_ids(self, ids: list[str], trace=None) -> list[dict[str, Any]]:
        self.validate_ids(ids)
        records = self._load_records()
        return [records[item_id] for item_id in ids if item_id in records]

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Invalid vector store JSON: {self.path}")
        return {
            str(record["id"]): dict(record)
            for record in payload
            if isinstance(record, Mapping) and isinstance(record.get("id"), str)
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF documents offline.")
    parser.add_argument("--path", required=True, help="PDF file or directory to ingest.")
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name used for vector/image storage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even when file integrity history says the file is unchanged.",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to YAML settings used for splitter and transform configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory for ingestion artifacts.",
    )
    parser.add_argument("--offline", action="store_true", help="Use the legacy JSON demo index and deterministic vectors.")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.settings)
        sources = _discover_sources(Path(args.path))
        pipeline = _build_pipeline(settings, Path(args.data_dir), offline=args.offline)
    except (OSError, SettingsValidationError, ValueError) as exc:
        print(f"ingest setup failed: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for source in sources:
        try:
            result = pipeline.run(
                source,
                collection=args.collection,
                force=args.force,
                on_progress=_print_progress,
            )
        except IngestionPipelineError as exc:
            failures += 1
            print(f"failed {source}: {exc}", file=sys.stderr)
            continue

        status = "skipped" if result.skipped else "ingested"
        print(
            f"{status} {source} "
            f"chunks={result.chunk_count} "
            f"vectors={len(result.vector_ids)} "
            f"images={len(result.image_records)}"
        )

    if failures:
        print(f"completed with {failures} failure(s)", file=sys.stderr)
        return 1
    return 0


def _discover_sources(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"unsupported input file extension: {path.suffix}")
        return [path]
    if path.is_dir():
        sources = sorted(item for item in path.rglob("*.pdf") if item.is_file())
        if sources:
            return sources
        raise ValueError(f"no PDF files found under: {path}")
    raise ValueError(f"path does not exist: {path}")


def _build_pipeline(settings: Any, data_dir: Path, *, offline: bool = False) -> IngestionPipeline:
    if not offline:
        return build_configured_pipeline(settings, data_dir)
    db_dir = data_dir / "db"
    return IngestionPipeline(
        settings,
        integrity_checker=SQLiteIntegrityChecker(db_dir / "ingestion_history.db"),
        batch_processor=BatchProcessor(
            settings,
            dense_encoder=DenseEncoder(embedding=LocalDeterministicEmbedding()),
            sparse_encoder=SparseEncoder(),
        ),
        bm25_indexer=BM25Indexer(db_dir / "bm25"),
        vector_upserter=VectorUpserter(
            vector_store=JsonVectorStore(db_dir / "vectors.json")
        ),
        image_storage=ImageStorage(
            image_root=data_dir / "images",
            db_path=db_dir / "image_index.db",
        ),
    )


def _print_progress(stage: str, current: int, total: int) -> None:
    print(f"[{current}/{total}] {stage}")


if __name__ == "__main__":
    raise SystemExit(main())
