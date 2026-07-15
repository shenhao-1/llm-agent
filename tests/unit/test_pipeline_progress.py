"""Unit tests for ingestion pipeline progress callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.types import Chunk, ChunkRecord, Document
from ingestion.embedding import (
    BatchProcessingResult,
    BatchResult,
    SparseEncodingResult,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.storage import ImageRecord, VectorUpsertResult


class FakeIntegrityChecker:
    def __init__(self, *, skip: bool = False) -> None:
        self.skip = skip
        self.successes: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def compute_sha256(self, path: str | Path) -> str:
        return "hash-progress"

    def should_skip(self, file_hash: str) -> bool:
        return self.skip

    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.successes.append(
            {
                "file_hash": file_hash,
                "file_path": str(file_path),
                "metadata": dict(metadata or {}),
            }
        )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str | Path | None = None,
    ) -> None:
        self.failures.append(
            {
                "file_hash": file_hash,
                "error_msg": error_msg,
                "file_path": str(file_path) if file_path is not None else None,
            }
        )


class FakeLoader:
    def load(self, path: str | Path, trace=None) -> Document:
        return Document(
            id="doc-progress",
            text="Progress callback document.",
            metadata={
                "source_path": str(path),
                "doc_hash": "hash-progress",
                "images": [],
            },
        )


class FakeChunker:
    def split_document(self, document: Document, trace=None) -> list[Chunk]:
        return [
            Chunk(
                id="chunk-progress",
                text=document.text,
                metadata={"source_path": document.metadata["source_path"], "chunk_index": 0},
                start_offset=0,
                end_offset=len(document.text),
                source_ref=document.id,
            )
        ]


class FakeTransform:
    def transform(self, chunks: list[Chunk], trace=None) -> list[Chunk]:
        return list(chunks)


class FakeBatchProcessor:
    def process(
        self,
        chunks: list[Chunk],
        trace=None,
    ) -> BatchProcessingResult:
        dense_records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[1.0, 2.0],
            )
            for chunk in chunks
        ]
        sparse_result = SparseEncodingResult(
            records=[
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    sparse_vector={"progress": 1.0},
                )
                for chunk in chunks
            ],
            corpus_stats={
                "doc_count": len(chunks),
                "avg_doc_length": 1.0,
                "document_frequency": {"progress": len(chunks)},
                "doc_lengths": {chunk.id: 1 for chunk in chunks},
            },
        )
        return BatchProcessingResult(
            batches=[
                BatchResult(
                    batch_index=0,
                    chunks=list(chunks),
                    dense_records=dense_records,
                    sparse_result=sparse_result,
                    elapsed_ms=0.0,
                )
            ],
            dense_records=dense_records,
            sparse_result=sparse_result,
            elapsed_ms=0.0,
        )


class FakeBM25Indexer:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path

    def build(self, sparse_result: SparseEncodingResult):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("{}", encoding="utf-8")
        return self


class FakeVectorUpserter:
    def upsert(
        self,
        records: list[ChunkRecord],
        trace=None,
    ) -> VectorUpsertResult:
        return VectorUpsertResult(
            ids=[record.id for record in records],
            records=[
                {
                    "id": record.id,
                    "text": record.text,
                    "vector": list(record.dense_vector or []),
                    "metadata": dict(record.metadata),
                }
                for record in records
            ],
        )


class FakeImageStorage:
    image_root = Path("unused-images")

    def save_image(self, **kwargs) -> ImageRecord:
        return ImageRecord(
            image_id=str(kwargs["image_id"]),
            file_path=str(kwargs["source_path"]),
            collection=str(kwargs.get("collection", "default")),
        )


def test_run_progress_callback_receives_each_stage_in_order(tmp_path: Path) -> None:
    progress: list[tuple[str, int, int]] = []
    pipeline = _pipeline(tmp_path)

    result = pipeline.run(
        tmp_path / "source.pdf",
        collection="docs",
        on_progress=lambda stage, current, total: progress.append(
            (stage, current, total)
        ),
    )

    assert result.skipped is False
    assert progress == [
        ("integrity", 1, 8),
        ("load", 2, 8),
        ("split", 3, 8),
        ("transform", 4, 8),
        ("encode", 5, 8),
        ("store_bm25", 6, 8),
        ("store_vectors", 7, 8),
        ("store_images", 8, 8),
    ]


def test_constructor_progress_callback_is_used_when_run_callback_is_absent(
    tmp_path: Path,
) -> None:
    progress: list[tuple[str, int, int]] = []
    pipeline = _pipeline(
        tmp_path,
        on_progress=lambda stage, current, total: progress.append(
            (stage, current, total)
        ),
    )

    pipeline.run(tmp_path / "source.pdf", collection="docs")

    assert progress[0] == ("integrity", 1, 8)
    assert progress[-1] == ("store_images", 8, 8)


def test_run_progress_callback_overrides_constructor_callback(tmp_path: Path) -> None:
    constructor_progress: list[tuple[str, int, int]] = []
    run_progress: list[tuple[str, int, int]] = []
    pipeline = _pipeline(
        tmp_path,
        on_progress=lambda stage, current, total: constructor_progress.append(
            (stage, current, total)
        ),
    )

    pipeline.run(
        tmp_path / "source.pdf",
        collection="docs",
        on_progress=lambda stage, current, total: run_progress.append(
            (stage, current, total)
        ),
    )

    assert constructor_progress == []
    assert run_progress[-1] == ("store_images", 8, 8)


def test_none_progress_callback_does_not_change_pipeline_result(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, on_progress=None)

    result = pipeline.run(tmp_path / "source.pdf", collection="docs")

    assert result.collection == "docs"
    assert result.chunk_count == 1
    assert result.vector_ids == ["chunk-progress"]
    assert result.image_records == []


def test_skipped_file_reports_integrity_progress_only(tmp_path: Path) -> None:
    progress: list[tuple[str, int, int]] = []
    pipeline = _pipeline(tmp_path, integrity_checker=FakeIntegrityChecker(skip=True))

    result = pipeline.run(
        tmp_path / "source.pdf",
        collection="docs",
        on_progress=lambda stage, current, total: progress.append(
            (stage, current, total)
        ),
    )

    assert result.skipped is True
    assert progress == [("integrity", 1, 8)]


def _pipeline(
    tmp_path: Path,
    *,
    integrity_checker: FakeIntegrityChecker | None = None,
    on_progress=None,
) -> IngestionPipeline:
    return IngestionPipeline(
        {"retrieval": {"splitter": {"provider": "recursive"}}},
        integrity_checker=integrity_checker or FakeIntegrityChecker(),
        loader=FakeLoader(),
        chunker=FakeChunker(),  # type: ignore[arg-type]
        transforms=[FakeTransform()],  # type: ignore[list-item]
        batch_processor=FakeBatchProcessor(),  # type: ignore[arg-type]
        bm25_indexer=FakeBM25Indexer(tmp_path / "db" / "bm25" / "index.json"),  # type: ignore[arg-type]
        vector_upserter=FakeVectorUpserter(),  # type: ignore[arg-type]
        image_storage=FakeImageStorage(),  # type: ignore[arg-type]
        on_progress=on_progress,
    )
