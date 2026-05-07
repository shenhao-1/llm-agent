"""Integration tests for the MVP ingestion pipeline."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.trace import TraceContext
from ingestion import IngestionPipeline, IngestionPipelineError, IngestionResult
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.embedding.base_embedding import BaseEmbedding
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store import BaseVectorStore


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding provider for pipeline integration tests."""

    def __init__(self) -> None:
        super().__init__({})
        self.calls: list[list[str]] = []

    def embed(
        self,
        texts: list[str],
        trace: Any | None = None,
    ) -> list[list[float]]:
        self.validate_texts(texts)
        self.calls.append(texts)
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class FakeVectorStore(BaseVectorStore):
    """In-memory idempotent vector store."""

    def __init__(self) -> None:
        super().__init__({})
        self.records_by_id: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        records: list[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        self.validate_records(records)
        for record in records:
            self.records_by_id[str(record["id"])] = dict(record)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        return list(self.records_by_id.values())[:top_k]


class FailingLoader:
    def load(self, path: str | Path, trace: Any | None = None):
        raise RuntimeError("loader exploded")


@pytest.mark.integration
def test_pipeline_ingests_simple_pdf_to_bm25_and_vector_store(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    _write_text_pdf(pdf_path, "Simple retrieval document about BM25 and dense vectors.")
    trace = TraceContext()
    progress: list[tuple[str, int, int]] = []
    pipeline, vector_store = _pipeline(
        tmp_path,
        lambda stage, current, total: progress.append((stage, current, total)),
    )

    result = pipeline.run(pdf_path, collection="docs", trace=trace)

    assert isinstance(result, IngestionResult)
    assert result.skipped is False
    assert result.document is not None
    assert result.chunk_count >= 1
    assert result.bm25_index_path is not None
    assert result.bm25_index_path.exists()
    assert result.vector_ids
    assert set(result.vector_ids) == set(vector_store.records_by_id)
    assert result.image_records == []
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
    assert {"integrity", "load", "split", "transform", "encode"}.issubset(
        {stage["name"] for stage in trace.stages}
    )


@pytest.mark.integration
def test_pipeline_ingests_pdf_with_image_to_image_storage(tmp_path: Path) -> None:
    pdf_path = tmp_path / "complex_technical_doc.pdf"
    image_source = tmp_path / "pixel.png"
    image_source.write_bytes(PNG_1X1)
    _write_pdf_with_image(
        pdf_path,
        image_source,
        text="Architecture section with an embedded image for multimodal retrieval.",
    )
    pipeline, _ = _pipeline(tmp_path)

    result = pipeline.run(pdf_path, collection="technical")

    assert result.document is not None
    assert len(result.document.metadata["images"]) == 1
    assert len(result.image_records) == 1
    image_record = result.image_records[0]
    assert Path(image_record.file_path).exists()
    assert image_record.collection == "technical"
    assert image_record.doc_hash == result.document.metadata["doc_hash"]
    assert pipeline.image_storage.get_record(image_record.image_id) == image_record
    assert pipeline.image_storage.list_by_collection("technical") == [image_record]


@pytest.mark.integration
def test_pipeline_skips_successfully_ingested_file_on_second_run(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    _write_text_pdf(pdf_path, "Repeat ingestion should be skipped.")
    pipeline, _ = _pipeline(tmp_path)

    first = pipeline.run(pdf_path, collection="docs")
    second = pipeline.run(pdf_path, collection="docs")

    assert first.skipped is False
    assert second.skipped is True
    assert second.file_hash == first.file_hash
    assert second.chunks == []


@pytest.mark.integration
def test_pipeline_reports_failed_stage_and_marks_integrity_failed(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    checker = SQLiteIntegrityChecker(tmp_path / "db" / "history.db")
    pipeline, _ = _pipeline(tmp_path, integrity_checker=checker, loader=FailingLoader())

    with pytest.raises(IngestionPipelineError) as exc_info:
        pipeline.run(pdf_path, collection="docs")

    assert exc_info.value.stage == "load"
    file_hash = checker.compute_sha256(pdf_path)
    record = checker.get_record(file_hash)
    assert record is not None
    assert record["status"] == "failed"
    assert "load" in record["error_msg"]


def _pipeline(
    tmp_path: Path,
    on_progress=None,
    *,
    integrity_checker: SQLiteIntegrityChecker | None = None,
    loader=None,
) -> tuple[IngestionPipeline, FakeVectorStore]:
    embedding = FakeEmbedding()
    vector_store = FakeVectorStore()
    image_storage = ImageStorage(
        image_root=tmp_path / "images",
        db_path=tmp_path / "db" / "image_index.db",
    )
    pipeline = IngestionPipeline(
        {
            "retrieval": {
                "splitter": {
                    "provider": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 0,
                }
            },
            "ingestion": {
                "chunk_refiner": {"use_llm": False},
                "metadata_enricher": {"use_llm": False},
                "image_captioner": {"enabled": False},
                "batch_processor": {"batch_size": 2},
            },
        },
        integrity_checker=integrity_checker
        or SQLiteIntegrityChecker(tmp_path / "db" / "history.db"),
        loader=loader,
        batch_processor=BatchProcessor(
            dense_encoder=DenseEncoder(embedding=embedding),
            sparse_encoder=SparseEncoder(),
            batch_size=2,
        ),
        bm25_indexer=BM25Indexer(tmp_path / "db" / "bm25"),
        vector_upserter=VectorUpserter(vector_store=vector_store),
        image_storage=image_storage,
        on_progress=on_progress,
    )
    return pipeline, vector_store


def _write_text_pdf(path: Path, text: str) -> None:
    import fitz  # type: ignore[import-not-found]

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def _write_pdf_with_image(path: Path, image_path: Path, text: str) -> None:
    import fitz  # type: ignore[import-not-found]

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    page.insert_image(fitz.Rect(72, 96, 112, 136), filename=str(image_path))
    pdf.save(path)
    pdf.close()
