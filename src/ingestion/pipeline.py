"""MVP ingestion pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING

from core.types import Chunk, ChunkRecord, Document
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessingResult, BatchProcessor, SparseEncodingResult
from ingestion.storage import BM25Indexer, ImageRecord, ImageStorage, VectorUpserter
from ingestion.transform import ChunkRefiner, ImageCaptioner, MetadataEnricher
from libs.loader import PdfLoader, SQLiteIntegrityChecker

if TYPE_CHECKING:
    from core.settings import Settings
    from core.trace.trace_context import TraceContext
    from libs.loader.file_integrity import FileIntegrityChecker


ProgressCallback = Callable[[str, int, int], None]


class LoaderProtocol(Protocol):
    def load(self, path: str | Path, trace: "TraceContext | None" = None) -> Document:
        ...


class TransformProtocol(Protocol):
    def transform(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None" = None,
    ) -> list[Chunk]:
        ...


@dataclass(frozen=True)
class IngestionResult:
    """Artifacts produced by one pipeline run."""

    file_hash: str
    source_path: str
    collection: str
    skipped: bool = False
    document: Document | None = None
    chunks: list[Chunk] = field(default_factory=list)
    dense_records: list[ChunkRecord] = field(default_factory=list)
    batch_result: BatchProcessingResult | None = None
    bm25_index_path: Path | None = None
    vector_ids: list[str] = field(default_factory=list)
    image_records: list[ImageRecord] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class IngestionPipelineError(RuntimeError):
    """Pipeline failure that names the stage that failed."""

    def __init__(self, stage: str, source_path: str | Path, cause: Exception) -> None:
        self.stage = stage
        self.source_path = str(source_path)
        self.cause = cause
        super().__init__(f"Ingestion pipeline failed at {stage}: {cause}")


class IngestionPipeline:
    """Run integrity, loading, chunking, transforms, encoding, and storage."""

    _PROGRESS_STAGES = (
        "integrity",
        "load",
        "split",
        "transform",
        "encode",
        "store_bm25",
        "store_vectors",
        "store_images",
    )

    def __init__(
        self,
        settings: "Settings | Mapping[str, Any] | None" = None,
        *,
        integrity_checker: "FileIntegrityChecker | None" = None,
        loader: LoaderProtocol | None = None,
        chunker: DocumentChunker | None = None,
        transforms: list[TransformProtocol] | None = None,
        batch_processor: BatchProcessor | None = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_upserter: VectorUpserter | None = None,
        image_storage: ImageStorage | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = _settings_with_defaults(settings or {})
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.loader = loader
        self.chunker = chunker or DocumentChunker(self.settings)
        self.transforms = transforms or [
            ChunkRefiner(self.settings),
            MetadataEnricher(self.settings),
            ImageCaptioner(self.settings),
        ]
        self.batch_processor = batch_processor or BatchProcessor(self.settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(self.settings)
        self.image_storage = image_storage or ImageStorage()
        self.on_progress = on_progress

    def run(
        self,
        source_path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        trace: "TraceContext | None" = None,
        on_progress: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Run the complete ingestion pipeline for one source file."""
        path = Path(source_path)
        progress = on_progress or self.on_progress
        file_hash = ""
        current_stage = "integrity"

        try:
            current_stage = "integrity"
            file_hash = self.integrity_checker.compute_sha256(path)
            store = self.vector_upserter.vector_store
            backend_id = f"{type(store).__module__}.{type(store).__qualname__}"
            previous = getattr(self.integrity_checker, "get_record", None)
            record = previous(file_hash) if callable(previous) else None
            same_target = (
                record is None
                or (
                    record.get("file_path") == str(path)
                    and isinstance(record.get("metadata"), Mapping)
                    and record["metadata"].get("collection") == collection
                    and record["metadata"].get("vector_backend") == backend_id
                )
            )
            if not force and same_target and self.integrity_checker.should_skip(file_hash):
                result = IngestionResult(
                    file_hash=file_hash,
                    source_path=str(path),
                    collection=collection,
                    skipped=True,
                )
                self._record_stage(
                    trace,
                    "integrity",
                    {"skipped": True},
                    method=_component_name(self.integrity_checker),
                )
                self._progress(progress, "integrity")
                return result
            self._record_stage(
                trace,
                "integrity",
                {"skipped": False},
                method=_component_name(self.integrity_checker),
            )
            self._progress(progress, "integrity")

            current_stage = "load"
            document = self._run_load(path, collection, trace)
            self._progress(progress, "load")

            current_stage = "split"
            chunks = self.chunker.split_document(document, trace=trace)
            self._record_stage(
                trace,
                "split",
                {"chunk_count": len(chunks)},
                method=_component_name(getattr(self.chunker, "splitter", self.chunker)),
            )
            self._progress(progress, "split")

            current_stage = "transform"
            transformed_chunks = self._run_transforms(chunks, trace)
            self._record_stage(
                trace,
                "transform",
                {"chunk_count": len(transformed_chunks)},
                method="ingestion_pipeline",
            )
            self._progress(progress, "transform")

            current_stage = "encode"
            batch_result = self.batch_processor.process(transformed_chunks, trace=trace)
            self._record_stage(
                trace,
                "embed",
                {
                    "chunk_count": len(transformed_chunks),
                    "dense_count": len(batch_result.dense_records),
                    "batch_count": batch_result.batch_count,
                },
                method=_component_name(self.batch_processor),
            )
            self._progress(progress, "encode")

            current_stage = "store_bm25"
            sparse_result = self._sparse_with_vector_ids(batch_result)
            # Preserve chunks from earlier files in the same index. Replacing a
            # source first also handles --force and changed file contents.
            if self.bm25_indexer.index_path.exists() and not self.bm25_indexer.documents:
                self.bm25_indexer.load()
            self.bm25_indexer.remove_document(str(path), collection, persist=False)
            self.bm25_indexer.update(sparse_result)
            bm25_path = self.bm25_indexer.index_path
            self._record_stage(
                trace,
                "store_bm25",
                {"index_path": str(bm25_path)},
                method=_component_name(self.bm25_indexer),
            )
            self._progress(progress, "store_bm25")

            current_stage = "store_vectors"
            # Remove superseded chunk IDs when the same source is re-ingested.
            delete_by_metadata = getattr(
                self.vector_upserter.vector_store, "delete_by_metadata", None
            )
            if callable(delete_by_metadata):
                delete_by_metadata({"source_path": str(path), "collection": collection})
            vector_result = self.vector_upserter.upsert(
                batch_result.dense_records, trace=trace
            )
            self._record_stage(
                trace,
                "upsert",
                {"record_count": vector_result.count},
                method=_component_name(self.vector_upserter),
            )
            self._progress(progress, "store_vectors")

            current_stage = "store_images"
            image_records = self._store_images(document, collection, trace)
            self._record_stage(
                trace,
                "store_images",
                {"image_count": len(image_records)},
                method=_component_name(self.image_storage),
            )
            self._progress(progress, "store_images")

            result = IngestionResult(
                file_hash=file_hash,
                source_path=str(path),
                collection=collection,
                document=document,
                chunks=transformed_chunks,
                dense_records=batch_result.dense_records,
                batch_result=batch_result,
                bm25_index_path=bm25_path,
                vector_ids=vector_result.ids,
                image_records=image_records,
            )
            self.integrity_checker.mark_success(
                file_hash,
                path,
                metadata={
                    "collection": collection,
                    "chunk_count": result.chunk_count,
                    "image_count": len(image_records),
                    "vector_backend": backend_id,
                },
            )
            return result
        except Exception as exc:
            if isinstance(exc, IngestionPipelineError):
                error = exc
            else:
                error = IngestionPipelineError(current_stage, path, exc)
            if file_hash:
                self.integrity_checker.mark_failed(file_hash, str(error), file_path=path)
            raise error from exc

    def _run_load(
        self,
        path: Path,
        collection: str,
        trace: "TraceContext | None",
    ) -> Document:
        loader = self.loader or PdfLoader(
            {
                "collection": collection,
                "image_output_dir": str(self.image_storage.image_root),
            }
        )
        document = loader.load(path, trace=trace)
        self._record_stage(
            trace,
            "load",
            {
                "document_id": document.id,
                "text_length": len(document.text),
                "image_count": len(document.metadata.get("images", [])),
            },
            method=_component_name(loader),
        )
        return document

    def _sparse_with_vector_ids(
        self, batch_result: BatchProcessingResult
    ) -> SparseEncodingResult:
        """Give both retrieval routes the same IDs for the same chunks."""
        stable_id = getattr(self.vector_upserter, "stable_id", None)
        if not callable(stable_id):
            return batch_result.sparse_result
        vector_ids = {record.id: stable_id(record) for record in batch_result.dense_records}
        records = [
            ChunkRecord(
                id=vector_ids[record.id],
                text=record.text,
                metadata={**record.metadata, "source_chunk_id": record.id},
                sparse_vector=record.sparse_vector,
            )
            for record in batch_result.sparse_result.records
        ]
        stats = dict(batch_result.sparse_result.corpus_stats)
        stats["doc_lengths"] = {
            vector_ids[chunk_id]: length
            for chunk_id, length in stats.get("doc_lengths", {}).items()
        }
        return SparseEncodingResult(records=records, corpus_stats=stats)

    def _run_transforms(
        self,
        chunks: list[Chunk],
        trace: "TraceContext | None",
    ) -> list[Chunk]:
        transformed = chunks
        for transform in self.transforms:
            transformed = transform.transform(transformed, trace=trace)
        return transformed

    def _store_images(
        self,
        document: Document,
        collection: str,
        trace: "TraceContext | None",
    ) -> list[ImageRecord]:
        images = document.metadata.get("images", [])
        if not isinstance(images, list):
            raise ValueError("document.metadata.images must be a list")

        records: list[ImageRecord] = []
        doc_hash = document.metadata.get("doc_hash")
        for image in images:
            if not isinstance(image, Mapping):
                continue
            image_id = image.get("id")
            image_path = image.get("path")
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image metadata id must be a non-empty string")
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValueError(f"image {image_id} path must be a non-empty string")

            records.append(
                self.image_storage.save_image(
                    image_id=image_id,
                    source_path=image_path,
                    collection=collection,
                    doc_hash=doc_hash if isinstance(doc_hash, str) else None,
                    page_num=image.get("page") if isinstance(image.get("page"), int) else None,
                    trace=trace,
                )
            )
        return records

    def _progress(
        self,
        callback: ProgressCallback | None,
        stage: str,
    ) -> None:
        if callback is None:
            return
        total = len(self._PROGRESS_STAGES)
        current = self._PROGRESS_STAGES.index(stage) + 1
        callback(stage, current, total)

    @staticmethod
    def _record_stage(
        trace: "TraceContext | None",
        stage: str,
        details: dict[str, Any],
        *,
        method: str = "ingestion_pipeline",
    ) -> None:
        if trace is not None:
            trace.record_stage(stage, method=method, details=details)


def _component_name(component: Any) -> str:
    return component.__class__.__name__

def _settings_with_defaults(settings: "Settings | Mapping[str, Any]") -> dict[str, Any]:
    if hasattr(settings, "llm"):
        merged = {
            "llm": dict(getattr(settings, "llm")),
            "embedding": dict(getattr(settings, "embedding")),
            "vector_store": dict(getattr(settings, "vector_store")),
            "retrieval": dict(getattr(settings, "retrieval")),
            "rerank": dict(getattr(settings, "rerank")),
            "evaluation": dict(getattr(settings, "evaluation")),
            "observability": dict(getattr(settings, "observability")),
            "ingestion": dict(getattr(settings, "ingestion", {})),
            "vision_llm": dict(getattr(settings, "vision_llm", {})),
        }
    else:
        merged = dict(settings)
    retrieval = dict(merged.get("retrieval", {})) if isinstance(merged.get("retrieval"), Mapping) else {}
    retrieval.setdefault(
        "splitter",
        {"provider": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
    )
    merged["retrieval"] = retrieval
    return merged


__all__ = ["IngestionPipeline", "IngestionPipelineError", "IngestionResult"]
