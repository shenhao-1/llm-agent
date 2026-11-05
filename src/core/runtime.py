"""Construct the configured storage and embedding components for all entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, RRFFusion, SparseRetriever
from ingestion.embedding import BatchProcessor
from ingestion.pipeline import IngestionPipeline
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.loader import SQLiteIntegrityChecker
from libs.vector_store import VectorStoreFactory
from libs.embedding import EmbeddingFactory


def runtime_settings(settings: Any, data_dir: str | Path = "data") -> dict[str, Any]:
    """Use the requested data root for the configured Chroma database."""
    payload = settings.to_dict() if hasattr(settings, "to_dict") else dict(settings)
    payload = {key: dict(value) if isinstance(value, dict) else value for key, value in payload.items()}
    vector_config = dict(payload["vector_store"])
    if vector_config.get("provider") == "chroma" and Path(data_dir) != Path("data"):
        vector_config["persist_directory"] = str(Path(data_dir) / "db" / "chroma")
    payload["vector_store"] = vector_config
    return payload


def build_pipeline(settings: Any, data_dir: str | Path = "data") -> IngestionPipeline:
    data_root = Path(data_dir)
    db_dir = data_root / "db"
    active = runtime_settings(settings, data_root)
    bm25 = BM25Indexer(db_dir / "bm25")
    if bm25.index_path.exists():
        bm25.load()
    return IngestionPipeline(
        active,
        integrity_checker=SQLiteIntegrityChecker(db_dir / "ingestion_history.db"),
        batch_processor=BatchProcessor(active),
        bm25_indexer=bm25,
        vector_upserter=VectorUpserter(vector_store=VectorStoreFactory.create(active)),
        image_storage=ImageStorage(
            image_root=data_root / "images", db_path=db_dir / "image_index.db"
        ),
    )


def build_search(settings: Any, data_dir: str | Path = "data") -> HybridSearch:
    data_root = Path(data_dir)
    active = runtime_settings(settings, data_root)
    strategy = active.get("retrieval", {}).get("strategy", "hybrid")
    retrieval_config = active.get("retrieval", {})
    rrf_config = retrieval_config.get("rrf", {})
    bm25 = BM25Indexer(data_root / "db" / "bm25").load() if strategy != "dense" else None
    vector_store = VectorStoreFactory.create(active)
    return HybridSearch(
        settings=active,
        query_processor=QueryProcessor(retrieval_config.get("query_processor", {})),
        dense_retriever=DenseRetriever(
            embedding_client=EmbeddingFactory.create(active), vector_store=vector_store
        ) if strategy != "sparse" else None,
        sparse_retriever=SparseRetriever(bm25_indexer=bm25, vector_store=vector_store)
        if bm25 is not None else None,
        fusion=RRFFusion(
            k=rrf_config.get("k", 60),
            dense_weight=rrf_config.get("dense_weight", 1.0),
            sparse_weight=rrf_config.get("sparse_weight", 1.0),
        ),
    )
