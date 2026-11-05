"""Configured storage must be shared by ingestion, retrieval, and deletion."""

from pathlib import Path

import fitz

from core.runtime import build_pipeline, build_search, runtime_settings
from ingestion.document_manager import DocumentManager
from libs.embedding import BaseEmbedding, EmbeddingFactory
from libs.vector_store import VectorStoreFactory


class FakeRuntimeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.validate_texts(texts)
        return [
            [float(text.lower().count("alpha") + 1), float(text.lower().count("beta") + 1), 1.0]
            for text in texts
        ]


def _write_pdf(path: Path, content: str) -> None:
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), content)
        pdf.save(path)


def test_configured_pipeline_query_and_delete_share_chroma(tmp_path: Path) -> None:
    EmbeddingFactory.register_provider("test-runtime", FakeRuntimeEmbedding)
    settings = {
        "embedding": {"provider": "test-runtime"},
        "vector_store": {"provider": "chroma", "persist_directory": "data/db/chroma"},
        "retrieval": {"top_k": 5},
        "rerank": {"provider": "none"},
        "ingestion": {"image_captioner": {"enabled": False}},
    }
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    _write_pdf(first, "alpha architecture document")
    _write_pdf(second, "beta retrieval document")

    pipeline = build_pipeline(settings, tmp_path / "data")
    pipeline.run(first, collection="docs")
    pipeline.run(second, collection="docs")
    search = build_search(settings, tmp_path / "data")
    results = search.search("alpha", top_k=5)
    sparse_results = search.sparse_retriever.retrieve(["alpha"], top_k=5)

    assert len(pipeline.bm25_indexer.documents) == 2
    assert {item.metadata["source_path"] for item in results} == {str(first), str(second)}
    assert len({item.chunk_id for item in results}) == len(results)
    assert sparse_results and sparse_results[0].text
    assert sparse_results[0].chunk_id in {item.chunk_id for item in results}
    sparse_settings = {
        **settings,
        "embedding": {"provider": "unused-in-sparse-only"},
        "retrieval": {"strategy": "sparse", "top_k": 5},
    }
    sparse_only = build_search(sparse_settings, tmp_path / "data")
    assert sparse_only.search("alpha", top_k=5)[0].text
    store = VectorStoreFactory.create(runtime_settings(settings, tmp_path / "data"))
    assert store.get_collection_stats()["record_count"] == 2

    manager = DocumentManager(
        chroma_store=store,
        bm25_indexer=pipeline.bm25_indexer,
        image_storage=pipeline.image_storage,
        file_integrity=pipeline.integrity_checker,
    )
    manager.delete_document(str(first), collection="docs")

    assert store.get_collection_stats()["record_count"] == 1
    assert pipeline.bm25_indexer.doc_count == 1
    assert all(item.metadata["source_path"] != str(first) for item in search.search("beta", top_k=5))


def test_reingest_changed_source_removes_old_vector_and_bm25_ids(tmp_path: Path) -> None:
    EmbeddingFactory.register_provider("test-runtime", FakeRuntimeEmbedding)
    settings = {
        "embedding": {"provider": "test-runtime"},
        "vector_store": {"provider": "chroma", "persist_directory": "data/db/chroma"},
        "retrieval": {"top_k": 5},
        "rerank": {"provider": "none"},
    }
    path = tmp_path / "changed.pdf"
    pipeline = build_pipeline(settings, tmp_path / "data")
    _write_pdf(path, "alpha original topic")
    initial = pipeline.run(path, collection="docs")
    _write_pdf(path, "beta replacement topic")
    updated = pipeline.run(path, collection="docs")

    assert not updated.skipped
    assert initial.vector_ids != updated.vector_ids
    assert pipeline.bm25_indexer.doc_count == 1
    assert set(pipeline.bm25_indexer.documents) == set(updated.vector_ids)
    store = VectorStoreFactory.create(runtime_settings(settings, tmp_path / "data"))
    assert store.get_collection_stats()["record_count"] == 1
    assert store.get_by_ids(initial.vector_ids) == []
