"""Optional real-LLM smoke test for ChunkRefiner.

Run with RUN_REAL_LLM_TESTS=1 when provider credentials are configured.
"""

from __future__ import annotations

import os

import pytest

from core import Chunk
from core.settings import load_settings
from ingestion.transform import ChunkRefiner


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_TESTS") != "1",
    reason="Set RUN_REAL_LLM_TESTS=1 to run real provider calls.",
)
def test_chunk_refiner_real_llm_call_preserves_content() -> None:
    settings = load_settings("config/settings.yaml")
    chunk = Chunk(
        id="chunk-1",
        text="Confidential\n\nPage 1\n\nHybrid search combines BM25 and embeddings.",
        metadata={"source_path": "docs/sample.md"},
    )

    refined = ChunkRefiner(
        {
            "llm": settings.llm,
            "ingestion": {"chunk_refiner": {"use_llm": True}},
        }
    ).transform([chunk])[0]

    assert refined.metadata["refined_by"] in {"llm", "rule"}
    assert "Hybrid search" in refined.text
