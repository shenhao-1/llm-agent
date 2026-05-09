"""Tests for the local query CLI."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from core.types import Chunk
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer


QUERY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "query.py"
SPEC = importlib.util.spec_from_file_location("query_script", QUERY_SCRIPT)
assert SPEC is not None
query_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(query_script)

LocalDeterministicEmbedding = query_script.LocalDeterministicEmbedding
main = query_script.main


def test_query_cli_outputs_top_results(tmp_path: Path, capsys) -> None:
    _write_query_index(
        tmp_path,
        [
            _chunk(
                "chunk-azure",
                "Azure configuration uses endpoints and API keys.",
                collection="docs",
            ),
            _chunk(
                "chunk-bm25",
                "BM25 sparse retrieval matches exact keywords.",
                collection="docs",
            ),
        ],
    )

    code = main(
        [
            "--query",
            "Azure configuration",
            "--top-k",
            "1",
            "--data-dir",
            str(tmp_path),
            "--no-rerank",
        ]
    )

    output = capsys.readouterr()
    assert code == 0
    assert "Top 1 results" in output.out
    assert "Azure configuration" in output.out
    assert "source=docs/chunk-azure.pdf" in output.out


def test_query_cli_filters_by_collection(tmp_path: Path, capsys) -> None:
    _write_query_index(
        tmp_path,
        [
            _chunk("chunk-docs", "shared query term in docs.", collection="docs"),
            _chunk("chunk-notes", "shared query term in notes.", collection="notes"),
        ],
    )

    code = main(
        [
            "--query",
            "shared query",
            "--collection",
            "notes",
            "--data-dir",
            str(tmp_path),
            "--no-rerank",
        ]
    )

    output = capsys.readouterr()
    assert code == 0
    assert "chunk-notes" in output.out
    assert "chunk-docs" not in output.out


def test_query_cli_verbose_prints_trace_without_rerank(tmp_path: Path, capsys) -> None:
    _write_query_index(
        tmp_path,
        [_chunk("chunk-trace", "traceable hybrid search result.", collection="docs")],
    )

    code = main(
        [
            "--query",
            "traceable hybrid",
            "--data-dir",
            str(tmp_path),
            "--verbose",
            "--no-rerank",
        ]
    )

    output = capsys.readouterr()
    assert code == 0
    assert "Trace stages" in output.out
    assert "hybrid_search" in output.out
    assert "rerank" not in output.out


def test_query_cli_reports_missing_index(tmp_path: Path, capsys) -> None:
    code = main(["--query", "missing data", "--data-dir", str(tmp_path)])

    output = capsys.readouterr()
    assert code == 1
    assert "No indexed data found" in output.out
    assert "scripts/ingest.py" in output.out


def _write_query_index(tmp_path: Path, chunks: list[Chunk]) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)

    BM25Indexer(db_dir / "bm25").build(SparseEncoder().encode(chunks))

    embedding = LocalDeterministicEmbedding()
    vectors = embedding.embed([chunk.text for chunk in chunks])
    payload = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "vector": vectors[index],
            "metadata": dict(chunk.metadata),
        }
        for index, chunk in enumerate(chunks)
    ]
    (db_dir / "vectors.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _chunk(chunk_id: str, text: str, *, collection: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={
            "source_path": f"docs/{chunk_id}.pdf",
            "doc_type": "pdf",
            "collection": collection,
            "chunk_index": 0,
        },
    )
