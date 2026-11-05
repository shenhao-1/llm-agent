"""End-to-end MCP client simulation over the stdio server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from core.types import Chunk
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Indexer
from mcp_server.tools.query_knowledge_hub import LocalDeterministicEmbedding


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_mcp_client_lists_tools_and_calls_query_knowledge_hub(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    data_dir = tmp_path / "data"
    _write_settings(settings_path)
    _write_query_index(
        data_dir,
        [
            Chunk(
                id="chunk-client-e2e",
                text=(
                    "MCP client simulation verifies tools list and query knowledge "
                    "hub citations for architecture retrieval evidence."
                ),
                metadata={
                    "source_path": "docs/mcp_client_guide.pdf",
                    "collection": "docs",
                    "page": 2,
                },
            ),
            Chunk(
                id="chunk-distractor",
                text="Unrelated dashboard settings and ingestion notes.",
                metadata={"source_path": "docs/other.pdf", "collection": "docs"},
            ),
        ],
    )
    requests = [
        {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest-e2e-client", "version": "0.1.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        },
        {"jsonrpc": "2.0", "id": "tools", "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": "query",
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_hub",
                "arguments": {
                    "query": "architecture retrieval evidence",
                    "top_k": 2,
                    "collection": "docs",
                    "settings_path": str(settings_path),
                    "data_dir": str(data_dir),
                    "offline": True,
                },
            },
        },
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input="\n".join(json.dumps(request) for request in requests) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    responses = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    assert [response["id"] for response in responses] == ["init", "tools", "query"]
    assert responses[0]["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"

    tool_names = {
        tool["name"]
        for tool in responses[1]["result"]["tools"]
    }
    assert "query_knowledge_hub" in tool_names

    query_result = responses[2]["result"]
    structured = query_result["structuredContent"]
    assert structured["result_count"] >= 1
    assert structured["query"] == "architecture retrieval evidence"
    assert any(
        citation["chunk_id"] == "chunk-client-e2e"
        and citation["source"] == "docs/mcp_client_guide.pdf"
        for citation in structured["citations"]
    )
    assert "docs/mcp_client_guide.pdf" in query_result["content"][0]["text"]


def _write_settings(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "embedding:",
                "  provider: local",
                "vector_store:",
                "  provider: json",
                "retrieval:",
                "  strategy: hybrid",
                "  top_k: 5",
                "rerank:",
                "  provider: none",
                "evaluation:",
                "  provider: custom",
                "observability:",
                "  level: INFO",
            ]
        ),
        encoding="utf-8",
    )


def _write_query_index(data_dir: Path, chunks: list[Chunk]) -> None:
    db_dir = data_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    BM25Indexer(db_dir / "bm25").build(SparseEncoder().encode(chunks))
    vectors = LocalDeterministicEmbedding().embed([chunk.text for chunk in chunks])
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


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    return env
