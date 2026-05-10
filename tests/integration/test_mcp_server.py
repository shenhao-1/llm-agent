"""Integration tests for the MCP stdio server entrypoint."""

from __future__ import annotations

import base64
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
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_stdio_server_initialize_keeps_stdout_json_only() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest-client", "version": "0.1.0"},
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert response["result"]["capabilities"] == {"tools": {}}
    assert "starting modular-rag-mcp-server stdio server" in completed.stderr
    assert "initialize request received" in completed.stderr


def test_stdio_server_returns_method_not_found_without_stdout_log_pollution() -> None:
    request = {"jsonrpc": "2.0", "id": "bad-method", "method": "unknown/method"}

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response == {
        "jsonrpc": "2.0",
        "id": "bad-method",
        "error": {"code": -32601, "message": "Method not found"},
    }
    assert "unsupported JSON-RPC method" in completed.stderr


def test_query_knowledge_hub_tool_returns_markdown_no_results_response() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": "query-tool",
        "method": "tools/call",
        "params": {
            "name": "query_knowledge_hub",
            "arguments": {"query": "missing topic", "top_k": 2},
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "query-tool"
    result = response["result"]
    assert result["content"][0]["type"] == "text"
    assert "No relevant documents found" in result["content"][0]["text"]
    assert result["structuredContent"]["citations"] == []
    assert result["structuredContent"]["result_count"] == 0
    assert "tools/call request received for query_knowledge_hub" in completed.stderr


def test_tools_list_includes_query_knowledge_hub_schema() -> None:
    request = {"jsonrpc": "2.0", "id": "list", "method": "tools/list", "params": {}}

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout.splitlines()[0])
    tools = response["result"]["tools"]
    by_name = {tool["name"]: tool for tool in tools}
    assert "list_collections" in by_name
    assert "query_knowledge_hub" in by_name
    assert "query" in by_name["query_knowledge_hub"]["inputSchema"]["required"]


def test_list_collections_tool_returns_collection_names(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    (docs_dir / "docs").mkdir(parents=True)
    (docs_dir / "notes").mkdir()
    (docs_dir / "docs" / "a.pdf").write_bytes(b"abc")
    request = {
        "jsonrpc": "2.0",
        "id": "collections",
        "method": "tools/call",
        "params": {
            "name": "list_collections",
            "arguments": {"documents_dir": str(docs_dir)},
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout.splitlines()[0])
    result = response["result"]
    assert result["structuredContent"]["collection_names"] == ["docs", "notes"]
    assert "- docs (1 documents)" in result["content"][0]["text"]


def test_get_document_summary_tool_returns_structured_summary(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "vectors.json").write_text(
        json.dumps(
            [
                {
                    "id": "vec-1",
                    "text": "chunk",
                    "vector": [1.0, 2.0, 3.0],
                    "metadata": {
                        "doc_hash": "doc-1",
                        "source_path": "docs/guide.pdf",
                        "collection": "docs",
                        "title": "Guide",
                        "summary": "Useful guide.",
                        "tags": ["guide"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    request = {
        "jsonrpc": "2.0",
        "id": "summary",
        "method": "tools/call",
        "params": {
            "name": "get_document_summary",
            "arguments": {"doc_id": "doc-1", "data_dir": str(tmp_path)},
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout.splitlines()[0])
    result = response["result"]
    assert result["structuredContent"]["doc_id"] == "doc-1"
    assert result["structuredContent"]["title"] == "Guide"
    assert result["structuredContent"]["tags"] == ["guide"]
    assert "## Guide" in result["content"][0]["text"]


def test_get_document_summary_tool_returns_json_rpc_error_for_missing_doc(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "vectors.json").write_text("[]", encoding="utf-8")
    request = {
        "jsonrpc": "2.0",
        "id": "missing-summary",
        "method": "tools/call",
        "params": {
            "name": "get_document_summary",
            "arguments": {"doc_id": "missing", "data_dir": str(tmp_path)},
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout.splitlines()[0])
    assert response["error"] == {
        "code": -32602,
        "message": "Document not found: missing",
    }


def test_query_knowledge_hub_tool_returns_image_content(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(PNG_1X1)
    chunk = Chunk(
        id="chunk-img",
        text="architecture diagram with image evidence",
        metadata={
            "source_path": "docs/architecture.pdf",
            "doc_type": "pdf",
            "collection": "docs",
            "chunk_index": 0,
            "image_refs": ["img-1"],
            "images": [
                {
                    "id": "img-1",
                    "path": str(image_path),
                    "page": 0,
                    "text_offset": 0,
                    "text_length": 0,
                }
            ],
        },
    )
    _write_query_index(tmp_path, [chunk])
    request = {
        "jsonrpc": "2.0",
        "id": "image-query",
        "method": "tools/call",
        "params": {
            "name": "query_knowledge_hub",
            "arguments": {
                "query": "architecture diagram",
                "top_k": 1,
                "data_dir": str(tmp_path),
            },
        },
    }

    completed = subprocess.run(
        [sys.executable, "-m", "mcp_server.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=_subprocess_env(),
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout.splitlines()[0])
    content = response["result"]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["mimeType"] == "image/png"
    assert content[1]["data"] == base64.b64encode(PNG_1X1).decode("ascii")
    assert response["result"]["structuredContent"]["image_count"] == 1


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    return env


def _write_query_index(tmp_path: Path, chunks: list[Chunk]) -> None:
    db_dir = tmp_path / "db"
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
