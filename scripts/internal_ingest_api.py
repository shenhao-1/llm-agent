"""Private HTTP bridge from the Java Kafka worker to the existing Python pipeline."""

from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class IngestService:
    def __init__(self, *, data_dir: Path, settings_path: Path, offline: bool,
                 minio_client=None, pipeline_factory=None) -> None:
        self.data_dir = data_dir
        self.settings_path = settings_path
        self.offline = offline
        if minio_client is None:
            from minio import Minio
            endpoint = urlparse(os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000"))
            minio_client = Minio(
                endpoint.netloc,
                access_key=os.environ["MINIO_ACCESS_KEY"],
                secret_key=os.environ["MINIO_SECRET_KEY"],
                secure=endpoint.scheme == "https",
            )
        self.minio = minio_client
        self.bucket = os.environ.get("MINIO_BUCKET", "rag-documents")
        self.pipeline_factory = pipeline_factory or self._build_pipeline

    def ingest(self, payload: dict) -> dict:
        task_id = str(UUID(payload["taskId"]))
        document_id = str(UUID(payload["documentId"]))
        collection = payload["collection"]
        if not isinstance(collection, str) or not collection.isascii() or not collection.replace("_", "").replace("-", "").isalnum() or len(collection) > 100:
            raise ValueError("Invalid collection")
        object_key = f"documents/{document_id}/original.pdf"
        if payload["objectKey"] != object_key:
            raise ValueError("Object key does not match document ID")

        source = self.data_dir / "incoming" / document_id / "original.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            temporary = source.with_suffix(".download")
            try:
                self.minio.fget_object(self.bucket, object_key, str(temporary))
                with temporary.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("MinIO object is not a PDF")
                temporary.replace(source)
            finally:
                temporary.unlink(missing_ok=True)

        result = self.pipeline_factory().run(source, collection=collection)
        return {
            "taskId": task_id,
            "documentId": document_id,
            "skipped": result.skipped,
            "chunkCount": result.chunk_count,
        }

    def _build_pipeline(self):
        from core.settings import load_settings
        from scripts.ingest import _build_pipeline

        settings = load_settings(self.settings_path)
        return _build_pipeline(settings, self.data_dir, offline=self.offline)


def create_server(host: str, port: int, token: str, service: IngestService) -> HTTPServer:
    if not token:
        raise ValueError("PYTHON_INGEST_TOKEN is required")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/internal/ingest":
                self._respond(404, {"error": "Not found"})
                return
            if not hmac.compare_digest(self.headers.get("X-Ingest-Token", ""), token):
                self._respond(401, {"error": "Unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                result = service.ingest(payload)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._respond(400, {"error": str(exc)})
                return
            except Exception as exc:
                self.log_error("Ingest failed: %s", exc)
                self._respond(500, {"error": str(exc)[:500]})
                return
            self._respond(200, result)

        def _respond(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer((host, port), Handler)


def main() -> None:
    service = IngestService(
        data_dir=Path(os.environ.get("INGEST_DATA_DIR", str(ROOT / "data"))),
        settings_path=Path(os.environ.get("INGEST_SETTINGS", str(ROOT / "config/settings.yaml"))),
        offline=os.environ.get("INGEST_OFFLINE", "false").lower() == "true",
    )
    host = os.environ.get("PYTHON_INGEST_HOST", "127.0.0.1")
    port = int(os.environ.get("PYTHON_INGEST_PORT", "8090"))
    server = create_server(host, port, os.environ["PYTHON_INGEST_TOKEN"], service)
    print(f"Python ingest API listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
