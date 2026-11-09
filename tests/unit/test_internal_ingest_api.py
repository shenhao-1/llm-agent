"""Contract and duplicate-call checks for the private ingestion bridge."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts.internal_ingest_api import IngestService


class FakeMinio:
    def __init__(self):
        self.downloads = 0

    def fget_object(self, bucket, key, path):
        self.downloads += 1
        Path(path).write_bytes(b"%PDF-1.7\nexample")


def test_repeated_task_uses_stable_source_and_pipeline_idempotency(tmp_path):
    minio = FakeMinio()
    calls = []

    class Pipeline:
        def run(self, source, *, collection):
            calls.append((source, collection))
            return SimpleNamespace(skipped=len(calls) > 1, chunk_count=0 if len(calls) > 1 else 2)

    service = IngestService(data_dir=tmp_path, settings_path=tmp_path / "settings.yaml",
                            offline=True, minio_client=minio, pipeline_factory=Pipeline)
    document_id = str(uuid4())
    task_id = str(uuid4())
    payload = {"documentId": document_id, "taskId": task_id, "collection": "docs",
               "objectKey": f"documents/{document_id}/original.pdf"}

    first = service.ingest(payload)
    second = service.ingest(payload)

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert minio.downloads == 1
    assert calls[0][0] == calls[1][0]


def test_rejects_mismatched_object_key(tmp_path):
    service = IngestService(data_dir=tmp_path, settings_path=tmp_path / "settings.yaml",
                            offline=True, minio_client=FakeMinio(), pipeline_factory=lambda: None)
    with pytest.raises(ValueError, match="Object key"):
        service.ingest({"taskId": str(uuid4()), "documentId": str(uuid4()),
                        "collection": "docs", "objectKey": "documents/other/original.pdf"})
