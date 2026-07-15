"""Tests for scripts/evaluate.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from core.types import RetrievalResult


EVALUATE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", EVALUATE_SCRIPT)
assert SPEC is not None
evaluate_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_script)


class FakeSearch:
    def search(self, query: str, top_k: int, filters=None, trace=None):
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=1.0,
                text="matched text",
                metadata={"source_path": "docs/a.pdf"},
            )
        ]


class FakeEvaluator:
    def evaluate(self, query: str, retrieved_ids: list[str], golden_ids: list[str], trace=None):
        return {"hit_rate": 1.0, "mrr": 1.0}


def test_evaluate_script_outputs_metrics(monkeypatch, tmp_path: Path, capsys) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "find docs",
                        "expected_chunk_ids": ["chunk-1"],
                        "expected_sources": ["docs/a.pdf"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_script, "load_settings", lambda path: {"retrieval": {"top_k": 5}, "evaluation": {"provider": "custom"}})
    monkeypatch.setattr(evaluate_script, "_build_search", lambda settings, data_dir: FakeSearch())
    monkeypatch.setattr(
        evaluate_script.EvaluatorFactory,
        "create",
        staticmethod(lambda settings: FakeEvaluator()),
    )

    code = evaluate_script.main(["--test-set", str(test_set)])

    output = capsys.readouterr()
    assert code == 0
    assert "Evaluation metrics" in output.out
    assert "hit_rate: 1.0000" in output.out
    assert "mrr: 1.0000" in output.out


def test_evaluate_script_reports_missing_index(monkeypatch, tmp_path: Path, capsys) -> None:
    test_set = tmp_path / "golden.json"
    test_set.write_text(
        json.dumps({"test_cases": [{"query": "q", "expected_chunk_ids": ["c"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_script, "load_settings", lambda path: {"retrieval": {"top_k": 5}, "evaluation": {"provider": "custom"}})

    def fail_search(settings, data_dir):
        raise evaluate_script.QuerySetupError("vector index not found")

    monkeypatch.setattr(evaluate_script, "_build_search", fail_search)

    code = evaluate_script.main(["--test-set", str(test_set)])

    output = capsys.readouterr()
    assert code == 1
    assert "No indexed data found" in output.out
