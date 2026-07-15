"""Unit tests for the dashboard G1 shell and services."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from observability.dashboard.pages.overview import _load_vector_stats
from observability.dashboard.services import ConfigService


START_DASHBOARD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "start_dashboard.py"
SPEC = importlib.util.spec_from_file_location("start_dashboard", START_DASHBOARD_SCRIPT)
assert SPEC is not None
start_dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(start_dashboard)


def test_config_service_formats_components_and_redacts_sensitive_values(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "  api_key: secret-value",
                "embedding:",
                "  provider: openai",
                "  model: text-embedding-3-small",
                "vector_store:",
                "  provider: chroma",
                f"  persist_directory: {tmp_path / 'chroma'}",
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

    summary = ConfigService(settings_path).get_summary()

    assert summary["settings_path"] == str(settings_path)
    assert summary["component_count"] == 7
    components = {item["name"]: item for item in summary["components"]}
    assert components["LLM"]["provider"] == "openai"
    assert components["LLM"]["settings"]["api_key"] == "***"
    assert components["Retrieval"]["provider"] == "hybrid"
    assert components["Observability"]["provider"] == "INFO"


def test_overview_vector_stats_handles_missing_chroma_directory(tmp_path: Path) -> None:
    stats = _load_vector_stats(
        {
            "provider": "chroma",
            "persist_directory": str(tmp_path / "missing-chroma"),
        }
    )

    assert stats == {
        "provider": "chroma",
        "persist_directory": str(tmp_path / "missing-chroma"),
        "record_count": 0,
        "status": "not_initialized",
    }


def test_start_dashboard_builds_streamlit_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_call(command, cwd=None):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(start_dashboard.subprocess, "call", fake_call)

    exit_code = start_dashboard.main(
        ["--port", "8600", "--address", "127.0.0.1", "--headless"]
    )

    command = captured["command"]
    assert exit_code == 0
    assert isinstance(command, list)
    assert command[:3] == [start_dashboard.sys.executable, "-m", "streamlit"]
    assert "run" in command
    assert str(start_dashboard.APP_PATH) in command
    assert "--server.port" in command
    assert "8600" in command
    assert "--server.address" in command
    assert "127.0.0.1" in command
    assert "--server.headless" in command
    assert captured["cwd"] == start_dashboard.ROOT
