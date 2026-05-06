"""Unit tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import SettingsValidationError, load_settings


@pytest.mark.unit
def test_load_settings_success() -> None:
    settings_path = Path("config/settings.yaml")
    settings = load_settings(settings_path)

    assert settings.llm["provider"] == "openai"
    assert settings.embedding["provider"] == "openai"
    assert settings.vector_store["provider"] == "chroma"


@pytest.mark.unit
def test_load_settings_missing_embedding_provider(tmp_path: Path) -> None:
    broken_settings = tmp_path / "broken.yaml"
    broken_settings.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openai",
                "embedding:",
                "  model: text-embedding-3-small",
                "vector_store:",
                "  provider: chroma",
                "retrieval:",
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

    with pytest.raises(SettingsValidationError, match="embedding.provider"):
        load_settings(broken_settings)
