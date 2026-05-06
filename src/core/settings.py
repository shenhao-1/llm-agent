"""Settings loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping


class SettingsValidationError(ValueError):
    """Raised when settings are malformed or missing required fields."""


@dataclass(frozen=True)
class Settings:
    """Minimal configuration object used in early project phases."""

    llm: dict[str, Any]
    embedding: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    evaluation: dict[str, Any]
    observability: dict[str, Any]


def load_settings(path: str | Path) -> Settings:
    """Read a YAML file into a Settings object and validate required fields."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsValidationError(f"Settings file not found: {settings_path}")

    raw_text = settings_path.read_text(encoding="utf-8").lstrip("\ufeff")
    payload = _parse_yaml_mapping(raw_text)

    settings = Settings(
        llm=_as_dict(payload.get("llm"), "llm"),
        embedding=_as_dict(payload.get("embedding"), "embedding"),
        vector_store=_as_dict(payload.get("vector_store"), "vector_store"),
        retrieval=_as_dict(payload.get("retrieval"), "retrieval"),
        rerank=_as_dict(payload.get("rerank"), "rerank"),
        evaluation=_as_dict(payload.get("evaluation"), "evaluation"),
        observability=_as_dict(payload.get("observability"), "observability"),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Validate mandatory sections and mandatory keys."""
    _require_paths(
        settings,
        [
            "llm.provider",
            "embedding.provider",
            "vector_store.provider",
            "retrieval.top_k",
            "rerank.provider",
            "evaluation.provider",
            "observability.level",
        ],
    )


def _require_paths(settings: Settings, dotted_paths: list[str]) -> None:
    root: Mapping[str, Any] = {
        "llm": settings.llm,
        "embedding": settings.embedding,
        "vector_store": settings.vector_store,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank,
        "evaluation": settings.evaluation,
        "observability": settings.observability,
    }
    for dotted_path in dotted_paths:
        current: Any = root
        for node in dotted_path.split("."):
            if not isinstance(current, Mapping) or node not in current:
                raise SettingsValidationError(f"Missing required setting: {dotted_path}")
            current = current[node]
        if current is None or current == "":
            raise SettingsValidationError(f"Missing required setting: {dotted_path}")


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SettingsValidationError(f"Expected mapping for '{field_name}'")
    return value


def _parse_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]

        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise SettingsValidationError("Top-level settings must be a mapping")
        return parsed
    except ModuleNotFoundError:
        parsed = _parse_basic_yaml(text)
        if not isinstance(parsed, dict):
            raise SettingsValidationError("Top-level settings must be a mapping")
        return parsed


def _parse_basic_yaml(text: str) -> dict[str, Any]:
    """Parse a strict YAML subset: nested mappings with 2-space indentation."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, line in enumerate(text.splitlines(), start=1):
        no_comment = line.split("#", 1)[0].rstrip()
        if not no_comment.strip():
            continue

        indent = len(no_comment) - len(no_comment.lstrip(" "))
        if indent % 2 != 0:
            raise SettingsValidationError(
                f"Invalid indentation at line {lineno}: use multiples of 2 spaces"
            )

        stripped = no_comment.strip()
        if ":" not in stripped:
            raise SettingsValidationError(f"Invalid YAML line {lineno}: '{line}'")

        key, value_text = stripped.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SettingsValidationError(f"Invalid indentation hierarchy at line {lineno}")

        current = stack[-1][1]

        if value_text == "":
            next_map: dict[str, Any] = {}
            current[key] = next_map
            stack.append((indent, next_map))
        else:
            current[key] = _parse_scalar(value_text)

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if value.startswith("0") and value != "0" and not value.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


