"""Application entrypoint for the MCP server runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.settings import SettingsValidationError, load_settings
from observability.logger import get_logger


def main() -> int:
    """Load settings and bootstrap the application."""
    parser = argparse.ArgumentParser(description="Start Modular RAG MCP server.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to the YAML settings file.",
    )
    args = parser.parse_args()

    logger = get_logger("main")
    settings_path = Path(args.settings)

    try:
        settings = load_settings(settings_path)
    except (OSError, SettingsValidationError) as exc:
        logger.error("Failed to load settings from %s: %s", settings_path, exc)
        return 1

    logger.info(
        "Settings loaded successfully (llm=%s, embedding=%s, vector_store=%s).",
        settings.llm.get("provider"),
        settings.embedding.get("provider"),
        settings.vector_store.get("provider"),
    )
    logger.info("Bootstrap complete. MCP server implementation will be added in Phase E.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
