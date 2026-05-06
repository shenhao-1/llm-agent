"""Smoke tests for package imports."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_top_level_packages_importable() -> None:
    for package_name in ["mcp_server", "core", "ingestion", "libs", "observability"]:
        module = importlib.import_module(package_name)
        assert module is not None
