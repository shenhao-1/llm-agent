"""Unit tests for the recursive splitter provider."""

from __future__ import annotations

from textwrap import dedent

import pytest

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


@pytest.fixture(autouse=True)
def reset_factory_registry():
    original = dict(SplitterFactory._providers)
    SplitterFactory._providers.clear()
    yield
    SplitterFactory._providers = original


@pytest.mark.unit
def test_factory_routes_recursive_provider() -> None:
    splitter = SplitterFactory.create(
        {
            "splitter": {
                "provider": "recursive",
                "chunk_size": 80,
                "chunk_overlap": 10,
            }
        }
    )

    assert isinstance(splitter, RecursiveSplitter)


@pytest.mark.unit
def test_recursive_splitter_handles_markdown_without_empty_chunks() -> None:
    splitter = RecursiveSplitter({"chunk_size": 80, "chunk_overlap": 10})
    text = dedent(
        """
        # Title

        Intro paragraph with enough content to require splitting across multiple chunks.

        ## Section

        ```python
        def hello():
            return "world"
        ```

        Final paragraph that should remain readable.
        """
    ).strip()

    chunks = splitter.split_text(text)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert any("# Title" in chunk for chunk in chunks)
    assert any("```python" in chunk for chunk in chunks)
    assert any("Final paragraph" in chunk for chunk in chunks)


@pytest.mark.unit
def test_recursive_splitter_rejects_invalid_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="recursive splitter chunk_overlap must be smaller than chunk_size",
    ):
        RecursiveSplitter({"chunk_size": 100, "chunk_overlap": 100})
