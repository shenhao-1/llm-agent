"""System overview page for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services import ConfigService


def render() -> None:
    import streamlit as st

    st.title("System Overview")
    st.caption("Current configuration and local index state.")

    service = ConfigService()
    try:
        settings = service.load()
        components = service.get_component_configs()
    except Exception as exc:
        st.error(f"Unable to load settings: {exc}")
        return

    stats = _load_vector_stats(settings.vector_store)
    left, middle, right = st.columns(3)
    left.metric("Components", len(components))
    middle.metric("Vector Records", stats.get("record_count", 0))
    right.metric("Trace Level", settings.observability.get("level", "unknown"))

    st.subheader("Component Configuration")
    for component in components:
        with st.container(border=True):
            st.markdown(f"**{component.name}**")
            st.write(f"Provider: `{component.provider}`")
            st.json(component.settings)

    st.subheader("Vector Store")
    st.json(stats)


def _load_vector_stats(vector_store_config: dict[str, Any]) -> dict[str, Any]:
    provider = str(vector_store_config.get("provider", "")).strip().lower()
    if provider != "chroma":
        return {
            "provider": provider or "unknown",
            "record_count": 0,
            "status": "unsupported",
        }

    persist_directory = Path(str(vector_store_config.get("persist_directory", "")))
    if not persist_directory.exists():
        return {
            "provider": "chroma",
            "persist_directory": str(persist_directory),
            "record_count": 0,
            "status": "not_initialized",
        }

    try:
        return ChromaStore(vector_store_config).get_collection_stats()
    except Exception as exc:
        return {
            "provider": "chroma",
            "persist_directory": str(persist_directory),
            "record_count": 0,
            "status": "unavailable",
            "error": str(exc),
        }


if __name__ == "__main__":
    render()
