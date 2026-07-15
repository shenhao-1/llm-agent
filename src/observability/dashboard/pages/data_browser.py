"""Dashboard page for browsing ingested documents and chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from observability.dashboard.services import DataService


def render() -> None:
    import streamlit as st

    st.title("Data Browser")
    try:
        service = DataService.from_settings()
    except Exception as exc:
        st.error(f"Unable to initialize data service: {exc}")
        return

    collections = service.list_collections()
    selected_collection = st.selectbox(
        "Collection",
        ["All collections", *collections],
    )
    collection_filter = (
        None if selected_collection == "All collections" else selected_collection
    )
    stats = service.get_collection_stats(collection_filter)
    left, middle, right = st.columns(3)
    left.metric("Documents", stats["document_count"])
    middle.metric("Chunks", stats["chunk_count"])
    right.metric("Images", stats["image_count"])

    documents = service.list_documents(collection_filter)
    if not documents:
        st.info("No ingested documents found.")
        return

    st.dataframe(
        documents,
        use_container_width=True,
        hide_index=True,
        column_order=[
            "source_path",
            "collection",
            "chunk_count",
            "image_count",
            "status",
            "updated_at",
        ],
    )
    options = {document["source_path"]: document["doc_id"] for document in documents}
    selected_source = st.selectbox("Document", list(options))
    detail = service.get_document_detail(options[selected_source])
    _render_detail(st, detail)


def _render_detail(st: Any, detail: dict[str, Any]) -> None:
    info = detail["info"]
    st.subheader(info["source_path"])
    st.json(info)

    st.markdown("### Chunks")
    for index, chunk in enumerate(detail["chunks"], start=1):
        label = f"{index}. {chunk['id']}"
        with st.expander(label):
            st.text_area(
                "Text",
                value=str(chunk.get("text", "")),
                height=160,
                key=f"chunk-text-{chunk['id']}",
            )
            st.json(chunk.get("metadata", {}))

    st.markdown("### Images")
    if not detail["images"]:
        st.caption("No images linked to this document.")
        return
    for image in detail["images"]:
        image_path = Path(str(image.get("file_path", "")))
        with st.expander(str(image.get("image_id", image_path.name))):
            st.json(image)
            if image_path.exists():
                st.image(str(image_path), caption=image_path.name)
            else:
                st.warning(f"Image file missing: {image_path}")


if __name__ == "__main__":
    render()
