"""Dashboard page for uploading, ingesting, and deleting documents."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services import IngestionService


def render() -> None:
    import streamlit as st

    st.title("Ingestion Manager")
    try:
        service = IngestionService.from_settings()
    except Exception as exc:
        st.error(f"Unable to initialize ingestion service: {exc}")
        return

    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader("PDF file", type=["pdf"])
    collection = st.text_input("Collection", value="default")
    force = st.checkbox("Force re-ingest", value=False)

    if st.button("Start ingestion", type="primary", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.warning("Select a PDF before starting ingestion.")
        else:
            _run_upload_ingestion(st, service, uploaded_file, collection, force)

    st.divider()
    _render_document_deletion(st, service)


def _run_upload_ingestion(
    st: Any,
    service: IngestionService,
    uploaded_file: Any,
    collection: str,
    force: bool,
) -> None:
    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(stage: str, current: int, total: int) -> None:
        ratio = current / total if total else 0.0
        progress_bar.progress(ratio)
        status.info(f"[{current}/{total}] {stage}")

    try:
        target = service.save_upload(
            uploaded_file.name,
            uploaded_file.getvalue(),
            collection=collection,
        )
        status.info(f"Saved upload to {target.path}")
        result = service.ingest_file(
            target.path,
            collection=target.collection,
            force=force,
            on_progress=on_progress,
        )
    except Exception as exc:
        st.error(f"Ingestion failed: {exc}")
        return

    progress_bar.progress(1.0)
    if result["skipped"]:
        st.info("Document was unchanged and ingestion was skipped.")
    else:
        st.success("Ingestion completed.")
    st.json(result)


def _render_document_deletion(st: Any, service: IngestionService) -> None:
    st.markdown("### Existing Documents")
    try:
        documents = service.list_documents()
    except Exception as exc:
        st.error(f"Unable to list documents: {exc}")
        return

    if not documents:
        st.info("No ingested documents found.")
        return

    collections = sorted({str(document["collection"]) for document in documents})
    selected_collection = st.selectbox(
        "Delete from collection",
        ["All collections", *collections],
    )
    collection_filter = (
        None if selected_collection == "All collections" else selected_collection
    )
    visible_documents = [
        document
        for document in documents
        if collection_filter is None or document["collection"] == collection_filter
    ]

    st.dataframe(
        visible_documents,
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
    selected_source = st.selectbox(
        "Document to delete",
        [str(document["source_path"]) for document in visible_documents],
    )
    if st.button("Delete selected document"):
        try:
            result = service.delete_document(selected_source, collection_filter)
        except Exception as exc:
            st.error(f"Document deletion failed: {exc}")
            return

        if result["deleted"]:
            st.success("Document deleted across available stores.")
        else:
            st.warning("Delete command completed, but no store reported deletions.")
        st.json(result)


if __name__ == "__main__":
    render()
