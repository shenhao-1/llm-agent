"""Streamlit dashboard entrypoint."""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="MR",
        layout="wide",
    )

    pages = {
        "Dashboard": [
            st.Page("pages/overview.py", title="System Overview"),
        ],
        "Data": [
            st.Page("pages/data_browser.py", title="Data Browser"),
            st.Page("pages/ingestion_manager.py", title="Ingestion Manager"),
        ],
        "Traces": [
            st.Page("pages/ingestion_traces.py", title="Ingestion Traces"),
            st.Page("pages/query_traces.py", title="Query Traces"),
        ],
        "Evaluation": [
            st.Page("pages/evaluation_panel.py", title="Evaluation Panel"),
        ],
    }
    navigation = st.navigation(pages)
    navigation.run()


if __name__ == "__main__":
    main()
