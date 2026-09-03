"""
app.py — Multi-Page Router for Scopus Research Suite.

Coordinates subpage routing for:
1. Interactive Bibliometric Analysis
2. Monochrome Print-Ready Report (A4 Layout)
3. Batch RIS Bibliographic Downloader

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

__version__ = "0.3.0"

# Set global page configuration (strictly zero emojis)
st.set_page_config(
    page_title="Scopus Research Suite",
    layout="wide",
)

# Configure subpage navigation
pages = {
    "Analytics": [
        st.Page(
            "pages/interactive_analysis.py",
            title="Interactive Analysis",
            default=True,
        ),
        st.Page(
            "pages/printable_report.py",
            title="LaTeX & Print Report",
        ),
    ],
    "Bibliographic Utilities": [
        st.Page(
            "pages/batch_ris_downloader.py",
            title="Batch RIS Downloader",
        ),
    ],
}

navigation = st.navigation(pages)
navigation.run()
