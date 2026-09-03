"""
pages/batch_ris_downloader.py — Batch RIS Bibliographic Downloader.

Accepts a CSV file containing multiple Scopus advanced search queries,
executes the queries sequentially, compiles the metadata into canonical
RIS format using rispy, and generates a unified .ris download.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import io
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ris_exporter import process_query_batch
from scopus_client import init_pybliometrics

load_dotenv()
env_api_key = os.getenv("SCOPUS_API_KEY", "").strip()
if env_api_key == "your_api_key_here":
    env_api_key = ""

# --- Sidebar Configuration -------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key_input = st.text_input(
        "Elsevier / Scopus API Key",
        value=env_api_key,
        type="password",
        help="Required for querying Scopus metadata.",
        key="ris_api_key_input",
    )

    inst_token_input = st.text_input(
        "Institutional Token (Optional)",
        value=os.getenv("SCOPUS_INST_TOKEN", "").strip(),
        type="password",
        help="Optional institutional token.",
        key="ris_inst_token_input",
    )

active_api_key = api_key_input.strip()

# --- Header -----------------------------------------------------------------

st.title("Batch RIS Bibliographic Downloader")
st.caption(
    "Automated batch utility for compiling research citations into standard "
    "Research Information Systems (RIS) format from tabular query specifications."
)

if not active_api_key:
    st.warning(
        "Authentication required: Please specify a valid Scopus API key in the sidebar "
        "or configure SCOPUS_API_KEY in the .env file."
    )
    st.stop()

init_pybliometrics(active_api_key, inst_token_input.strip() or None)

# --- Template CSV Generator -------------------------------------------------

st.subheader("Query Specification Input")

col_info, col_template = st.columns([3, 1])
with col_info:
    st.markdown(
        "Upload a CSV file containing Scopus search queries. The file must contain "
        "either a column named `query` or `search_query`, or the queries should be "
        "listed in the first column."
    )
with col_template:
    sample_csv = (
        'query\n'
        'TITLE-ABS-KEY ( ( "Advanced Material" OR "Advanced Materials" ) AND "Safe and sustainable by Design" )\n'
        'TITLE-ABS-KEY ( "Safe-by-Design" AND "Nanomaterials" AND PUBYEAR > 2022 )\n'
    )
    st.download_button(
        label="Download Template CSV",
        data=sample_csv.encode("utf-8"),
        file_name="scopus_queries_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --- File Uploader ----------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Query File",
    type=["csv", "txt"],
    help="Select a CSV or delimited text file containing Scopus search patterns.",
)

queries_to_run: list[str] = []

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        df_queries = pd.read_csv(io.StringIO(content))

        query_col = None
        for col_name in ["query", "search_query", "pattern", "scopus_query", "queries"]:
            for actual_col in df_queries.columns:
                if actual_col.strip().lower() == col_name:
                    query_col = actual_col
                    break
            if query_col:
                break

        if not query_col:
            query_col = df_queries.columns[0]

        st.info(f"Targeting query column: '{query_col}'")
        raw_queries = df_queries[query_col].dropna().astype(str).tolist()
        queries_to_run = [q.strip() for q in raw_queries if q.strip()]

        st.markdown(f"**Identified Queries ({len(queries_to_run)}):**")
        st.dataframe(
            pd.DataFrame({"Query Index": range(1, len(queries_to_run) + 1), "Search Query": queries_to_run}),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Error parsing uploaded file: {exc}")
        st.stop()

# --- Batch Processing Trigger -----------------------------------------------

if queries_to_run:
    execute_batch = st.button("Execute Batch Retrieval and RIS Compilation", type="primary")

    if execute_batch:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio: float, msg: str) -> None:
            progress_bar.progress(min(max(ratio, 0.0), 1.0))
            status_text.text(msg)

        with st.spinner("Processing batch queries via Scopus Search API..."):
            ris_content, q_count, ref_count, summaries = process_query_batch(
                queries=queries_to_run,
                progress_callback=update_progress,
            )

        progress_bar.empty()
        status_text.empty()

        # Save to session_state
        st.session_state["batch_ris_content"] = ris_content
        st.session_state["batch_q_count"] = q_count
        st.session_state["batch_ref_count"] = ref_count
        st.session_state["batch_summaries"] = summaries

# --- Results & Download -----------------------------------------------------

if "batch_ris_content" in st.session_state and st.session_state["batch_ris_content"]:
    ris_text = st.session_state["batch_ris_content"]
    q_count = st.session_state["batch_q_count"]
    ref_count = st.session_state["batch_ref_count"]
    summaries = st.session_state["batch_summaries"]

    st.markdown("---")
    st.subheader("Batch Execution Summary")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Queries Executed", str(q_count))
    with m2:
        st.metric("Total References Compiled", str(ref_count))
    with m3:
        st.metric("RIS Payload Size", f"{len(ris_text.encode('utf-8')) / 1024:.1f} KB")

    st.markdown("**Per-Query Breakdown:**")
    st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Download Bibliographic Data")

    st.download_button(
        label="Download Combined RIS File (.ris)",
        data=ris_text.encode("utf-8"),
        file_name="scopus_batch_export.ris",
        mime="application/x-research-info-systems",
        use_container_width=True,
    )
    st.caption(
        "Standard Research Information Systems (RIS) format compatible with Zotero, "
        "Mendeley, EndNote, Citavi, and JabRef."
    )

    with st.expander("Preview Compiled RIS Data (First 1,500 Characters)"):
        st.code(ris_text[:1500], language="text")
