"""
pages/mass_dataset_exporter.py — Mass Dataset & Provenance ZIP Exporter Page.

Provides a Streamlit interface for executing batch Scopus queries defined
in a CSV specification, applying full-text extraction, bibliometric
classifications, granular filtering, and exporting complete cryptographic
provenance manifests in an organized ZIP bundle.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from batch_dataset_exporter import (
    JobResult,
    JobSpecification,
    build_batch_provenance_zip,
    execute_batch_job,
    generate_template_csv,
    parse_batch_spec_csv,
)
from prism_exporter import generate_prism_xml
from scopus_client import format_eta, init_pybliometrics

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
        key="mass_api_key_input",
    )

    inst_token_input = st.text_input(
        "Institutional Token (Optional)",
        value=os.getenv("SCOPUS_INST_TOKEN", "").strip(),
        type="password",
        help="Optional institutional token for subscriber-level features.",
        key="mass_inst_token_input",
    )

    st.markdown("---")
    batch_debug_mode = st.toggle(
        "Enable Batch Debug Diagnostics",
        value=False,
        help="Display low-level telemetry, timing breakdowns, and raw JSON manifests.",
    )

    export_prism_toggle = st.toggle(
        "Export GraphPad Prism Project (.pzfx)",
        value=True,
        help="Automatically compile a unified GraphPad Prism XML project (.pzfx) containing multi-table and multi-graph datasets across all batch jobs.",
        key="mass_export_prism_toggle",
    )

active_api_key = api_key_input.strip()
active_inst_token = inst_token_input.strip() or None

# --- Page Header -----------------------------------------------------------

st.title("Mass Dataset & Provenance Exporter")
st.caption(
    "Automated batch utility for executing multi-query research cohorts from tabular CSV specifications, "
    "applying thematic classifications, granular filters, cryptographic integrity verification, "
    "and compiling structured datasets and provenance manifests into a unified research bundle (.zip)."
)

if not active_api_key:
    st.warning(
        "Authentication Required: Please specify a valid Scopus API key in the sidebar "
        "or configure SCOPUS_API_KEY in the .env file."
    )
    st.stop()

init_pybliometrics(active_api_key, active_inst_token)

# --- Specification CSV Input Section ---------------------------------------

st.markdown("---")
st.subheader("Batch Specification Input")

col_info, col_template = st.columns([3, 1])
with col_info:
    st.markdown(
        "Upload a CSV specification defining the batch research queries. Each row represents "
        "an independent retrieval cohort with customizable parameters:\n\n"
        "- **`query`**: Advanced Scopus search syntax (e.g. `TITLE-ABS-KEY(\"Safe-by-Design\")`).\n"
        "- **`job_name`** *(optional)*: Cohort descriptor used for labeling and folder names.\n"
        "- **`keywords`** *(optional)*: Semicolon- or comma-separated terms to evaluate within abstracts/full-texts.\n"
        "- **`countries`** *(optional)*: Semicolon- or comma-separated countries to filter affiliation cohorts.\n"
        "- **`institutions`** *(optional)*: Semicolon- or comma-separated institutions to filter affiliation cohorts.\n"
        "- **`fetch_full_text`** *(optional)*: Boolean flag (`True`/`False`) enabling Elsevier Article Retrieval API calls.\n"
        "- **`max_results`** *(optional)*: Maximum publications to retrieve per query (default: 50)."
    )
with col_template:
    template_data = generate_template_csv()
    st.download_button(
        label="Download Specification Template (.csv)",
        data=template_data.encode("utf-8"),
        file_name="mass_export_specification_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

uploaded_spec = st.file_uploader(
    "Upload Batch Specification CSV",
    type=["csv", "txt"],
    help="Select a CSV file containing query definitions and execution parameters.",
    key="mass_spec_file_uploader",
)

parsed_jobs: list[JobSpecification] = []

if uploaded_spec is not None:
    try:
        content = uploaded_spec.getvalue().decode("utf-8")
        parsed_jobs = parse_batch_spec_csv(content)

        if not parsed_jobs:
            st.error("The uploaded CSV does not contain any valid query rows.")
            st.stop()

        st.markdown(f"**Identified Query Cohorts ({len(parsed_jobs)}):**")
        table_rows = [
            {
                "Job ID": j.job_id,
                "Cohort Name": j.job_name,
                "Search Query": j.query,
                "Target Keywords": "; ".join(j.keywords) if j.keywords else "None",
                "Country Filters": "; ".join(j.countries) if j.countries else "All Countries",
                "Institution Filters": "; ".join(j.institutions) if j.institutions else "All Institutions",
                "Full Text API": "Active" if j.fetch_full_text else "Bypassed (Abstract Only)",
                "Limit": j.max_results,
            }
            for j in parsed_jobs
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Error parsing specification file: {exc}")
        st.stop()

# --- Execution Section -----------------------------------------------------

if parsed_jobs:
    st.markdown("---")
    st.subheader("Execute Batch Retrieval")

    execute_clicked = st.button(
        f"Execute Batch Extraction ({len(parsed_jobs)} Cohorts)",
        type="primary",
        key="btn_execute_mass_export",
    )

    if execute_clicked:
        total_jobs = len(parsed_jobs)
        overall_progress = st.progress(0.0)
        overall_status = st.empty()
        stage_status = st.empty()

        completed_results: list[JobResult] = []
        batch_start_time = time.time()

        for idx, job in enumerate(parsed_jobs):
            job_num = idx + 1
            # Estimate remaining time
            elapsed = time.time() - batch_start_time
            avg_per_job = (elapsed / idx) if idx > 0 else 0.0
            rem_jobs = total_jobs - idx
            eta_str = format_eta(avg_per_job * rem_jobs) if idx > 0 else "Calculating..."

            overall_status.text(
                f"Processing Cohort {job_num} of {total_jobs}: '{job.job_name}' | Elapsed: {elapsed:.1f}s | Batch ETA: {eta_str}"
            )

            def update_sub_progress(fraction: float, msg: str) -> None:
                # Scaled job progress
                curr_ratio = (idx + fraction) / total_jobs
                overall_progress.progress(min(max(curr_ratio, 0.0), 1.0))
                stage_status.text(f"[{job.job_name}] {msg}")

            try:
                res = execute_batch_job(
                    job_spec=job,
                    api_key=active_api_key,
                    inst_token=active_inst_token,
                    progress_callback=update_sub_progress,
                )
                completed_results.append(res)
            except Exception as exc:
                st.error(f"Execution failure on Job {job.job_id} ('{job.job_name}'): {exc}")

        overall_progress.progress(1.0)
        overall_status.text(f"Completed all {total_jobs} cohorts in {time.time() - batch_start_time:.2f} seconds.")
        stage_status.empty()

        # Compile ZIP archive and Prism project
        with st.spinner("Compiling cryptographic provenance manifests and assembling ZIP archive..."):
            zip_bytes = build_batch_provenance_zip(
                results=completed_results,
                batch_metadata={
                    "title": "Scopus Batch Bibliometric Research Bundle",
                    "source_specification": uploaded_spec.name if uploaded_spec else "uploaded_spec.csv",
                },
                export_prism=export_prism_toggle,
            )
            prism_xml = ""
            if export_prism_toggle and completed_results:
                try:
                    prism_xml = generate_prism_xml(
                        completed_results,
                        project_title="Scopus Batch Bibliometric Research Bundle",
                    )
                except Exception:
                    prism_xml = ""

        st.session_state["mass_export_results"] = completed_results
        st.session_state["mass_export_zip"] = zip_bytes
        st.session_state["mass_export_prism_xml"] = prism_xml
        st.session_state["mass_export_time"] = time.time() - batch_start_time

# --- Results & Download Section --------------------------------------------

if "mass_export_results" in st.session_state and st.session_state["mass_export_results"]:
    results: list[JobResult] = st.session_state["mass_export_results"]
    zip_data: bytes = st.session_state["mass_export_zip"]
    prism_xml: str = st.session_state.get("mass_export_prism_xml", "")
    total_time_taken: float = st.session_state.get("mass_export_time", 0.0)

    st.markdown("---")
    st.subheader("Batch Execution & Provenance Summary")

    total_cohorts = len(results)
    total_retrieved = sum(r.total_retrieved for r in results)
    total_cohort_records = sum(r.total_filtered for r in results)
    zip_size_kb = len(zip_data) / 1024.0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Cohorts Executed", str(total_cohorts))
    with k2:
        st.metric("Raw Records Retrieved", f"{total_retrieved:,}")
    with k3:
        st.metric("Filtered Cohort Records", f"{total_cohort_records:,}")
    with k4:
        st.metric("Total Processing Time", f"{total_time_taken:.1f}s")
    with k5:
        st.metric("ZIP Bundle Size", f"{zip_size_kb:.1f} KB")

    st.markdown("---")
    st.subheader("Download Research Artifacts")
    st.caption(
        "The ZIP archive includes the overarching `batch_manifest.json`, `consolidated_dataset.csv`, "
        "`consolidated_citations.ris`, and dedicated subfolders for each cohort containing individual "
        "CSV datasets, RIS bibliographies, and SHA-256 cryptographic provenance manifests."
        + (" A consolidated GraphPad Prism project (`scopus_mass_export.pzfx`) is also bundled." if prism_xml else "")
    )

    if prism_xml:
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label="Download Complete Research Bundle (.zip)",
                data=zip_data,
                file_name="scopus_mass_export_bundle.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        with c_dl2:
            st.download_button(
                label="Download GraphPad Prism Project (.pzfx)",
                data=prism_xml.encode("utf-8"),
                file_name="scopus_mass_export.pzfx",
                mime="application/xml",
                use_container_width=True,
            )
    else:
        st.download_button(
            label="Download Complete Research Bundle (.zip)",
            data=zip_data,
            file_name="scopus_mass_export_bundle.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Per-Cohort Execution Breakdown")

    summary_rows = [
        {
            "Cohort ID": r.job_spec.job_id,
            "Cohort Name": r.job_spec.job_name,
            "Query": r.job_spec.query,
            "Retrieved": r.total_retrieved,
            "Filtered Cohort": r.total_filtered,
            "Full Text Mode": "Active" if r.job_spec.fetch_full_text else "Bypassed",
            "Full Text Rate": f"{r.provenance.get('full_text_telemetry', {}).get('full_text_retrieval_rate_pct', 0.0):.1f}%",
            "Duration": f"{r.duration_seconds:.2f}s",
            "Dataset SHA-256": r.provenance.get("artifact_verification", {}).get("dataset_csv", {}).get("sha256_checksum", "")[:16] + "...",
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # --- Interactive Inspect Tabs ------------------------------------------

    tab_titles = [
        "Master Batch Manifest",
        "Consolidated Dataset Preview",
        "Cohort Provenance Inspector",
    ]
    if prism_xml:
        tab_titles.append("GraphPad Prism Project Preview")

    created_tabs = st.tabs(tab_titles)
    tab_manifest = created_tabs[0]
    tab_consolidated = created_tabs[1]
    tab_provenance = created_tabs[2]

    with tab_manifest:
        st.markdown("**Master Batch Manifest (`batch_manifest.json`)**")
        st.caption("Cryptographic summary of all jobs, input parameters, cohort sizes, and file checksums.")
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf_manifest:
                manifest_obj = json.loads(zf_manifest.read("batch_manifest.json").decode("utf-8"))
        except Exception:
            manifest_obj = {
                "summary": {
                    "total_jobs_executed": total_cohorts,
                    "total_records_retrieved": total_retrieved,
                    "total_records_in_cohorts": total_cohort_records,
                    "total_processing_seconds": round(total_time_taken, 2),
                },
                "jobs": [
                    {
                        "job_id": r.job_spec.job_id,
                        "job_name": r.job_spec.job_name,
                        "search_query": r.job_spec.query,
                        "target_keywords": r.job_spec.keywords,
                        "country_filters": r.job_spec.countries,
                        "institution_filters": r.job_spec.institutions,
                        "records_filtered": r.total_filtered,
                        "dataset_sha256": r.provenance.get("artifact_verification", {}).get("dataset_csv", {}).get("sha256_checksum"),
                        "citations_sha256": r.provenance.get("artifact_verification", {}).get("citations_ris", {}).get("sha256_checksum"),
                    }
                    for r in results
                ],
            }
        st.json(manifest_obj)

    with tab_consolidated:
        st.markdown("**Consolidated Dataset (`consolidated_dataset.csv`)**")
        st.caption("Unified tabular dataset with `batch_job_id` and `batch_job_name` identifier columns.")
        all_dfs = [r.df for r in results if not r.df.empty]
        if all_dfs:
            preview_df = pd.concat(all_dfs, ignore_index=True)
            st.dataframe(preview_df.head(50), use_container_width=True)
        else:
            st.text("All query cohorts returned zero records.")

    with tab_provenance:
        st.markdown("**Cohort Provenance Inspector (`provenance.json`)**")
        st.caption("Select a cohort to view its complete scientific and cryptographic provenance manifest.")
        cohort_choices = [f"Cohort {r.job_spec.job_id}: {r.job_spec.job_name}" for r in results]
        selected_choice = st.selectbox("Select Cohort:", cohort_choices)
        selected_idx = cohort_choices.index(selected_choice)
        st.json(results[selected_idx].provenance)

    if prism_xml:
        tab_prism = created_tabs[3]
        with tab_prism:
            st.markdown("**Unified GraphPad Prism Project (`scopus_mass_export.pzfx`)**")
            st.caption(
                "Standardized XML project conforming to PrismXMLVersion 5.00 containing multi-table "
                "and multi-graph datasets across all executed research cohorts."
            )

            p_bytes = prism_xml.encode("utf-8")
            p_sha256 = hashlib.sha256(p_bytes).hexdigest()
            pm1, pm2, pm3 = st.columns(3)
            with pm1:
                st.metric("XML Project Size", f"{len(p_bytes) / 1024.0:.1f} KB")
            with pm2:
                st.metric("Prism Specification", "XML 5.00 / Prism 10")
            with pm3:
                st.metric("SHA-256 Digest", p_sha256[:16] + "...")

            st.markdown("**Structured Data Tables & Graphs Catalog:**")
            prism_tables_summary = [
                {
                    "Table ID": "Table0",
                    "Title": "Sectoral Affiliation Dynamics",
                    "Graph Type": "OneWay (Column / Grouped Bar)",
                    "Variables": "Academia, Industry, Mixed, Unknown",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table1",
                    "Title": "Geopolitical Classification",
                    "Graph Type": "OneWay (Column / Grouped Bar)",
                    "Variables": "EU/EEC, Non-EU/EEC, Mixed Geo, Unknown Geo",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table2",
                    "Title": "Annual Publication Dynamics",
                    "Graph Type": "XY (Longitudinal Trajectories / Line Plots)",
                    "Variables": "Publication Year (X) vs Publication Count (Y)",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table3",
                    "Title": "Comparative Bibliometric Indicators",
                    "Graph Type": "OneWay (Metric Bar Charts)",
                    "Variables": "Total Papers, EU/EEC %, Keyword %, Full-Text %, Fallback %",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table4",
                    "Title": "Top Publishing Countries",
                    "Graph Type": "OneWay (Ranked Horizontal / Vertical Bar)",
                    "Variables": "Top 15 Countries across Cohorts",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table5",
                    "Title": "Top Publishing Institutions",
                    "Graph Type": "OneWay (Ranked Horizontal / Vertical Bar)",
                    "Variables": "Top 15 Institutions across Cohorts",
                    "Scope": f"All {len(results)} Cohorts",
                },
                {
                    "Table ID": "Table6",
                    "Title": "Target Keywords Occurrence Dynamics",
                    "Graph Type": "OneWay (Grouped Bar / Frequency Chart)",
                    "Variables": "Specified Target Keywords",
                    "Scope": f"All {len(results)} Cohorts",
                },
            ]
            for idx, r in enumerate(results):
                prism_tables_summary.append({
                    "Table ID": f"Table{7 + idx}",
                    "Title": f"Cohort Trajectory: {r.job_spec.job_name}",
                    "Graph Type": "XY (Annual Distribution)",
                    "Variables": f"Publication Years vs Cohort {r.job_spec.job_id} Counts",
                    "Scope": f"Cohort {r.job_spec.job_id}",
                })
            st.dataframe(pd.DataFrame(prism_tables_summary), use_container_width=True, hide_index=True)

            with st.expander("Inspect Raw GraphPad Prism XML Structure"):
                st.code(prism_xml[:4000] + ("\n... [Truncated for preview]" if len(prism_xml) > 4000 else ""), language="xml")

    # Developer diagnostics expander
    if batch_debug_mode:
        with st.expander("Batch Debug Diagnostics & Telemetry"):
            st.markdown("**ZIP File Integrity & Contents**")
            import zipfile
            z_buf = io.BytesIO(zip_data)
            with zipfile.ZipFile(z_buf, "r") as zf_test:
                file_list = [
                    {"File Name": info.filename, "Uncompressed Size (bytes)": info.file_size, "CRC32": hex(info.CRC)}
                    for info in zf_test.infolist()
                ]
                st.dataframe(pd.DataFrame(file_list), use_container_width=True, hide_index=True)
