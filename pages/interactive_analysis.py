"""
pages/interactive_analysis.py — Interactive Bibliometric Dashboard.

Primary interactive dashboard for institutional affiliation analysis,
geopolitical classification (EU/EEC vs. Non-EU/EEC vs. Mixed Geo),
cascading Country/Institution filters, toggleable DOI-based full-text
retrieval with abstract fallback, and keyword frequency dynamics.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from classifier import classify_geography, classify_paper
from fulltext_client import enrich_dataset_with_text
from keyword_matcher import add_keyword_features, parse_keywords
from network_analyzer import (
    apply_adaptive_theme,
    build_citation_graph,
    build_coauthorship_graph,
    build_coauthorship_plotly_figure,
    build_coinstitution_graph,
    build_coinstitution_plotly_figure,
    build_descriptive_frequency_figures,
    build_network_plotly_figure,
    extract_citation_edges,
)
from scopus_client import init_pybliometrics, search_scopus

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
        help="Obtain an API key from https://dev.elsevier.com/myapikey.html",
    )

    inst_token_input = st.text_input(
        "Institutional Token (Optional)",
        value=os.getenv("SCOPUS_INST_TOKEN", "").strip(),
        type="password",
        help="Optional institutional token for entitled full-text retrieval.",
    )

    st.markdown("---")
    st.header("Search Parameters")

    query = st.text_input(
        "Scopus Advanced Query",
        value='TITLE-ABS-KEY ( ( "Advanced Material" OR "Advanced Materials" ) AND ( "Safe and sustainable by Design" OR "Safe-and-Sustainable-by-Design" OR "SsbD" ) )',
        help="Follows Scopus advanced query syntax. See Elsevier documentation for syntax rules.",
    )

    raw_keywords_input = st.text_input(
        "Target Keywords (comma-separated)",
        value="safety, sustainability, toxic, lifecycle, nano, risk",
        help="Keywords to detect within retrieved text or fallback abstracts.",
    )

    current_year = datetime.now().year
    year_range: tuple[int, int] = st.slider(
        "Publication Year Range",
        min_value=1990,
        max_value=current_year,
        value=(2018, current_year),
    )

    st.markdown("---")
    st.header("Performance & API Management")

    enable_fulltext = st.toggle(
        "Enable Full-Text Retrieval (Slower)",
        value=False,
        help="When enabled, queries the Elsevier Article Retrieval API for each DOI. When disabled, completely bypasses the API and performs keyword matching exclusively on Scopus abstracts.",
    )

    max_fulltext: int = 50
    if enable_fulltext:
        max_fulltext = st.slider(
            "Max Full-Text API Requests",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Upper limit on full-text queries to respect Elsevier rate quotas.",
        )

    st.markdown("---")
    execute_clicked = st.button(
        "Execute Analysis",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.header("Developer & Diagnostics")
    debug_mode = st.toggle(
        "Enable Debug Mode",
        value=st.session_state.get("debug_mode", False),
        help="Display runtime telemetry, execution timings, network topology metrics, session state inspector, and raw metadata inspection.",
    )
    st.session_state["debug_mode"] = debug_mode

active_api_key = api_key_input.strip()

if debug_mode:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("scopus_client").setLevel(logging.DEBUG)
    logging.getLogger("network_analyzer").setLevel(logging.DEBUG)

# --- Header -----------------------------------------------------------------

st.title("Interactive Bibliometric Analysis")
st.caption(
    "Analytical dashboard for institutional affiliation categorization, geopolitical "
    "classification (EU/EEC vs. Non-EU/EEC), cascading geographical filters, and toggleable text retrieval."
)

if not active_api_key:
    st.warning(
        "Authentication required: Please specify a valid Scopus API key in the sidebar "
        "or configure SCOPUS_API_KEY in the .env file."
    )
    st.stop()

init_pybliometrics(active_api_key, inst_token_input.strip() or None)

# --- Execution Pipeline -----------------------------------------------------

if execute_clicked:
    if not query.strip():
        st.warning("Please specify a valid Scopus search query.")
        st.stop()

    st.markdown("### Execution Progress")

    # Phase 1: Scopus Search API Pagination Progress
    scopus_progress_bar = st.progress(0.0)
    scopus_status_box = st.empty()

    def update_scopus_progress(
        chunk_idx: int, total_chunks: int, current_count: int, total_count: int, msg: str
    ) -> None:
        ratio = chunk_idx / total_chunks if total_chunks > 0 else 1.0
        scopus_progress_bar.progress(min(max(ratio, 0.0), 1.0))
        scopus_status_box.text(msg)

    t_search_start = time.time()
    try:
        raw_data = search_scopus(
            query=query.strip(),
            count=25,
            api_key=active_api_key,
            inst_token=inst_token_input.strip() or None,
            progress_callback=update_scopus_progress,
        )
    except RuntimeError as exc:
        scopus_progress_bar.empty()
        scopus_status_box.empty()
        st.error(f"Scopus API Query Failure: {exc}")
        st.stop()

    st.session_state["timing_scopus_search"] = time.time() - t_search_start
    scopus_progress_bar.empty()
    scopus_status_box.empty()

    if raw_data.empty:
        st.warning("No publication records matched the provided query.")
        st.stop()

    st.session_state["raw_df"] = raw_data
    st.session_state["last_query"] = query.strip()
    st.session_state["enable_fulltext"] = enable_fulltext

    # Phase 2: Dual Progress Tracking for Text Retrieval (Full-Text Pipeline)
    ft_progress_bar = st.progress(0.0)
    ft_status_box = st.empty()

    def update_ft_progress(ratio: float, msg: str) -> None:
        ft_progress_bar.progress(min(max(ratio, 0.0), 1.0))
        ft_status_box.text(msg)

    t_ft_start = time.time()
    enriched = enrich_dataset_with_text(
        df=raw_data,
        api_key=active_api_key,
        max_fulltext=max_fulltext if enable_fulltext else 0,
        progress_callback=update_ft_progress,
        inst_token=inst_token_input.strip() or None,
        fetch_full_text=enable_fulltext,
    )
    st.session_state["timing_fulltext"] = time.time() - t_ft_start

    ft_progress_bar.empty()
    ft_status_box.empty()

    st.session_state["enriched_df"] = enriched

    # Citation Reference Extraction for Internal Network Mapping
    progress_bar_cit = st.progress(0.0)
    status_box_cit = st.empty()

    def update_cit_progress(ratio: float, msg: str) -> None:
        progress_bar_cit.progress(min(max(ratio, 0.0), 1.0))
        status_box_cit.text(msg)

    t_cit_start = time.time()
    with st.spinner("Harvesting references and resolving internal cross-citations..."):
        edges = extract_citation_edges(
            df=enriched,
            api_key=active_api_key,
            inst_token=inst_token_input.strip() or None,
            progress_callback=update_cit_progress,
        )
    st.session_state["timing_citations"] = time.time() - t_cit_start

    progress_bar_cit.empty()
    status_box_cit.empty()

    st.session_state["citation_edges"] = edges

if "enriched_df" not in st.session_state:
    st.info("Configure parameters in the sidebar and select 'Execute Analysis' to begin.")
    st.stop()

base_df: pd.DataFrame = st.session_state["enriched_df"]

if "citation_edges" not in st.session_state:
    with st.spinner("Extracting reference lists for internal network analysis..."):
        st.session_state["citation_edges"] = extract_citation_edges(
            df=base_df,
            api_key=active_api_key,
            inst_token=inst_token_input.strip() or None,
        )

if base_df.empty:
    st.warning("The dataset contains zero records.")
    st.stop()

# --- Cascading Country & Institution Sidebar Filters ------------------------

with st.sidebar:
    st.markdown("---")
    st.header("Granular Filters")

    # Extract all unique countries from base dataset
    all_countries_set: set[str] = set()
    for c_list in base_df.get("countries", []):
        if isinstance(c_list, list):
            for c in c_list:
                if c and str(c).strip():
                    all_countries_set.add(str(c).strip())
    all_countries = sorted(list(all_countries_set))

    selected_countries = st.multiselect(
        "Filter by Country",
        options=all_countries,
        default=[],
        help="Select one or more countries to restrict analysis cohort.",
    )

    # Cascading Institution options: restricted to selected countries if specified
    available_institutions_set: set[str] = set()
    for _, row in base_df.iterrows():
        details = row.get("affiliations_detail")
        if isinstance(details, list) and details:
            for item in details:
                inst_name = item.get("institution")
                cntry_name = item.get("country")
                if inst_name and inst_name != "Unknown Institution":
                    if not selected_countries or cntry_name in selected_countries:
                        available_institutions_set.add(inst_name)
        else:
            # Fallback if details not present
            inst_list = row.get("institutions") or row.get("affiliations")
            if isinstance(inst_list, list):
                for i in inst_list:
                    if i and str(i).strip():
                        available_institutions_set.add(str(i).strip())

    available_institutions = sorted(list(available_institutions_set))

    selected_institutions = st.multiselect(
        "Filter by Institution",
        options=available_institutions,
        default=[],
        help="Select specific research institutions (cascades from selected countries).",
    )

# --- Data Processing & Subset Filtering -------------------------------------

filtered_df = base_df.copy()
filtered_df = filtered_df.dropna(subset=["year"])
filtered_df["year"] = filtered_df["year"].astype(int)
filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) & (filtered_df["year"] <= year_range[1])
]

# Apply Country Filter
if selected_countries:
    filtered_df = filtered_df[
        filtered_df["countries"].apply(
            lambda c_list: any(c in selected_countries for c in c_list)
            if isinstance(c_list, list)
            else False
        )
    ]

# Apply Institution Filter
if selected_institutions:
    filtered_df = filtered_df[
        filtered_df["institutions"].apply(
            lambda i_list: any(i in selected_institutions for i in i_list)
            if isinstance(i_list, list)
            else False
        )
    ]

if filtered_df.empty:
    st.warning("No publication records match the applied year, country, or institutional filters.")
    st.stop()

# Classifications
filtered_df["category"] = filtered_df["affiliations"].apply(classify_paper)
filtered_df["geo_category"] = filtered_df["countries"].apply(classify_geography)

# Keyword Feature Extraction
parsed_kws = parse_keywords(raw_keywords_input)
analyzed_df = add_keyword_features(filtered_df, parsed_kws)

# Graph Construction & Centrality Metrics Calculation
citation_edges = st.session_state.get("citation_edges", [])
citation_graph, analyzed_df = build_citation_graph(analyzed_df, citation_edges)
st.session_state["citation_graph"] = citation_graph

# Author Collaboration Network Construction
coauth_graph, coauth_df = build_coauthorship_graph(analyzed_df)
st.session_state["coauth_graph"] = coauth_graph
st.session_state["coauth_df"] = coauth_df

# Institutional Collaboration Network Construction
coinstr_graph, coinst_df = build_coinstitution_graph(analyzed_df)
st.session_state["coinstr_graph"] = coinstr_graph
st.session_state["coinst_df"] = coinst_df

# Save to shared session_state for Printable Report page
st.session_state["analyzed_df"] = analyzed_df
st.session_state["parsed_kws"] = parsed_kws
st.session_state["year_range"] = year_range
st.session_state["active_query"] = query.strip()
st.session_state["selected_countries"] = selected_countries
st.session_state["selected_institutions"] = selected_institutions

# --- Top KPI Metrics Row ----------------------------------------------------

total_records = len(analyzed_df)
full_text_count = int((analyzed_df["text_source"] == "Full Text").sum())
abstract_count = int((analyzed_df["text_source"] == "Abstract").sum())
full_text_rate = (full_text_count / total_records * 100) if total_records > 0 else 0.0

is_ft_enabled = st.session_state.get("enable_fulltext", False)

eu_records = int((analyzed_df["geo_category"] == "EU/EEC").sum())
mixed_geo_records = int((analyzed_df["geo_category"] == "Mixed Geo").sum())
eu_involvement_rate = ((eu_records + mixed_geo_records) / total_records * 100) if total_records > 0 else 0.0

keyword_match_count = int(analyzed_df["has_any_keyword"].sum())
keyword_match_pct = (keyword_match_count / total_records * 100) if total_records > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="Total Papers Analyzed",
        value=f"{total_records:,}",
        help=f"Filtered cohort ({year_range[0]} - {year_range[1]}).",
    )
with m2:
    if is_ft_enabled:
        st.metric(
            label="Full-Text Status",
            value=f"{full_text_rate:.1f}% Retrieved",
            delta=f"{full_text_count} Full / {abstract_count} Abstract",
            delta_color="off",
            help="Full-text retrieval active via Elsevier Article Retrieval API.",
        )
    else:
        st.metric(
            label="Full-Text Status",
            value="Abstract Only",
            delta="API Bypassed (Fast)",
            delta_color="off",
            help="Full-text API bypassed to optimize latency and conserve API limits.",
        )
with m3:
    st.metric(
        label="EU/EEC Engagement Rate",
        value=f"{eu_involvement_rate:.1f}%",
        delta=f"{eu_records} Sovereign / {mixed_geo_records} Mixed",
        delta_color="off",
        help="Proportion of papers involving EU/EEC institutions (sovereign or international collaborative).",
    )
with m4:
    st.metric(
        label="Keyword Match Rate",
        value=f"{keyword_match_pct:.1f}%",
        delta=f"{keyword_match_count} of {total_records} papers",
        delta_color="off",
        help="Proportion of articles containing one or more user-defined target keywords.",
    )

st.markdown("---")

# --- Developer Diagnostics Panel (Debug Mode) -------------------------------

if debug_mode:
    with st.expander("Developer Diagnostics & Telemetry", expanded=True):
        st.subheader("System Telemetry & Performance")

        # 1. Execution Timers
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_scopus = st.session_state.get("timing_scopus_search", 0.0)
        t_ft = st.session_state.get("timing_fulltext", 0.0)
        t_cit = st.session_state.get("timing_citations", 0.0)
        total_api_time = t_scopus + t_ft + t_cit
        with t_col1:
            st.metric("Scopus Query Latency", f"{t_scopus:.2f}s")
        with t_col2:
            st.metric("Full-Text API Latency", f"{t_ft:.2f}s")
        with t_col3:
            st.metric("Citation Extraction Latency", f"{t_cit:.2f}s")
        with t_col4:
            st.metric("Total Network API Time", f"{total_api_time:.2f}s")

        st.markdown("---")
        st.markdown("**Graph Topology Telemetry**")

        topo_data = [
            {
                "Network Scope": "Internal Citation Network (Directed)",
                "Nodes (V)": len(citation_graph.nodes()),
                "Edges (E)": len(citation_graph.edges()),
                "Density": round(float(nx.density(citation_graph)), 5) if len(citation_graph) > 0 else 0.0,
                "Connected Components": nx.number_weakly_connected_components(citation_graph) if len(citation_graph) > 0 else 0,
                "Isolated Nodes": sum(1 for _, d in citation_graph.degree() if d == 0),
            },
            {
                "Network Scope": "Author Collaboration Network (Undirected)",
                "Nodes (V)": len(coauth_graph.nodes()),
                "Edges (E)": len(coauth_graph.edges()),
                "Density": round(float(nx.density(coauth_graph)), 5) if len(coauth_graph) > 0 else 0.0,
                "Connected Components": nx.number_connected_components(coauth_graph) if len(coauth_graph) > 0 else 0,
                "Isolated Nodes": sum(1 for _, d in coauth_graph.degree() if d == 0),
            },
            {
                "Network Scope": "Institutional Collaboration Network (Undirected)",
                "Nodes (V)": len(coinstr_graph.nodes()),
                "Edges (E)": len(coinstr_graph.edges()),
                "Density": round(float(nx.density(coinstr_graph)), 5) if len(coinstr_graph) > 0 else 0.0,
                "Connected Components": nx.number_connected_components(coinstr_graph) if len(coinstr_graph) > 0 else 0,
                "Isolated Nodes": sum(1 for _, d in coinstr_graph.degree() if d == 0),
            },
        ]
        st.dataframe(pd.DataFrame(topo_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Full-Text Extraction Telemetry & Status Breakdown**")
        ft_t1, ft_t2, ft_t3, ft_t4 = st.columns(4)
        total_cohort = len(analyzed_df)
        n_full = int((analyzed_df.get("text_source", pd.Series(dtype=str)) == "Full Text").sum())
        n_abs = int((analyzed_df.get("text_source", pd.Series(dtype=str)) == "Abstract").sum())
        n_none = int((analyzed_df.get("text_source", pd.Series(dtype=str)) == "None").sum())
        with ft_t1:
            st.metric("Pipeline Mode", "Full-Text API Active" if is_ft_enabled else "Abstract Bypassed")
        with ft_t2:
            st.metric("Full-Text Extracted", f"{n_full} / {total_cohort}")
        with ft_t3:
            st.metric("Abstract Fallbacks", f"{n_abs} / {total_cohort}")
        with ft_t4:
            st.metric("Missing Payloads", f"{n_none} / {total_cohort}")

        if "text_status_detail" in analyzed_df.columns:
            detail_counts = analyzed_df["text_status_detail"].value_counts().reset_index()
            detail_counts.columns = ["Status Detail", "Count"]
            detail_counts["Percentage"] = (detail_counts["Count"] / total_cohort * 100).round(1).astype(str) + "%"
            st.dataframe(detail_counts, use_container_width=True, hide_index=True)

            has_auth_err = analyzed_df["text_status_detail"].str.contains("Requestor configuration settings|403", case=False, na=False).any()
            if has_auth_err:
                st.info(
                    "ScienceDirect API Entitlement Advisory: Queries to the Elsevier Article Retrieval API returned HTTP 403 "
                    "('Requestor configuration settings insufficient for access to this resource'). While your credentials "
                    "successfully authorize Scopus Search and Abstract endpoints, Elsevier requires explicit ScienceDirect Text "
                    "and Data Mining (TDM) or campus IP-range authorization for full-text article XML/text downloads. "
                    "All records have safely fallen back to authoritative Scopus abstracts."
                )

        st.markdown("---")
        d_sub1, d_sub2 = st.columns(2)
        with d_sub1:
            st.markdown("**Dataset Schema & Memory Footprint**")
            mem_kb = analyzed_df.memory_usage(deep=True).sum() / 1024.0
            st.text(f"Dimensions: {analyzed_df.shape[0]} rows x {analyzed_df.shape[1]} columns")
            st.text(f"Total Memory Footprint: {mem_kb:.1f} KB")
            check_cols = [c for c in ["doi", "abstract", "authors", "institutions", "countries"] if c in analyzed_df.columns]
            null_summary = analyzed_df[check_cols].isnull().sum()
            st.text(f"Missing Values:\n{null_summary.to_string()}")
        with d_sub2:
            st.markdown("**Session State Variable Inspector**")
            session_keys_summary = [
                {"Key": k, "Type": type(v).__name__, "Size/Length": len(v) if hasattr(v, "__len__") else "N/A"}
                for k, v in st.session_state.items()
                if not k.startswith("_")
            ]
            st.dataframe(pd.DataFrame(session_keys_summary), use_container_width=True, hide_index=True)

        with st.expander("Inspect Raw Search Record Payloads (First 2 Documents)"):
            raw_sample = st.session_state.get("raw_df")
            if raw_sample is not None and not raw_sample.empty:
                st.json(raw_sample.head(2).to_dict(orient="records"))
            else:
                st.text("No raw dataset available in session.")

    st.markdown("---")

# --- Tabbed Analytical Views ------------------------------------------------

tab_affil, tab_geo, tab_kw, tab_net, tab_auth_net, tab_inst_net, tab_desc, tab_data = st.tabs(
    [
        "Affiliation Trends",
        "Geo Trends",
        "Keyword Frequency",
        "Citation Network",
        "Author Collaboration",
        "Institutional Collaboration",
        "Descriptive Metrics",
        "Structured Dataset & Export",
    ]
)

COLOR_MAP = {
    "Academia": "#1f77b4",
    "Industry": "#d62728",
    "Mixed": "#9467bd",
    "Unknown": "#7f7f7f",
}

GEO_COLOR_MAP = {
    "EU/EEC": "#2ca02c",
    "Non-EU/EEC": "#ff7f0e",
    "Mixed Geo": "#1f77b4",
    "Unknown Geo": "#7f7f7f",
}

# --- Tab 1: Affiliation Trends ----------------------------------------------

with tab_affil:
    st.subheader("Institutional Affiliation Dynamics")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Sectoral Classification Distribution**")
        cat_counts = analyzed_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_donut = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map=COLOR_MAP,
            hole=0.40,
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(family="Arial", size=13),
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("**Institutional Share Over Publication Years**")
        yearly_cat = (
            analyzed_df.groupby(["year", "category"]).size().reset_index(name="count")
        )
        yearly_sums = yearly_cat.groupby("year")["count"].transform("sum")
        yearly_cat["share"] = (yearly_cat["count"] / yearly_sums * 100).round(2)

        fig_temporal = px.line(
            yearly_cat,
            x="year",
            y="share",
            color="category",
            color_discrete_map=COLOR_MAP,
            markers=True,
            labels={
                "year": "Publication Year",
                "share": "Institutional Share (%)",
                "category": "Category",
            },
        )
        fig_temporal.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(dtick=1, showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

# --- Tab 2: Geo Trends ------------------------------------------------------

with tab_geo:
    st.subheader("Geopolitical Cohort Distribution & Cross-Sector Correlation")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Geopolitical Cohort Ratio (EU/EEC Perimeter)**")
        geo_counts = analyzed_df["geo_category"].value_counts().reset_index()
        geo_counts.columns = ["Geopolitical Scope", "Count"]

        fig_geo_donut = px.pie(
            geo_counts,
            names="Geopolitical Scope",
            values="Count",
            color="Geopolitical Scope",
            color_discrete_map=GEO_COLOR_MAP,
            hole=0.40,
        )
        fig_geo_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(family="Arial", size=13),
        )
        fig_geo_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_geo_donut, use_container_width=True)

    with g2:
        st.markdown("**Geopolitical Scope vs. Institutional Affiliation Breakdown**")
        cross_df = (
            analyzed_df.groupby(["geo_category", "category"]).size().reset_index(name="count")
        )

        fig_cross = px.bar(
            cross_df,
            x="geo_category",
            y="count",
            color="category",
            barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={
                "geo_category": "Geopolitical Scope",
                "count": "Publication Count",
                "category": "Affiliation Category",
            },
        )
        fig_cross.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_cross, use_container_width=True)

# --- Tab 3: Keyword Frequency -----------------------------------------------

with tab_kw:
    if not parsed_kws:
        st.info("Specify one or more comma-separated target keywords in the sidebar to view keyword analytics.")
    else:
        corpus_label = "Full Text + Abstracts" if is_ft_enabled else "Abstracts (Full-Text Bypassed)"
        st.subheader(f"Target Keyword Prevalence Across Affiliation Types ({corpus_label})")

        metric_mode = st.radio(
            "Aggregation Metric:",
            options=["Paper Count (Presence)", "Total Occurrences (Frequency)"],
            horizontal=True,
            key="interactive_kw_metric",
        )
        use_presence = metric_mode == "Paper Count (Presence)"

        kw_data_rows: list[dict[str, Any]] = []
        categories = ["Academia", "Industry", "Mixed", "Unknown"]

        for kw in parsed_kws:
            count_col = f"kw_{kw}_count"
            presence_col = f"kw_{kw}_present"

            for cat in categories:
                sub = analyzed_df[analyzed_df["category"] == cat]
                val = int(sub[presence_col].sum()) if use_presence else int(sub[count_col].sum())
                kw_data_rows.append(
                    {
                        "Keyword": kw,
                        "Category": cat,
                        "Value": val,
                    }
                )

        kw_chart_df = pd.DataFrame(kw_data_rows)

        fig_kw_bar = px.bar(
            kw_chart_df,
            x="Keyword",
            y="Value",
            color="Category",
            barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={
                "Value": "Paper Count" if use_presence else "Total Occurrences",
                "Keyword": "Target Keyword",
                "Category": "Affiliation Category",
            },
        )
        fig_kw_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_kw_bar, use_container_width=True)

        st.markdown("---")
        st.subheader(f"Temporal Keyword Dynamics ({corpus_label})")

        view_mode = st.radio(
            "Visualization Format:",
            options=["Heatmap Matrix", "Temporal Trend Lines"],
            horizontal=True,
            key="interactive_kw_view",
        )

        all_years = sorted(analyzed_df["year"].unique())

        if view_mode == "Heatmap Matrix":
            heatmap_matrix: list[list[int]] = []
            for kw in parsed_kws:
                row_vals: list[int] = []
                for yr in all_years:
                    yr_df = analyzed_df[analyzed_df["year"] == yr]
                    val = (
                        int(yr_df[f"kw_{kw}_present"].sum())
                        if use_presence
                        else int(yr_df[f"kw_{kw}_count"].sum())
                    )
                    row_vals.append(val)
                heatmap_matrix.append(row_vals)

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=heatmap_matrix,
                    x=all_years,
                    y=parsed_kws,
                    colorscale="Blues",
                    colorbar=dict(title="Count" if use_presence else "Occurrences"),
                )
            )
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Publication Year", dtick=1),
                yaxis=dict(title="Keyword"),
                margin=dict(l=20, r=20, t=20, b=30),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            trend_rows: list[dict[str, Any]] = []
            for kw in parsed_kws:
                for yr in all_years:
                    yr_df = analyzed_df[analyzed_df["year"] == yr]
                    val = (
                        int(yr_df[f"kw_{kw}_present"].sum())
                        if use_presence
                        else int(yr_df[f"kw_{kw}_count"].sum())
                    )
                    total_yr_papers = len(yr_df)
                    share = (val / total_yr_papers * 100) if total_yr_papers > 0 else 0.0
                    trend_rows.append(
                        {
                            "Year": yr,
                            "Keyword": kw,
                            "Value": val,
                            "Percentage": round(share, 2),
                        }
                    )

            trend_df = pd.DataFrame(trend_rows)
            fig_trend = px.line(
                trend_df,
                x="Year",
                y="Percentage" if use_presence else "Value",
                color="Keyword",
                markers=True,
                labels={
                    "Year": "Publication Year",
                    "Percentage": "Papers Containing Keyword (%)",
                    "Value": "Occurrences Count",
                },
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(dtick=1, showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=20, b=30),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

# --- Tab 4: Citation Network ------------------------------------------------

with tab_net:
    st.subheader("Internal Citation Network Analysis")
    st.caption(
        "Maps directed cross-citations restricted exclusively to publications within the active search pool. "
        "Connections indicate that Paper A explicitly references Paper B."
    )

    n_nodes = len(citation_graph.nodes())
    n_edges = len(citation_graph.edges())
    max_in = int(analyzed_df["in_degree"].max()) if "in_degree" in analyzed_df.columns and not analyzed_df.empty else 0
    density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

    cn1, cn2, cn3, cn4 = st.columns(4)
    with cn1:
        st.metric(label="Total Cohort Nodes", value=n_nodes)
    with cn2:
        st.metric(label="Internal Cross-Citations", value=n_edges)
    with cn3:
        st.metric(label="Max In-Citations Received", value=max_in)
    with cn4:
        st.metric(label="Network Density", value=f"{density:.4f}")

    st.markdown("---")

    scale_metric = st.radio(
        "Scale Node Size By Centrality Metric:",
        options=["In-Degree (Citations Received)", "Betweenness Centrality (Bridging Hubs)"],
        horizontal=True,
        key="net_node_scale",
    )
    metric_key = "in_degree" if "In-Degree" in scale_metric else "betweenness_centrality"

    fig_net = build_network_plotly_figure(
        G=citation_graph,
        df=analyzed_df,
        metric=metric_key,
        monochrome=False,
    )
    st.plotly_chart(fig_net, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Hub Papers (Cohort Cross-Citations)")
    st.caption("Ranked by internal in-degree citations received from other papers in this dataset.")

    if "in_degree" in analyzed_df.columns and not analyzed_df.empty:
        hub_df = analyzed_df.sort_values(
            by=["in_degree", "betweenness_centrality"], ascending=False
        ).head(10)[
            ["eid", "title", "year", "category", "in_degree", "out_degree", "betweenness_centrality", "pagerank", "doi"]
        ].copy()
        hub_df.columns = [
            "EID", "Title", "Year", "Sector", "In-Citations", "References", "Betweenness", "PageRank", "DOI"
        ]
        st.dataframe(hub_df, use_container_width=True, hide_index=True)


# --- Tab 5: Author Collaboration --------------------------------------------

with tab_auth_net:
    st.subheader("Author Collaboration Network Analysis")
    st.caption(
        "Models undirected co-authorship relationships across the research cohort. "
        "Edges connect researchers who co-authored one or more publications together, "
        "revealing collaborative research clusters and cross-team bridging authors."
    )

    n_authors = len(coauth_graph.nodes())
    n_auth_edges = len(coauth_graph.edges())
    density_auth = (2.0 * n_auth_edges / (n_authors * (n_authors - 1))) if n_authors > 1 else 0.0
    isolated_auth = sum(1 for _, d in coauth_graph.degree() if d == 0)

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric(label="Total Cohort Authors", value=f"{n_authors:,}")
    with ac2:
        st.metric(label="Co-authorship Ties", value=f"{n_auth_edges:,}")
    with ac3:
        st.metric(label="Isolated Authors", value=f"{isolated_auth:,}")
    with ac4:
        st.metric(label="Collaboration Density", value=f"{density_auth:.4f}")

    st.markdown("---")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        auth_scale_metric = st.radio(
            "Scale Node Size By:",
            options=[
                "Collaborators Count (Degree)",
                "Cohort Publications",
                "Betweenness Centrality (Bridging Authors)",
            ],
            horizontal=True,
            key="auth_node_scale",
        )
    with col_ctrl2:
        max_edge_w = max([d.get("weight", 1) for _, _, d in coauth_graph.edges(data=True)] or [1])
        min_coauth = st.slider(
            "Minimum Co-authored Papers Threshold:",
            min_value=1,
            max_value=max(2, max_edge_w),
            value=1,
            key="auth_min_coauth",
            help="Filter edges to only display relationships with at least this number of joint papers.",
        )

    auth_metric_key = (
        "collaborators_count"
        if "Collaborators" in auth_scale_metric
        else "publications"
        if "Publications" in auth_scale_metric
        else "betweenness_centrality"
    )

    if min_coauth > 1:
        filtered_auth_graph, _ = build_coauthorship_graph(analyzed_df, min_collaborations=min_coauth)
    else:
        filtered_auth_graph = coauth_graph

    fig_coauth = build_coauthorship_plotly_figure(
        G=filtered_auth_graph,
        author_df=coauth_df,
        metric=auth_metric_key,
        monochrome=False,
    )
    st.plotly_chart(fig_coauth, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Collaborating Researchers")
    st.caption("Ranked by number of distinct collaborators and total publication count.")

    if not coauth_df.empty:
        auth_display_df = coauth_df.head(15)[
            [
                "author",
                "publications",
                "collaborators_count",
                "collaboration_volume",
                "betweenness_centrality",
                "pagerank",
                "community",
            ]
        ].copy()
        auth_display_df.columns = [
            "Researcher",
            "Cohort Publications",
            "Collaborators",
            "Joint Papers",
            "Betweenness Centrality",
            "PageRank",
            "Research Cluster",
        ]
        st.dataframe(auth_display_df, use_container_width=True, hide_index=True)

    # Top Collaborating Pairs
    auth_pairs: list[dict[str, Any]] = []
    for u, v, data in coauth_graph.edges(data=True):
        w = data.get("weight", 1)
        auth_pairs.append({"Author 1": u, "Author 2": v, "Joint Publications": w})

    if auth_pairs:
        st.markdown("---")
        st.subheader("Top Research Partnerships (Author Pairs)")
        st.caption("Pairs of researchers with the highest number of co-authored publications in this cohort.")
        top_pairs_df = (
            pd.DataFrame(auth_pairs)
            .sort_values(by="Joint Publications", ascending=False)
            .head(10)
        )
        st.dataframe(top_pairs_df, use_container_width=True, hide_index=True)


# --- Tab 6: Institutional Collaboration -------------------------------------

with tab_inst_net:
    st.subheader("Institutional Collaboration Network Analysis")
    st.caption(
        "Maps inter-organizational co-affiliations across the research cohort. "
        "Nodes represent research organizations color-coded by sector (Academia blue, "
        "Industry red, Unknown gray), illustrating public-private research partnerships and consortia."
    )

    n_insts = len(coinstr_graph.nodes())
    n_inst_edges = len(coinstr_graph.edges())
    density_inst = (2.0 * n_inst_edges / (n_insts * (n_insts - 1))) if n_insts > 1 else 0.0
    isolated_inst = sum(1 for _, d in coinstr_graph.degree() if d == 0)

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.metric(label="Total Organizations", value=f"{n_insts:,}")
    with ic2:
        st.metric(label="Inter-organizational Ties", value=f"{n_inst_edges:,}")
    with ic3:
        st.metric(label="Independent Institutions", value=f"{isolated_inst:,}")
    with ic4:
        st.metric(label="Network Density", value=f"{density_inst:.4f}")

    st.markdown("---")

    icol_ctrl1, icol_ctrl2 = st.columns(2)
    with icol_ctrl1:
        inst_scale_metric = st.radio(
            "Scale Node Size By:",
            options=[
                "Partner Organizations (Degree)",
                "Cohort Publications",
                "Betweenness Centrality (Connecting Hubs)",
            ],
            horizontal=True,
            key="inst_node_scale",
        )
    with icol_ctrl2:
        max_inst_w = max([d.get("weight", 1) for _, _, d in coinstr_graph.edges(data=True)] or [1])
        min_coinst = st.slider(
            "Minimum Joint Publications Threshold:",
            min_value=1,
            max_value=max(2, max_inst_w),
            value=1,
            key="inst_min_coinst",
            help="Filter edges to only display partnerships with at least this number of joint papers.",
        )

    inst_metric_key = (
        "partners_count"
        if "Partner" in inst_scale_metric
        else "publications"
        if "Publications" in inst_scale_metric
        else "betweenness_centrality"
    )

    if min_coinst > 1:
        filtered_inst_graph, _ = build_coinstitution_graph(analyzed_df, min_collaborations=min_coinst)
    else:
        filtered_inst_graph = coinstr_graph

    fig_coinst = build_coinstitution_plotly_figure(
        G=filtered_inst_graph,
        inst_df=coinst_df,
        metric=inst_metric_key,
        monochrome=False,
    )
    st.plotly_chart(fig_coinst, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Hub Organizations & Consortia")
    st.caption("Ranked by number of partnering research organizations and publication volume.")

    if not coinst_df.empty:
        inst_display_df = coinst_df.head(15)[
            [
                "institution",
                "sector",
                "publications",
                "partners_count",
                "collaboration_volume",
                "betweenness_centrality",
                "pagerank",
            ]
        ].copy()
        inst_display_df.columns = [
            "Organization",
            "Sector",
            "Cohort Publications",
            "Partner Organizations",
            "Joint Papers",
            "Betweenness Centrality",
            "PageRank",
        ]
        st.dataframe(inst_display_df, use_container_width=True, hide_index=True)

    # Top Collaborating Institutional Pairs
    inst_pairs: list[dict[str, Any]] = []
    for u, v, data in coinstr_graph.edges(data=True):
        w = data.get("weight", 1)
        inst_pairs.append({"Organization 1": u, "Organization 2": v, "Shared Publications": w})

    if inst_pairs:
        st.markdown("---")
        st.subheader("Top Institutional Partnerships")
        st.caption("Pairs of organizations with the highest number of co-authored publications in this cohort.")
        top_inst_pairs_df = (
            pd.DataFrame(inst_pairs)
            .sort_values(by="Shared Publications", ascending=False)
            .head(10)
        )
        st.dataframe(top_inst_pairs_df, use_container_width=True, hide_index=True)


# --- Tab 7: Descriptive Metrics ---------------------------------------------

with tab_desc:
    st.subheader("Descriptive Bibliometric Metrics")
    st.caption(
        "Frequency distributions of leading researchers and research organizations across the publication cohort."
    )

    fig_top_auth, fig_top_inst = build_descriptive_frequency_figures(
        df=analyzed_df,
        top_n=20,
        monochrome=False,
    )

    c_desc1, c_desc2 = st.columns(2)
    with c_desc1:
        st.markdown("**Top 20 Most Frequent Authors**")
        st.plotly_chart(fig_top_auth, use_container_width=True)
    with c_desc2:
        st.markdown("**Top 20 Most Frequent Institutions**")
        st.plotly_chart(fig_top_inst, use_container_width=True)


# --- Tab 8: Structured Dataset & Export -------------------------------------

with tab_data:
    st.subheader("Structured Dataset Viewer")

    display_cols = [
        "eid",
        "title",
        "year",
        "category",
        "geo_category",
        "in_degree",
        "out_degree",
        "betweenness_centrality",
        "pagerank",
        "countries",
        "institutions",
        "text_source",
        "matched_keywords",
        "total_keyword_hits",
        "doi",
    ]
    present_cols = [c for c in display_cols if c in analyzed_df.columns]
    table_df = analyzed_df[present_cols].copy()

    for list_col in ["countries", "institutions", "matched_keywords"]:
        if list_col in table_df.columns:
            table_df[list_col] = table_df[list_col].apply(
                lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else str(x)
            )

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Data Export with Full Provenance")

    export_df = analyzed_df.copy()
    for list_col in ["countries", "institutions", "affiliations", "matched_keywords"]:
        if list_col in export_df.columns:
            export_df[list_col] = export_df[list_col].apply(
                lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else str(x)
            )

    if "text" in export_df.columns:
        export_df = export_df.drop(columns=["text"])

    export_df["provenance_query"] = st.session_state.get("last_query", query)
    export_df["provenance_keywords"] = ", ".join(parsed_kws)
    export_df["provenance_year_start"] = year_range[0]
    export_df["provenance_year_end"] = year_range[1]
    export_df["provenance_fulltext_enabled"] = is_ft_enabled
    export_df["provenance_filter_countries"] = ", ".join(selected_countries) if selected_countries else "All"
    export_df["provenance_filter_institutions"] = ", ".join(selected_institutions) if selected_institutions else "All"
    export_df["provenance_exported_at"] = datetime.now().isoformat()
    export_df["provenance_tool_version"] = "0.3.0"

    csv_payload = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results as CSV",
        data=csv_payload,
        file_name=f"scopus_analysis_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "The exported CSV incorporates complete provenance metadata columns "
        "facilitating reproduction and auditability."
    )
