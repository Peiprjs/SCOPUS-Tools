"""
pages/interactive_analysis.py — Interactive Bibliometric Dashboard.

Primary interactive dashboard for institutional affiliation analysis,
geopolitical classification (EU/EEC vs. Non-EU/EEC vs. Mixed Geo),
cascading Country/Institution filters, toggleable DOI-based full-text
retrieval with abstract fallback, and keyword frequency dynamics.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from classifier import classify_geography, classify_paper
from fulltext_client import enrich_dataset_with_text
from keyword_matcher import add_keyword_features, parse_keywords
from network_analyzer import (
    build_citation_graph,
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

active_api_key = api_key_input.strip()

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

    with st.spinner("Querying Scopus Search API..."):
        try:
            raw_data = search_scopus(query.strip())
        except RuntimeError as exc:
            st.error(f"Scopus API Query Failure: {exc}")
            st.stop()

    if raw_data.empty:
        st.warning("No publication records matched the provided query.")
        st.stop()

    st.session_state["raw_df"] = raw_data
    st.session_state["last_query"] = query.strip()
    st.session_state["enable_fulltext"] = enable_fulltext

    # Progress-tracked text enrichment
    progress_bar = st.progress(0.0)
    status_box = st.empty()

    def update_progress(ratio: float, msg: str) -> None:
        progress_bar.progress(min(max(ratio, 0.0), 1.0))
        status_box.text(msg)

    spinner_msg = (
        "Retrieving article text from Elsevier API (with abstract fallback)..."
        if enable_fulltext
        else "Extracting abstracts (full-text API bypassed)..."
    )

    with st.spinner(spinner_msg):
        enriched = enrich_dataset_with_text(
            df=raw_data,
            api_key=active_api_key,
            max_fulltext=max_fulltext if enable_fulltext else 0,
            progress_callback=update_progress,
            inst_token=inst_token_input.strip() or None,
            fetch_full_text=enable_fulltext,
        )

    progress_bar.empty()
    status_box.empty()

    st.session_state["enriched_df"] = enriched

    # Citation Reference Extraction for Internal Network Mapping
    progress_bar_cit = st.progress(0.0)
    status_box_cit = st.empty()

    def update_cit_progress(ratio: float, msg: str) -> None:
        progress_bar_cit.progress(min(max(ratio, 0.0), 1.0))
        status_box_cit.text(msg)

    with st.spinner("Harvesting references and resolving internal cross-citations..."):
        edges = extract_citation_edges(
            df=enriched,
            api_key=active_api_key,
            inst_token=inst_token_input.strip() or None,
            progress_callback=update_cit_progress,
        )

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

# --- Tabbed Analytical Views ------------------------------------------------

tab_affil, tab_geo, tab_kw, tab_net, tab_desc, tab_data = st.tabs(
    [
        "Affiliation Trends",
        "Geo Trends",
        "Keyword Frequency",
        "Citation Network",
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
            xaxis=dict(dtick=1, showgrid=True, gridcolor="#e5e5e5"),
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="#e5e5e5"),
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
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
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
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
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
                xaxis=dict(dtick=1, showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
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


# --- Tab 5: Descriptive Metrics ---------------------------------------------

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


# --- Tab 6: Dataset & Export ------------------------------------------------

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
