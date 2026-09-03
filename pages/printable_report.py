"""
pages/printable_report.py — A4 Monochrome Printable Report.

A mathematically constrained A4 view designed for high-resolution
black-and-white printing. Strips all color, employing Plotly pattern_shape
fills and the simple_white template for publication-ready figures.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from classifier import classify_geography
from latex_exporter import create_latex_bundle

# --- Page Header & Styling (A4 Layout Constraints) -------------------------

A4_CSS = """
<style>
/* Approximate A4 proportions at 96 DPI: 210mm width (~794px), 297mm height */
.a4-container {
    max-width: 210mm;
    margin: 0 auto;
    padding: 12mm 15mm;
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #d0d0d0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    font-family: 'Times New Roman', Times, serif;
}

.a4-header {
    border-bottom: 2px solid #000000;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

.a4-title {
    font-size: 20pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 4px 0;
}

.a4-metadata {
    font-size: 9.5pt;
    color: #333333;
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.a4-section-title {
    font-size: 13pt;
    font-weight: bold;
    border-bottom: 1px solid #000000;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 10px;
    font-family: Arial, sans-serif;
}

.a4-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    font-family: Arial, sans-serif;
    margin: 10px 0;
}

.a4-table th, .a4-table td {
    border: 1px solid #000000;
    padding: 4px 6px;
    text-align: left;
}

.a4-table th {
    background-color: #f0f0f0;
}

@media print {
    @page {
        size: A4 portrait;
        margin: 12mm 15mm;
    }
    body {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .a4-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"], .no-print {
        display: none !important;
    }
}
</style>
"""

st.markdown(A4_CSS, unsafe_allow_html=True)

# --- Data Preparation -------------------------------------------------------

analyzed_df: pd.DataFrame | None = st.session_state.get("analyzed_df")
parsed_kws: list[str] = st.session_state.get("parsed_kws", [])
year_range: tuple[int, int] = st.session_state.get("year_range", (2018, datetime.datetime.now().year))
active_query: str = st.session_state.get("active_query", 'TITLE-ABS-KEY("Advanced Materials")')

# Demo data fallback if user navigated here before executing Page 1
if analyzed_df is None or analyzed_df.empty:
    st.info(
        "Notice: No active analysis dataset found in the current session. "
        "You can execute a search on the 'Interactive Analysis' page, or load "
        "representative benchmark data below to preview the A4 monochrome report."
    )

    load_demo = st.button("Load Representative Benchmark Data for A4 Preview")
    if load_demo:
        sample_years = [2021, 2022, 2023, 2024, 2025] * 8
        sample_cats = ["Academia"] * 18 + ["Mixed"] * 14 + ["Industry"] * 6 + ["Unknown"] * 2
        sample_geo = ["EU/EEC"] * 18 + ["Mixed Geo"] * 16 + ["Non-EU/EEC"] * 6
        demo_df = pd.DataFrame(
            {
                "eid": [f"2-s2.0-{i:06d}" for i in range(40)],
                "title": [f"Bibliometric Study on Advanced Materials {i+1}" for i in range(40)],
                "year": sample_years[:40],
                "category": sample_cats[:40],
                "geo_category": sample_geo[:40],
                "countries": [["Germany", "Spain"] if i % 2 == 0 else ["Italy", "United States"] for i in range(40)],
                "institutions": [["Max Planck Institute"] if i % 2 == 0 else ["MIT", "CNRS"] for i in range(40)],
                "text_source": ["Full Text"] * 18 + ["Abstract"] * 22,
                "total_keyword_hits": [2, 0, 4, 1, 0] * 8,
                "has_any_keyword": [True, False, True, True, False] * 8,
                "kw_safety_present": [True, False, True, False, False] * 8,
                "kw_safety_count": [2, 0, 1, 0, 0] * 8,
                "kw_sustainability_present": [False, False, True, True, False] * 8,
                "kw_sustainability_count": [0, 0, 3, 1, 0] * 8,
                "kw_nano_present": [True, False, False, False, False] * 8,
                "kw_nano_count": [1, 0, 0, 0, 0] * 8,
                "matched_keywords": [["safety"], [], ["safety", "sustainability"], ["sustainability"], []] * 8,
            }
        )
        analyzed_df = demo_df
        parsed_kws = ["safety", "sustainability", "nano"]
        st.session_state["analyzed_df"] = demo_df
        st.session_state["parsed_kws"] = parsed_kws
        st.rerun()
    else:
        st.stop()

# Ensure geo_category is present
if "geo_category" not in analyzed_df.columns:
    if "countries" in analyzed_df.columns:
        analyzed_df["geo_category"] = analyzed_df["countries"].apply(classify_geography)
    else:
        analyzed_df["geo_category"] = "Unknown Geo"

# --- LaTeX Report Bundle Export ---------------------------------------------

st.subheader("LaTeX Report Bundle Generation")
st.caption(
    "Synthesizes a complete standalone LaTeX (.tex) document accompanied by "
    "high-resolution monochrome figures (rendered with Plotly pattern shapes and simple_white), "
    "packaged into a downloadable ZIP archive."
)

col_bundle1, col_bundle2 = st.columns([2, 2])
with col_bundle1:
    generate_bundle_btn = st.button(
        "Generate LaTeX Report Bundle",
        type="primary",
        use_container_width=True,
    )

if generate_bundle_btn:
    with st.spinner("Rendering monochrome figures via Kaleido and assembling ZIP archive..."):
        metadata = {
            "query": active_query,
            "year_start": year_range[0],
            "year_end": year_range[1],
            "selected_countries": st.session_state.get("selected_countries", []),
            "selected_institutions": st.session_state.get("selected_institutions", []),
            "fulltext_enabled": st.session_state.get("enable_fulltext", False),
        }
        zip_bytes = create_latex_bundle(
            analyzed_df=analyzed_df,
            parsed_kws=parsed_kws,
            metadata=metadata,
        )
        st.session_state["latex_zip_bytes"] = zip_bytes

if "latex_zip_bytes" in st.session_state and st.session_state["latex_zip_bytes"]:
    zip_payload = st.session_state["latex_zip_bytes"]
    with col_bundle2:
        st.download_button(
            label="Download LaTeX Report Bundle (.zip)",
            data=zip_payload,
            file_name=f"scopus_report_bundle_{year_range[0]}_{year_range[1]}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("LaTeX Bundle Manifest & Compilation Instructions"):
        st.markdown(
            """
            **Archive Contents:**
            * `report.tex`: Standalone LaTeX document with geometry, booktabs, graphicx, and figure floats.
            * `fig_affiliation_distribution.png`: Sectoral distribution (Academia vs. Industry).
            * `fig_institutional_trends.png`: Temporal share over publication years.
            * `fig_geopolitical_correlation.png`: Geopolitical scope cross-tabulated with institutional sector.
            * `fig_keyword_prevalence.png`: Keyword occurrence frequency across sectors (if keywords specified).

            **Compilation Instructions:**
            ```bash
            unzip scopus_report_bundle_*.zip
            pdflatex report.tex
            ```
            """
        )

st.markdown("---")

# --- Print Controls Bar (No Print) ------------------------------------------

col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    st.caption(
        "Printer-Ready A4 View: Formatted strictly for monochrome reproduction. "
        "Differentiates series through Plotly pattern shapes and linestyles."
    )
with col_ctrl2:
    st.components.v1.html(
        """
        <button onclick="window.print()" style="
            background-color: #000000;
            color: #ffffff;
            border: 1px solid #000000;
            padding: 8px 16px;
            font-family: Arial, sans-serif;
            font-size: 11pt;
            cursor: pointer;
            width: 100%;
            text-transform: uppercase;
            font-weight: bold;
        ">
            Print Document
        </button>
        """,
        height=45,
    )

st.markdown("---")

# --- A4 Document Body Container ---------------------------------------------

total_papers = len(analyzed_df)
fulltext_papers = int((analyzed_df["text_source"] == "Full Text").sum())
abstract_papers = int((analyzed_df["text_source"] == "Abstract").sum())
fulltext_share = (fulltext_papers / total_papers * 100) if total_papers > 0 else 0.0

cat_dist = analyzed_df["category"].value_counts()
academia_count = int(cat_dist.get("Academia", 0))
industry_count = int(cat_dist.get("Industry", 0))
mixed_count = int(cat_dist.get("Mixed", 0))
unknown_count = int(cat_dist.get("Unknown", 0))

geo_dist = analyzed_df["geo_category"].value_counts()
eu_count = int(geo_dist.get("EU/EEC", 0))
noneu_count = int(geo_dist.get("Non-EU/EEC", 0))
mixed_geo_count = int(geo_dist.get("Mixed Geo", 0))

PATTERN_MAP = {
    "Academia": "",
    "Industry": "/",
    "Mixed": "x",
    "Unknown": ".",
}

is_ft_enabled = st.session_state.get("enable_fulltext", False)

st.markdown(
    f"""
    <div class="a4-container">
        <div class="a4-header">
            <div class="a4-title">Bibliometric & Geopolitical Analysis Report</div>
            <div class="a4-metadata">
                <span><b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                <span><b>Standard:</b> Monochrome A4 Technical Document</span>
            </div>
            <div class="a4-metadata">
                <span><b>Query:</b> {active_query}</span>
                <span><b>Temporal Scope:</b> {year_range[0]} – {year_range[1]}</span>
            </div>
        </div>

        <div class="a4-section-title">1. Executive Summary & Key Indicators</div>
        <table class="a4-table">
            <tr>
                <th>Metric Description</th>
                <th>Quantification</th>
                <th>Proportion / Details</th>
            </tr>
            <tr>
                <td>Total Indexed Publications</td>
                <td><b>{total_papers:,}</b></td>
                <td>100.0% of analyzed cohort</td>
            </tr>
            <tr>
                <td>Text Retrieval Strategy</td>
                <td><b>{'Full-Text Retrieval Active' if is_ft_enabled else 'Abstract-Only (API Bypassed)'}</b></td>
                <td>{fulltext_share:.1f}% Full Text ({fulltext_papers:,} Full / {abstract_papers:,} Abstract)</td>
            </tr>
            <tr>
                <td>Sovereign EU/EEC Research Cohort</td>
                <td><b>{eu_count:,}</b></td>
                <td>{(eu_count/total_papers*100):.1f}% (Member State Affiliations Only)</td>
            </tr>
            <tr>
                <td>Transnational Collaboration (Mixed Geo)</td>
                <td><b>{mixed_geo_count:,}</b></td>
                <td>{(mixed_geo_count/total_papers*100):.1f}% (EU/EEC + International Co-authors)</td>
            </tr>
            <tr>
                <td>Non-EU/EEC Sovereign Cohort</td>
                <td><b>{noneu_count:,}</b></td>
                <td>{(noneu_count/total_papers*100):.1f}% (Outside EU/EEC Perimeter)</td>
            </tr>
            <tr>
                <td>Academic Institutional Share</td>
                <td><b>{academia_count:,}</b></td>
                <td>{(academia_count/total_papers*100):.1f}%</td>
            </tr>
            <tr>
                <td>Corporate / Commercial Share</td>
                <td><b>{industry_count:,}</b></td>
                <td>{(industry_count/total_papers*100):.1f}%</td>
            </tr>
            <tr>
                <td>Cross-Sector Collaboration (Mixed Sector)</td>
                <td><b>{mixed_count:,}</b></td>
                <td>{(mixed_count/total_papers*100):.1f}%</td>
            </tr>
        </table>
    """,
    unsafe_allow_html=True,
)

# --- Monochrome Figure 1: Institutional Affiliation Distribution ------------

st.markdown('<div class="a4-section-title">2. Institutional Affiliation Distribution</div>', unsafe_allow_html=True)

affil_summary = analyzed_df["category"].value_counts().reset_index()
affil_summary.columns = ["Category", "Count"]

fig_affil_mono = px.bar(
    affil_summary,
    x="Category",
    y="Count",
    color="Category",
    color_discrete_sequence=["#ffffff", "#f0f0f0", "#e0e0e0", "#d0d0d0"],
    pattern_shape="Category",
    pattern_shape_sequence=["", "/", "x", "."],
    template="simple_white",
    labels={"Category": "Institutional Classification", "Count": "Publication Count"},
)
fig_affil_mono.update_traces(
    marker=dict(line=dict(width=1.5, color="#000000")),
    textposition="outside",
    texttemplate="%{y}",
)
fig_affil_mono.update_layout(
    showlegend=False,
    height=260,
    margin=dict(l=40, r=20, t=20, b=30),
    xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
    yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
)
st.plotly_chart(fig_affil_mono, use_container_width=True)

# --- Monochrome Figure 2: Affiliation Proportions Over Time -----------------

st.markdown('<div class="a4-section-title">3. Institutional Proportions Over Time</div>', unsafe_allow_html=True)

yearly_cat = (
    analyzed_df.groupby(["year", "category"]).size().reset_index(name="count")
)
yearly_sums = yearly_cat.groupby("year")["count"].transform("sum")
yearly_cat["share"] = (yearly_cat["count"] / yearly_sums * 100).round(2)

LINE_DASH_MAP = {
    "Academia": "solid",
    "Industry": "dash",
    "Mixed": "dashdot",
    "Unknown": "dot",
}
SYMBOL_MAP = {
    "Academia": "circle",
    "Industry": "square",
    "Mixed": "diamond",
    "Unknown": "triangle-up",
}

fig_trend_mono = go.Figure()
for cat in sorted(yearly_cat["category"].unique()):
    sub = yearly_cat[yearly_cat["category"] == cat]
    fig_trend_mono.add_trace(
        go.Scatter(
            x=sub["year"],
            y=sub["share"],
            name=cat,
            mode="lines+markers",
            line=dict(
                color="#000000",
                width=1.5,
                dash=LINE_DASH_MAP.get(cat, "solid"),
            ),
            marker=dict(
                color="#000000",
                size=6,
                symbol=SYMBOL_MAP.get(cat, "circle"),
            ),
        )
    )

fig_trend_mono.update_layout(
    template="simple_white",
    height=260,
    margin=dict(l=40, r=20, t=20, b=35),
    xaxis=dict(title="Publication Year", dtick=1, showline=True, linecolor="#000000"),
    yaxis=dict(title="Proportion of Year (%)", range=[0, 105], showline=True, linecolor="#000000"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.35,
        xanchor="center",
        x=0.5,
        font=dict(size=10, color="#000000"),
    ),
)
st.plotly_chart(fig_trend_mono, use_container_width=True)

# --- Monochrome Figure 3: Geopolitical & Cross-Sector Distribution ----------

st.markdown('<div class="a4-section-title">4. Geopolitical Scope & Sectoral Cross-Correlation</div>', unsafe_allow_html=True)

cross_geo = (
    analyzed_df.groupby(["geo_category", "category"]).size().reset_index(name="count")
)

fig_geo_cross = px.bar(
    cross_geo,
    x="geo_category",
    y="count",
    color="category",
    barmode="group",
    color_discrete_sequence=["#ffffff", "#f0f0f0", "#e0e0e0", "#d0d0d0"],
    pattern_shape="category",
    pattern_shape_sequence=["", "/", "x", "."],
    template="simple_white",
    labels={
        "geo_category": "Geopolitical Perimeter",
        "count": "Publication Count",
        "category": "Sectoral Category",
    },
)
fig_geo_cross.update_traces(
    marker=dict(line=dict(width=1.5, color="#000000")),
)
fig_geo_cross.update_layout(
    height=260,
    margin=dict(l=40, r=20, t=20, b=35),
    xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
    yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.35,
        xanchor="center",
        x=0.5,
        font=dict(size=10, color="#000000"),
    ),
)
st.plotly_chart(fig_geo_cross, use_container_width=True)

# --- Monochrome Figure 4: Keyword Prevalence Across Affiliations ------------

if parsed_kws:
    st.markdown('<div class="a4-section-title">5. Target Keyword Incidence by Sector</div>', unsafe_allow_html=True)

    kw_rows: list[dict[str, Any]] = []
    for kw in parsed_kws:
        presence_col = f"kw_{kw}_present"
        for cat in ["Academia", "Industry", "Mixed", "Unknown"]:
            sub = analyzed_df[analyzed_df["category"] == cat]
            c = int(sub[presence_col].sum()) if presence_col in sub.columns else 0
            kw_rows.append({"Keyword": kw, "Category": cat, "Papers": c})

    kw_mono_df = pd.DataFrame(kw_rows)

    fig_kw_mono = px.bar(
        kw_mono_df,
        x="Keyword",
        y="Papers",
        color="Category",
        barmode="group",
        color_discrete_sequence=["#ffffff", "#f0f0f0", "#e0e0e0", "#d0d0d0"],
        pattern_shape="Category",
        pattern_shape_sequence=["", "/", "x", "."],
        template="simple_white",
        labels={"Papers": "Paper Count", "Keyword": "Target Term"},
    )
    fig_kw_mono.update_traces(
        marker=dict(line=dict(width=1.5, color="#000000")),
    )
    fig_kw_mono.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=20, b=35),
        xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
        yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=10)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#000000"),
        ),
    )
    st.plotly_chart(fig_kw_mono, use_container_width=True)

# --- Document Footer --------------------------------------------------------

st.markdown(
    """
        <div style="border-top: 1px solid #000000; margin-top: 25px; padding-top: 6px; font-size: 8pt; color: #555555; text-align: center; font-family: Arial, sans-serif;">
            Rendered for Black-and-White Reproduction · Scopus Affiliation Analyzer v0.3.0 · FAIR Compliant Open Research Software
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
