"""
latex_exporter.py — LaTeX Report Bundle Generation Engine.

Generates standalone LaTeX (.tex) reports accompanied by publication-ready
monochrome figures exported via kaleido, packaged into a downloadable ZIP archive.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# LaTeX Formatting & Escaping
# ---------------------------------------------------------------------------

_LATEX_CHAR_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: Any) -> str:
    """Escape LaTeX special characters to ensure flawless compilation.

    Parameters
    ----------
    text:
        Raw string or object to format for LaTeX insertion.

    Returns
    -------
    Safely escaped LaTeX string.
    """
    if text is None:
        return ""
    return "".join(_LATEX_CHAR_MAP.get(ch, ch) for ch in str(text))


# ---------------------------------------------------------------------------
# Monochrome Figure Generation (Pattern Shapes & Simple White)
# ---------------------------------------------------------------------------

def build_monochrome_figures(
    analyzed_df: pd.DataFrame,
    parsed_kws: list[str] | None = None,
    citation_graph: Any = None,
) -> dict[str, go.Figure]:
    """Generate publication-ready black-and-white Plotly figures.

    Parameters
    ----------
    analyzed_df:
        DataFrame containing classified Scopus publication records.
    parsed_kws:
        Optional list of analyzed target keywords.

    Returns
    -------
    Dictionary mapping figure keys to Plotly Figure instances.
    """
    figures: dict[str, go.Figure] = {}
    if analyzed_df.empty:
        return figures

    # 1. Institutional Affiliation Distribution
    affil_summary = analyzed_df["category"].value_counts().reset_index()
    affil_summary.columns = ["Category", "Count"]

    fig_affil = px.bar(
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
    fig_affil.update_traces(
        marker=dict(line=dict(width=1.5, color="#000000")),
        textposition="outside",
        texttemplate="%{y}",
    )
    fig_affil.update_layout(
        showlegend=False,
        height=450,
        margin=dict(l=50, r=30, t=30, b=40),
        xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
        yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
    )
    figures["fig_affiliation_distribution"] = fig_affil

    # 2. Institutional Share Over Time
    yearly_cat = (
        analyzed_df.groupby(["year", "category"]).size().reset_index(name="count")
    )
    yearly_sums = yearly_cat.groupby("year")["count"].transform("sum")
    yearly_cat["share"] = (yearly_cat["count"] / yearly_sums * 100).round(2)

    line_dash_map = {
        "Academia": "solid",
        "Industry": "dash",
        "Mixed": "dashdot",
        "Unknown": "dot",
    }
    symbol_map = {
        "Academia": "circle",
        "Industry": "square",
        "Mixed": "diamond",
        "Unknown": "triangle-up",
    }

    fig_trends = go.Figure()
    for cat in sorted(yearly_cat["category"].unique()):
        sub = yearly_cat[yearly_cat["category"] == cat]
        fig_trends.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["share"],
                name=cat,
                mode="lines+markers",
                line=dict(
                    color="#000000",
                    width=1.5,
                    dash=line_dash_map.get(cat, "solid"),
                ),
                marker=dict(
                    color="#000000",
                    size=7,
                    symbol=symbol_map.get(cat, "circle"),
                ),
            )
        )

    fig_trends.update_layout(
        template="simple_white",
        height=450,
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(title="Publication Year", dtick=1, showline=True, linecolor="#000000"),
        yaxis=dict(title="Proportion of Year (%)", range=[0, 105], showline=True, linecolor="#000000"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#000000"),
        ),
    )
    figures["fig_institutional_trends"] = fig_trends

    # 3. Geopolitical Scope vs. Sector Correlation
    if "geo_category" in analyzed_df.columns:
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
            height=450,
            margin=dict(l=50, r=30, t=30, b=50),
            xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
            yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#000000"),
            ),
        )
        figures["fig_geopolitical_correlation"] = fig_geo_cross

    # 4. Keyword Prevalence by Sector
    if parsed_kws:
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
            height=450,
            margin=dict(l=50, r=30, t=30, b=50),
            xaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
            yaxis=dict(showline=True, linecolor="#000000", tickfont=dict(color="#000000", size=11)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#000000"),
            ),
        )
        figures["fig_keyword_prevalence"] = fig_kw_mono

    # 5. Citation Network Map (Monochrome)
    from network_analyzer import (
        build_citation_graph,
        build_descriptive_frequency_figures,
        build_network_plotly_figure,
    )

    if citation_graph is not None:
        figures["fig_citation_network"] = build_network_plotly_figure(
            citation_graph, analyzed_df, monochrome=True
        )
    else:
        G_fallback, _ = build_citation_graph(analyzed_df, [])
        figures["fig_citation_network"] = build_network_plotly_figure(
            G_fallback, analyzed_df, monochrome=True
        )

    # 6. Descriptive Frequency Visualizations (Top Authors & Institutions)
    fig_authors, fig_insts = build_descriptive_frequency_figures(
        analyzed_df, top_n=20, monochrome=True
    )
    figures["fig_top_authors"] = fig_authors
    figures["fig_top_institutions"] = fig_insts

    return figures


# ---------------------------------------------------------------------------
# LaTeX Document Generator
# ---------------------------------------------------------------------------

def generate_latex_document(
    analyzed_df: pd.DataFrame,
    parsed_kws: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    image_filenames: dict[str, str] | None = None,
) -> str:
    """Generate a clean, standalone LaTeX (.tex) report document.

    Parameters
    ----------
    analyzed_df:
        Classified publication records.
    parsed_kws:
        Analyzed keyword tokens.
    metadata:
        Search and filter parameters.
    image_filenames:
        Mapping of figure keys to relative image filenames for \\includegraphics.

    Returns
    -------
    Complete LaTeX document string.
    """
    if metadata is None:
        metadata = {}
    if image_filenames is None:
        image_filenames = {}

    query_escaped = escape_latex(metadata.get("query", "N/A"))
    year_start = metadata.get("year_start", "N/A")
    year_end = metadata.get("year_end", "N/A")
    countries_escaped = escape_latex(", ".join(metadata.get("selected_countries", [])) or "All")
    insts_escaped = escape_latex(", ".join(metadata.get("selected_institutions", [])) or "All")
    fulltext_status = "Active (Retrieved via Elsevier API)" if metadata.get("fulltext_enabled", False) else "Bypassed (Abstract-Only Analysis)"

    # Compute metric figures
    total_papers = len(analyzed_df)
    fulltext_count = int((analyzed_df["text_source"] == "Full Text").sum()) if "text_source" in analyzed_df.columns else 0
    abstract_count = int((analyzed_df["text_source"] == "Abstract").sum()) if "text_source" in analyzed_df.columns else 0

    cat_counts = analyzed_df["category"].value_counts() if "category" in analyzed_df.columns else pd.Series()
    academia_count = int(cat_counts.get("Academia", 0))
    industry_count = int(cat_counts.get("Industry", 0))
    mixed_count = int(cat_counts.get("Mixed", 0))

    geo_counts = analyzed_df["geo_category"].value_counts() if "geo_category" in analyzed_df.columns else pd.Series()
    eu_count = int(geo_counts.get("EU/EEC", 0))
    noneu_count = int(geo_counts.get("Non-EU/EEC", 0))
    mixed_geo_count = int(geo_counts.get("Mixed Geo", 0))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Construct LaTeX content
    tex_parts: list[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\usepackage{microtype}",
        r"",
        r"\title{\textbf{Bibliometric \& Geopolitical Research Report}}",
        r"\author{Scopus Affiliation Analyzer Suite v0.3.0}",
        rf"\date{{{now_str}}}",
        r"",
        r"\begin{document}",
        r"",
        r"\maketitle",
        r"",
        r"\section{Study Parameters \& Filter Configuration}",
        r"This document compiles institutional affiliation patterns, geopolitical cohort distributions (EU/EEC vs. Non-EU/EEC), and keyword dynamics derived from the Scopus database.",
        r"",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{lX}",
        r"\toprule",
        r"\textbf{Configuration Field} & \textbf{Value / Specification} \\",
        r"\midrule",
        rf"Scopus Advanced Query & \texttt{{{query_escaped}}} \\",
        rf"Temporal Window & {year_start} -- {year_end} \\",
        rf"Geographic Scope Filter & {countries_escaped} \\",
        rf"Institutional Scope Filter & {insts_escaped} \\",
        rf"Full-Text Retrieval Strategy & {fulltext_status} \\",
        rf"Target Keywords & {escape_latex(', '.join(parsed_kws or [])) or 'None'} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Analytical query parameters and filter boundaries.}",
        r"\end{table}",
        r"",
        r"\section{Executive Summary \& Key Performance Indicators}",
        r"The table below outlines aggregate cohort metrics across sector classifications and geopolitical boundaries.",
        r"",
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"\textbf{Analytical Metric} & \textbf{Count} & \textbf{Proportion} \\",
        r"\midrule",
        rf"Total Indexed Publications & {total_papers:,} & 100.0\% \\",
        rf"Full-Text Retrieved Articles & {fulltext_count:,} & {(fulltext_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        rf"Abstract Fallback Articles & {abstract_count:,} & {(abstract_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        r"\midrule",
        rf"Sovereign EU/EEC Research Cohort & {eu_count:,} & {(eu_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        rf"Transnational Collaboration (Mixed Geo) & {mixed_geo_count:,} & {(mixed_geo_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        rf"Non-EU/EEC Sovereign Cohort & {noneu_count:,} & {(noneu_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        r"\midrule",
        rf"Academic Sector Share & {academia_count:,} & {(academia_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        rf"Corporate / Industry Share & {industry_count:,} & {(industry_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        rf"Cross-Sector Collaboration (Mixed Sector) & {mixed_count:,} & {(mixed_count/total_papers*100 if total_papers else 0.0):.1f}\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cohort breakdown across geopolitical boundaries and institutional sectors.}",
        r"\end{table}",
        r"",
        r"\newpage",
        r"\section{Institutional Affiliation Dynamics}",
        r"Author affiliations are classified via heuristic detection into Academia, Industry, Mixed, or Unknown entities.",
        r"",
    ]

    # Embedded Figures
    if "fig_affiliation_distribution" in image_filenames:
        tex_parts.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_affiliation_distribution']}}}",
                r"\caption{Distribution of publication outputs across institutional sectors (Monochrome with pattern fills).}",
                r"\end{figure}",
                r"",
            ]
        )

    if "fig_institutional_trends" in image_filenames:
        tex_parts.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_institutional_trends']}}}",
                r"\caption{Institutional sector proportions evaluated annually over the selected temporal window.}",
                r"\end{figure}",
                r"",
            ]
        )

    if "fig_geopolitical_correlation" in image_filenames:
        tex_parts.extend(
            [
                r"\section{Geopolitical Perimeter \& Cross-Sector Correlation}",
                r"Analysis of research origins categorizing papers into sovereign EU/EEC member state outputs, international mixed consortia, and non-EU/EEC cohorts.",
                r"",
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_geopolitical_correlation']}}}",
                r"\caption{Geopolitical scope cross-tabulated with institutional sector (Academia vs. Industry).}",
                r"\end{figure}",
                r"",
            ]
        )

    if "fig_keyword_prevalence" in image_filenames and parsed_kws:
        tex_parts.extend(
            [
                r"\section{Target Keyword Frequency Analysis}",
                r"Prevalence and occurrence frequency of user-specified terms quantified across sectoral affiliations.",
                r"",
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_keyword_prevalence']}}}",
                r"\caption{Target keyword paper incidence grouped by institutional classification.}",
                r"\end{figure}",
                r"",
            ]
        )

    # Citation Network & Centrality Hub Analysis
    if "fig_citation_network" in image_filenames:
        tex_parts.extend(
            [
                r"\newpage",
                r"\section{Internal Citation Network \& Central Hub Analysis}",
                r"Internal cross-citation dynamics restricted exclusively to citations connecting papers within the study cohort.",
                r"",
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_citation_network']}}}",
                r"\caption{Internal citation network graph. Node sizes scale with internal in-degree centrality; node symbols distinguish institutional sectors.}",
                r"\end{figure}",
                r"",
            ]
        )

        if "in_degree" in analyzed_df.columns and not analyzed_df.empty:
            hub_papers = analyzed_df.sort_values(by=["in_degree", "betweenness_centrality"], ascending=False).head(10)
            hub_rows: list[str] = []
            for rank, (_, hrow) in enumerate(hub_papers.iterrows(), start=1):
                raw_title = str(hrow.get("title", "Untitled"))
                short_title = (raw_title[:65] + "...") if len(raw_title) > 65 else raw_title
                esc_title = escape_latex(short_title)
                sec = escape_latex(str(hrow.get("category", "Unknown")))
                in_c = int(hrow.get("in_degree", 0))
                betw = float(hrow.get("betweenness_centrality", 0.0))
                hub_rows.append(f"{rank} & {esc_title} & {sec} & {in_c} & {betw:.4f} \\\\")

            table_hub_str = "\n".join(hub_rows)
            tex_parts.extend(
                [
                    r"\begin{table}[H]",
                    r"\centering",
                    r"\small",
                    r"\begin{tabularx}{\textwidth}{rXlrr}",
                    r"\toprule",
                    r"\textbf{\#} & \textbf{Publication Title} & \textbf{Sector} & \textbf{In-Citations} & \textbf{Betweenness} \\",
                    r"\midrule",
                    table_hub_str,
                    r"\bottomrule",
                    r"\end{tabularx}",
                    r"\caption{Top hub publications ranked by internal citation in-degree within the analyzed cohort.}",
                    r"\end{table}",
                    r"",
                ]
            )

    # Descriptive Productivity Metrics
    if "fig_top_authors" in image_filenames or "fig_top_institutions" in image_filenames:
        tex_parts.extend(
            [
                r"\section{Descriptive Productivity Metrics}",
                r"Institutional and author output frequencies evaluated across the retrieved publication cohort.",
                r"",
            ]
        )
        if "fig_top_authors" in image_filenames:
            tex_parts.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_top_authors']}}}",
                    r"\caption{Top 20 most frequent authors contributing to the publication corpus.}",
                    r"\end{figure}",
                    r"",
                ]
            )
        if "fig_top_institutions" in image_filenames:
            tex_parts.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.85\linewidth]{{{image_filenames['fig_top_institutions']}}}",
                    r"\caption{Top 20 most frequent research institutions associated with the publication corpus.}",
                    r"\end{figure}",
                    r"",
                ]
            )

    tex_parts.extend(
        [
            r"\section{Methodological Framework \& FAIR Compliance}",
            r"\begin{itemize}",
            r"\item \textbf{Defensive Data Ingestion:} Scopus metadata fields (\texttt{affilname}, \texttt{affiliation-country}) are split and aligned with boundary verification, mitigating index exceptions.",
            r"\item \textbf{Internal Citation Resolution:} Reference lists are queried via the Scopus Abstract Retrieval API and filtered against the cohort lookup index to isolate cross-citations exclusively within the search results.",
            r"\item \textbf{Geopolitical Criteria:} Evaluated against the 27 EU member states plus 3 EEC/EEA EFTA members (Iceland, Liechtenstein, Norway).",
            r"\item \textbf{Monochrome Graphics:} Figures are generated using black outlines, grayscale fills, and pattern hatching (\texttt{pattern\_shape}) to guarantee legibility under standard monochrome printing.",
            r"\item \textbf{Provenance Stamp:} Exported report bundle includes complete metadata parameters supporting reproducibility.",
            r"\end{itemize}",
            r"",
            r"\end{document}",
        ]
    )

    return "\n".join(tex_parts)


# ---------------------------------------------------------------------------
# ZIP Bundle Packaging
# ---------------------------------------------------------------------------

def create_latex_bundle(
    analyzed_df: pd.DataFrame,
    parsed_kws: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    citation_graph: Any = None,
) -> bytes:
    """Render figures, build report.tex, and compile a downloadable ZIP archive.

    Parameters
    ----------
    analyzed_df:
        Classified publication DataFrame.
    parsed_kws:
        Target keyword list.
    metadata:
        Query and filter metadata.
    citation_graph:
        Optional precomputed NetworkX citation graph.

    Returns
    -------
    Bytes content of the generated ZIP archive.
    """
    if metadata is None:
        metadata = {}

    figures = build_monochrome_figures(
        analyzed_df=analyzed_df,
        parsed_kws=parsed_kws,
        citation_graph=citation_graph,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        image_filenames: dict[str, str] = {}

        # Export static PNG images via kaleido
        for key, fig in figures.items():
            fname = f"{key}.png"
            fpath = f"{tmpdir}/{fname}"
            fig.write_image(fpath, format="png", width=1000, height=520, scale=2)
            image_filenames[key] = fname

        # Synthesize LaTeX document
        latex_text = generate_latex_document(
            analyzed_df=analyzed_df,
            parsed_kws=parsed_kws,
            metadata=metadata,
            image_filenames=image_filenames,
        )

        tex_path = f"{tmpdir}/report.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_text)

        # Assemble in-memory ZIP archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(tex_path, arcname="report.tex")
            for fname in image_filenames.values():
                zip_file.write(f"{tmpdir}/{fname}", arcname=fname)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()
