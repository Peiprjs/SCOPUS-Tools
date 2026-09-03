"""
network_analyzer.py — Internal Citation Network & Bibliometric Metrics Engine.

Constructs internal cross-citation graphs, computes graph theory metrics
(In-Degree, Out-Degree, Betweenness Centrality, PageRank), and builds
interactive & monochrome Plotly network maps and descriptive frequency charts.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation Reference Extraction & Internal Edge Resolution
# ---------------------------------------------------------------------------

def extract_citation_edges(
    df: pd.DataFrame,
    api_key: str | None = None,
    inst_token: str | None = None,
    progress_callback: Any = None,
    sleep_between_calls: float = 0.05,
) -> list[tuple[str, str]]:
    """Harvest paper references and extract internal cross-citations.

    Identifies directed edges (citing_eid -> cited_eid) exclusively where
    both the citing and cited publications exist within the current dataset.

    Parameters
    ----------
    df:
        DataFrame containing 'eid' and optional 'doi' columns.
    api_key:
        Elsevier / Scopus API key.
    inst_token:
        Optional Elsevier Institutional Token.
    progress_callback:
        Optional callback accepting (fraction: float, message: str).
    sleep_between_calls:
        Throttling delay between network queries to respect rate limits.

    Returns
    -------
    List of unique directed edge tuples: [(citing_eid, cited_eid), ...].
    """
    if df.empty or "eid" not in df.columns:
        return []

    # Import pybliometrics locally
    try:
        from pybliometrics.scopus import AbstractRetrieval
    except Exception as exc:
        logger.warning("Could not import pybliometrics.scopus.AbstractRetrieval: %s", exc)
        return []

    # Construct multi-key lookup dictionary for the active dataset
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        eid = row.get("eid")
        if eid and isinstance(eid, str) and eid.strip():
            clean_eid = eid.strip()
            lookup[clean_eid] = clean_eid
            # Map numeric Scopus ID (e.g. 2-s2.0-851234 -> 851234)
            if "-" in clean_eid:
                numeric_id = clean_eid.split("-")[-1]
                if numeric_id:
                    lookup[numeric_id] = clean_eid

        doi = row.get("doi")
        if doi and isinstance(doi, str) and doi.strip():
            clean_doi = doi.strip().lower()
            lookup[clean_doi] = str(eid).strip()
            # Also map normalized un-prefixed DOI
            for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
                if clean_doi.startswith(prefix):
                    clean_doi = clean_doi[len(prefix):]
                    lookup[clean_doi] = str(eid).strip()
                    break

    edges: list[tuple[str, str]] = []
    total_papers = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        citing_eid = row.get("eid")
        if not citing_eid or not isinstance(citing_eid, str) or not citing_eid.strip():
            continue
        clean_citing = citing_eid.strip()

        if progress_callback:
            progress_callback(
                (idx + 1) / total_papers,
                f"Resolving citation references: {idx + 1} of {total_papers}",
            )

        try:
            ab = AbstractRetrieval(clean_citing, view="REF", refresh=False)
            refs = getattr(ab, "references", None) or []

            for r in refs:
                target_eid: str | None = None
                r_id = getattr(r, "id", None)
                r_doi = getattr(r, "doi", None)

                # 1. Match by Scopus ID
                if r_id is not None:
                    str_id = str(r_id).strip()
                    if str_id in lookup:
                        target_eid = lookup[str_id]
                    elif f"2-s2.0-{str_id}" in lookup:
                        target_eid = lookup[f"2-s2.0-{str_id}"]

                # 2. Match by DOI if ID did not resolve
                if not target_eid and r_doi:
                    clean_r_doi = str(r_doi).strip().lower()
                    if clean_r_doi in lookup:
                        target_eid = lookup[clean_r_doi]
                    else:
                        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
                            if clean_r_doi.startswith(prefix):
                                clean_r_doi = clean_r_doi[len(prefix):]
                                if clean_r_doi in lookup:
                                    target_eid = lookup[clean_r_doi]
                                break

                # Add directed edge if target belongs to cohort and is not a self-citation
                if target_eid and target_eid != clean_citing:
                    edges.append((clean_citing, target_eid))

        except Exception as exc:
            logger.debug("Reference retrieval exception for %s: %s", clean_citing, exc)

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    # Return deduplicated, deterministic edge list
    return sorted(list(set(edges)))


# ---------------------------------------------------------------------------
# Graph Construction & Graph Theory Metrics
# ---------------------------------------------------------------------------

def build_citation_graph(
    df: pd.DataFrame,
    edges: list[tuple[str, str]],
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Construct directed citation graph and compute graph centrality metrics.

    Parameters
    ----------
    df:
        DataFrame containing publication metadata.
    edges:
        List of directed (citing_eid, cited_eid) pairs.

    Returns
    -------
    tuple of:
        - nx.DiGraph instance
        - DataFrame enriched with 'in_degree', 'out_degree', 'betweenness_centrality', 'pagerank'
    """
    G = nx.DiGraph()

    if df.empty or "eid" not in df.columns:
        return G, df.copy()

    # Add all papers as nodes to preserve isolated publications (0 edges)
    for _, row in df.iterrows():
        eid = str(row.get("eid", "")).strip()
        if eid:
            G.add_node(
                eid,
                title=str(row.get("title", "Untitled")),
                year=row.get("year"),
                category=str(row.get("category", "Unknown")),
                geo_category=str(row.get("geo_category", "Unknown Geo")),
            )

    # Add internal cross-citation edges
    for u, v in edges:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)

    # Compute Centrality & Graph Metrics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}

    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    # Attach computed metrics to DataFrame
    df_out = df.copy()
    df_out["in_degree"] = df_out["eid"].map(in_degrees).fillna(0).astype(int)
    df_out["out_degree"] = df_out["eid"].map(out_degrees).fillna(0).astype(int)
    df_out["betweenness_centrality"] = df_out["eid"].map(betweenness).fillna(0.0).round(4)
    df_out["pagerank"] = df_out["eid"].map(pagerank).fillna(0.0).round(4)

    return G, df_out


# ---------------------------------------------------------------------------
# Plotly Network Visualization
# ---------------------------------------------------------------------------

def build_network_plotly_figure(
    G: nx.DiGraph,
    df: pd.DataFrame,
    metric: str = "in_degree",
    monochrome: bool = False,
) -> go.Figure:
    """Render a publication-ready Plotly network graph visualization.

    Parameters
    ----------
    G:
        NetworkX directed graph.
    df:
        DataFrame containing node metadata and computed metrics.
    metric:
        Centrality metric used to scale node sizes ('in_degree' or 'betweenness_centrality').
    monochrome:
        If True, renders in grayscale using pattern symbols and simple_white template.

    Returns
    -------
    Plotly Figure instance.
    """
    if len(G) == 0:
        fig_empty = go.Figure()
        fig_empty.update_layout(
            template="simple_white",
            title="Citation Network (No Data Available)",
        )
        return fig_empty

    # Calculate 2D coordinates using spring layout
    pos = nx.spring_layout(G, k=0.35, seed=42, iterations=50)

    # 1. Construct Edge Traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="#888888" if not monochrome else "#555555"),
        hoverinfo="none",
        mode="lines",
        name="Citations",
    )

    # 2. Construct Node Traces
    # Build lookup from DataFrame for rich hover data
    df_lookup = df.set_index("eid").to_dict(orient="index") if not df.empty and "eid" in df.columns else {}

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_sizes: list[float] = []
    node_categories: list[str] = []

    # Determine metric scaling bounds
    metric_vals = [
        float(df_lookup.get(n, {}).get(metric, 0.0))
        for n in G.nodes()
    ]
    max_metric = max(metric_vals) if metric_vals and max(metric_vals) > 0 else 1.0

    color_map = {
        "Academia": "#1f77b4",
        "Industry": "#d62728",
        "Mixed": "#9467bd",
        "Unknown": "#7f7f7f",
    }
    symbol_map = {
        "Academia": "circle",
        "Industry": "square",
        "Mixed": "diamond",
        "Unknown": "triangle-up",
    }

    for n in G.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        info = df_lookup.get(n, {})
        title = info.get("title", "Untitled")
        year = info.get("year", "N/A")
        cat = info.get("category", "Unknown")
        geo = info.get("geo_category", "Unknown Geo")
        in_deg = info.get("in_degree", 0)
        out_deg = info.get("out_degree", 0)
        betw = info.get("betweenness_centrality", 0.0)
        p_rank = info.get("pagerank", 0.0)

        # Scale node size: minimum 10px, maximum 40px
        curr_val = float(info.get(metric, 0.0))
        scaled_size = 10.0 + 30.0 * (curr_val / max_metric)
        node_sizes.append(scaled_size)
        node_categories.append(cat)

        short_title = (title[:70] + "...") if len(title) > 70 else title
        hover_str = (
            f"<b>{short_title}</b><br>"
            f"Year: {year} | Sector: {cat}<br>"
            f"Geopolitical Scope: {geo}<br>"
            f"Internal In-Citations: {in_deg} | References to Cohort: {out_deg}<br>"
            f"Betweenness Centrality: {betw:.4f} | PageRank: {p_rank:.4f}"
        )
        node_text.append(hover_str)

    fig = go.Figure()
    fig.add_trace(edge_trace)

    if monochrome:
        # Single monochrome trace using symbols to distinguish categories
        node_symbols = [symbol_map.get(c, "circle") for c in node_categories]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=node_sizes,
                color="#f0f0f0",
                symbol=node_symbols,
                line=dict(width=1.5, color="#000000"),
            ),
            name="Publications",
        )
        fig.add_trace(node_trace)
    else:
        # Split by category to provide an interactive legend
        for cat in ["Academia", "Industry", "Mixed", "Unknown"]:
            sub_indices = [i for i, c in enumerate(node_categories) if c == cat]
            if not sub_indices:
                continue
            cat_trace = go.Scatter(
                x=[node_x[i] for i in sub_indices],
                y=[node_y[i] for i in sub_indices],
                mode="markers",
                hoverinfo="text",
                text=[node_text[i] for i in sub_indices],
                marker=dict(
                    size=[node_sizes[i] for i in sub_indices],
                    color=color_map.get(cat, "#7f7f7f"),
                    symbol=symbol_map.get(cat, "circle"),
                    line=dict(width=1.2, color="#333333"),
                ),
                name=cat,
            )
            fig.add_trace(cat_trace)

    fig.update_layout(
        template="simple_white",
        height=550,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#000000"),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Descriptive Frequency Visualizations (Top Authors & Institutions)
# ---------------------------------------------------------------------------

def build_descriptive_frequency_figures(
    df: pd.DataFrame,
    top_n: int = 20,
    monochrome: bool = False,
) -> tuple[go.Figure, go.Figure]:
    """Generate horizontal bar charts for Top Authors and Top Institutions.

    Parameters
    ----------
    df:
        DataFrame containing 'authors' and 'institutions' lists.
    top_n:
        Maximum number of entities to display (default: 20).
    monochrome:
        If True, applies simple_white template and pattern shapes.

    Returns
    -------
    tuple of (fig_top_authors, fig_top_institutions).
    """
    # 1. Top Authors
    author_counts: Counter[str] = Counter()
    if "authors" in df.columns:
        for auth_list in df["authors"].dropna():
            if isinstance(auth_list, list):
                for a in auth_list:
                    if a and str(a).strip():
                        author_counts[str(a).strip()] += 1

    top_authors = author_counts.most_common(top_n)
    if top_authors:
        # Reverse to show highest on top in horizontal bar chart
        authors_df = pd.DataFrame(reversed(top_authors), columns=["Author", "Publications"])
    else:
        authors_df = pd.DataFrame({"Author": ["No Author Data"], "Publications": [0]})

    if monochrome:
        fig_authors = px.bar(
            authors_df,
            x="Publications",
            y="Author",
            orientation="h",
            template="simple_white",
            labels={"Publications": "Publication Count", "Author": "Researcher"},
        )
        fig_authors.update_traces(
            marker=dict(
                color="#f0f0f0",
                line=dict(width=1.5, color="#000000"),
                pattern_shape="/",
            ),
            textposition="outside",
        )
    else:
        fig_authors = px.bar(
            authors_df,
            x="Publications",
            y="Author",
            orientation="h",
            color="Publications",
            color_continuous_scale="Blues",
            labels={"Publications": "Publication Count", "Author": "Researcher"},
        )
        fig_authors.update_layout(coloraxis_showscale=False)
        fig_authors.update_traces(textposition="outside")

    fig_authors.update_layout(
        height=max(380, len(authors_df) * 22),
        margin=dict(l=150, r=40, t=30, b=40),
        xaxis=dict(showline=True, linecolor="#000000", dtick=1),
        yaxis=dict(showline=True, linecolor="#000000"),
    )

    # 2. Top Institutions
    inst_counts: Counter[str] = Counter()
    inst_col = "institutions" if "institutions" in df.columns else "affiliations"
    if inst_col in df.columns:
        for inst_list in df[inst_col].dropna():
            if isinstance(inst_list, list):
                for i in inst_list:
                    if i and str(i).strip() and str(i).strip() != "Unknown Institution":
                        inst_counts[str(i).strip()] += 1

    top_insts = inst_counts.most_common(top_n)
    if top_insts:
        insts_df = pd.DataFrame(reversed(top_insts), columns=["Institution", "Publications"])
    else:
        insts_df = pd.DataFrame({"Institution": ["No Institution Data"], "Publications": [0]})

    if monochrome:
        fig_insts = px.bar(
            insts_df,
            x="Publications",
            y="Institution",
            orientation="h",
            template="simple_white",
            labels={"Publications": "Publication Count", "Institution": "Research Institution"},
        )
        fig_insts.update_traces(
            marker=dict(
                color="#e5e5e5",
                line=dict(width=1.5, color="#000000"),
                pattern_shape="x",
            ),
            textposition="outside",
        )
    else:
        fig_insts = px.bar(
            insts_df,
            x="Publications",
            y="Institution",
            orientation="h",
            color="Publications",
            color_continuous_scale="Teal",
            labels={"Publications": "Publication Count", "Institution": "Research Institution"},
        )
        fig_insts.update_layout(coloraxis_showscale=False)
        fig_insts.update_traces(textposition="outside")

    fig_insts.update_layout(
        height=max(380, len(insts_df) * 22),
        margin=dict(l=220, r=40, t=30, b=40),
        xaxis=dict(showline=True, linecolor="#000000", dtick=1),
        yaxis=dict(showline=True, linecolor="#000000"),
    )

    return fig_authors, fig_insts
