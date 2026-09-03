"""
ris_exporter.py — Standard RIS bibliographic reference export using rispy.

Converts Scopus publication metadata into RFC-compliant RIS bibliographic
records for citation managers (Zotero, Mendeley, EndNote).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
import rispy

from scopus_client import search_scopus

logger = logging.getLogger(__name__)


def row_to_ris_entry(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Convert a single publication record into a rispy-compliant dictionary.

    Parameters
    ----------
    row:
        Dictionary or pandas Series with keys: 'title', 'year', 'doi',
        'abstract', 'eid', 'affiliations', 'category'.

    Returns
    -------
    Dictionary structured with canonical rispy tag keys.
    """
    entry: dict[str, Any] = {
        "type_of_reference": "JOUR",
    }

    title = row.get("title")
    if title and isinstance(title, str) and title.strip():
        entry["primary_title"] = title.strip()

    year = row.get("year")
    if year is not None and not pd.isna(year):
        entry["publication_year"] = str(int(year))

    doi = row.get("doi")
    if doi and isinstance(doi, str) and doi.strip():
        clean_doi = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if clean_doi.lower().startswith(prefix):
                clean_doi = clean_doi[len(prefix):]
                break
        entry["doi"] = clean_doi

    abstract = row.get("abstract")
    if abstract and isinstance(abstract, str) and abstract.strip():
        entry["abstract"] = abstract.strip()

    eid = row.get("eid")
    if eid and isinstance(eid, str) and eid.strip():
        entry["accession_number"] = eid.strip()
        entry["urls"] = [f"https://www.scopus.com/record/display.uri?eid={eid.strip()}&origin=resultslist"]

    category = row.get("category")
    if category and isinstance(category, str):
        entry["custom1"] = f"Affiliation: {category}"

    geo_category = row.get("geo_category")
    if geo_category and isinstance(geo_category, str):
        entry["custom2"] = f"Geopolitical: {geo_category}"

    note_parts: list[str] = []
    affils = row.get("affiliations") or row.get("institutions")
    if affils:
        if isinstance(affils, list):
            note_parts.append("Institutions: " + "; ".join(str(a) for a in affils if a))
        elif isinstance(affils, str):
            note_parts.append("Institutions: " + affils)

    countries = row.get("countries")
    if countries:
        if isinstance(countries, list):
            note_parts.append("Countries: " + "; ".join(str(c) for c in countries if c))
        elif isinstance(countries, str):
            note_parts.append("Countries: " + countries)

    if note_parts:
        entry["notes"] = [" | ".join(note_parts)]

    return entry


def dataframe_to_ris_entries(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame of Scopus publications into a list of RIS entries.

    Parameters
    ----------
    df:
        DataFrame containing publication metadata.

    Returns
    -------
    List of entry dictionaries ready for rispy.dumps.
    """
    if df.empty:
        return []

    entries: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        entries.append(row_to_ris_entry(row))
    return entries


def export_ris_string(entries: list[dict[str, Any]]) -> str:
    """Serialize a list of reference entries into standard RIS text format.

    Parameters
    ----------
    entries:
        List of reference dictionaries.

    Returns
    -------
    String containing valid RIS records.
    """
    if not entries:
        return ""
    return rispy.dumps(entries)


def process_query_batch(
    queries: list[str],
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[str, int, int, list[dict[str, Any]]]:
    """Execute a batch of Scopus queries and compile all results into a unified RIS string.

    Parameters
    ----------
    queries:
        List of Scopus search query strings.
    progress_callback:
        Optional callback accepting (fraction: float, message: str).

    Returns
    -------
    tuple of:
        - ris_text: str (the complete combined RIS string)
        - queries_executed: int
        - references_compiled: int
        - query_summaries: list of summary dicts per query
    """
    cleaned_queries = [q.strip() for q in queries if q and q.strip()]
    if not cleaned_queries:
        return "", 0, 0, []

    all_entries: list[dict[str, Any]] = []
    seen_eids: set[str] = set()
    query_summaries: list[dict[str, Any]] = []
    total = len(cleaned_queries)

    for idx, q in enumerate(cleaned_queries):
        if progress_callback:
            progress_callback(
                (idx + 1) / total,
                f"Processing query {idx + 1} of {total}: {q[:45]}...",
            )

        try:
            df = search_scopus(q)
            count = len(df)
            added = 0
            if not df.empty:
                for _, row in df.iterrows():
                    eid = row.get("eid")
                    # Deduplicate across queries if same paper is returned
                    if eid and eid in seen_eids:
                        continue
                    if eid:
                        seen_eids.add(eid)
                    all_entries.append(row_to_ris_entry(row))
                    added += 1

            query_summaries.append(
                {
                    "query": q,
                    "status": "Success",
                    "papers_found": count,
                    "unique_added": added,
                }
            )
        except Exception as exc:
            logger.error("Error executing query '%s': %s", q, exc)
            query_summaries.append(
                {
                    "query": q,
                    "status": f"Error: {exc.__class__.__name__}",
                    "papers_found": 0,
                    "unique_added": 0,
                }
            )

    ris_text = export_ris_string(all_entries)
    return ris_text, len(cleaned_queries), len(all_entries), query_summaries
