"""
scopus_client.py — Scopus API data-fetching and pagination layer.

Executes Scopus searches with chunk-level pagination, real-time progress callbacks,
and dynamic Estimated Time of Arrival (ETA) calculation.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import configparser
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing & ETA Formatting Helper
# ---------------------------------------------------------------------------

def format_eta(seconds: float | None) -> str:
    """Format duration in seconds to MM:SS string (or HH:MM:SS if >= 1 hour).

    Parameters
    ----------
    seconds:
        Projected remaining duration in seconds.

    Returns
    -------
    Formatted time string (e.g. '02:15', '05:30', '--:--').
    """
    if seconds is None or seconds < 0:
        return "--:--"
    total_sec = int(round(seconds))
    mins, secs = divmod(total_sec, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# pybliometrics configuration bootstrap
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".config"
_CONFIG_PATH = _CONFIG_DIR / "pybliometrics.cfg"


def init_pybliometrics(api_key: str, inst_token: str | None = None) -> None:
    """Write (or update) the pybliometrics config file with *api_key*.

    Parameters
    ----------
    api_key:
        Elsevier / Scopus API key.
    inst_token:
        Optional Elsevier Institutional Token.
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    clean_key = api_key.strip()
    clean_token = inst_token.strip() if inst_token and inst_token.strip() else None

    # Use pybliometrics built-in create_config if file does not exist
    try:
        from pybliometrics.utils.create_config import create_config
        tokens = [clean_token] if clean_token else None
        create_config(config_dir=_CONFIG_PATH, keys=[clean_key], insttoken=tokens)
    except Exception:
        config = configparser.ConfigParser(strict=False)
        config.optionxform = str
        if _CONFIG_PATH.exists():
            try:
                config.read(str(_CONFIG_PATH))
            except Exception:
                pass

        if "Authentication" not in config:
            config["Authentication"] = {}
        config["Authentication"]["APIKey"] = clean_key
        if clean_token:
            config["Authentication"]["InstToken"] = clean_token

        if "Directories" not in config:
            from pybliometrics.utils.constants import DEFAULT_PATHS
            config["Directories"] = {k: str(v) for k, v in DEFAULT_PATHS.items()}

        if "Requests" not in config:
            config["Requests"] = {"Timeout": "20", "Retries": "5"}

        with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
            config.write(fh)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_year(cover_date: Any) -> int | None:
    """Extract a 4-digit publication year from a Scopus coverDate string."""
    if not cover_date or not isinstance(cover_date, str):
        return None
    token = cover_date.strip()[:4]
    return int(token) if token.isdigit() and len(token) == 4 else None


def _parse_affiliations(raw: Any) -> list[str]:
    """Split a semicolon-delimited Scopus affilname string into unique names."""
    if not raw or not isinstance(raw, str):
        return []
    names = [part.strip() for part in raw.split(";") if part.strip()]
    return list(dict.fromkeys(names))


def _parse_institutions_and_countries(
    affilname: Any, country_str: Any
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Parse and align semicolon-delimited institutions and countries."""
    raw_insts = (
        [p.strip() for p in affilname.split(";") if p.strip()]
        if isinstance(affilname, str)
        else []
    )
    raw_countries = (
        [p.strip() for p in country_str.split(";") if p.strip()]
        if isinstance(country_str, str)
        else []
    )

    unique_insts = list(dict.fromkeys(raw_insts))
    unique_countries = list(dict.fromkeys(raw_countries))

    details: list[dict[str, str]] = []
    max_len = max(len(raw_insts), len(raw_countries))
    for idx in range(max_len):
        inst_val = raw_insts[idx] if idx < len(raw_insts) else "Unknown Institution"
        country_val = raw_countries[idx] if idx < len(raw_countries) else "Unknown Country"
        details.append({"institution": inst_val, "country": country_val})

    return unique_insts, unique_countries, details


def _parse_authors(author_names: Any, creator: Any = None) -> list[str]:
    """Extract a list of author names from Scopus metadata."""
    if author_names and isinstance(author_names, str):
        return [a.strip() for a in author_names.split(";") if a.strip()]
    if creator and isinstance(creator, str) and creator.strip():
        return [creator.strip()]
    return []


def _parse_entry_to_row(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Scopus JSON search entry into a structured row dictionary."""
    affil_field = entry.get("affiliation")
    insts: list[str] = []
    countries: list[str] = []
    details: list[dict[str, str]] = []

    if isinstance(affil_field, list):
        for af in affil_field:
            if isinstance(af, dict):
                inst = af.get("affilname", "Unknown Institution")
                cntry = af.get("affiliation-country", "Unknown Country")
                if inst:
                    insts.append(inst)
                if cntry:
                    countries.append(cntry)
                details.append({
                    "institution": inst or "Unknown Institution",
                    "country": cntry or "Unknown Country",
                })
    elif isinstance(affil_field, dict):
        inst = affil_field.get("affilname", "Unknown Institution")
        cntry = affil_field.get("affiliation-country", "Unknown Country")
        if inst:
            insts.append(inst)
        if cntry:
            countries.append(cntry)
        details.append({
            "institution": inst or "Unknown Institution",
            "country": cntry or "Unknown Country",
        })

    authors_list: list[str] = []
    auth_field = entry.get("author")
    if isinstance(auth_field, list):
        for a in auth_field:
            if isinstance(a, dict):
                surname = a.get("surname")
                given = a.get("given-name")
                authname = a.get("authname")
                if surname and given:
                    authors_list.append(f"{surname}, {given}")
                elif authname:
                    authors_list.append(authname)
    elif isinstance(auth_field, dict):
        surname = auth_field.get("surname")
        given = auth_field.get("given-name")
        authname = auth_field.get("authname")
        if surname and given:
            authors_list.append(f"{surname}, {given}")
        elif authname:
            authors_list.append(authname)

    if not authors_list and entry.get("dc:creator"):
        authors_list = [str(entry.get("dc:creator"))]

    return {
        "eid": entry.get("eid") or entry.get("dc:identifier"),
        "title": entry.get("dc:title"),
        "year": _parse_year(entry.get("prism:coverDate")),
        "affiliations": list(dict.fromkeys(insts)),
        "doi": entry.get("prism:doi"),
        "abstract": entry.get("dc:description"),
        "institutions": list(dict.fromkeys(insts)),
        "countries": list(dict.fromkeys(countries)),
        "affiliations_detail": details,
        "authors": authors_list,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_RESULT_COLUMNS = [
    "eid",
    "title",
    "year",
    "affiliations",
    "doi",
    "abstract",
    "institutions",
    "countries",
    "affiliations_detail",
    "authors",
]


def search_scopus(
    query: str,
    count: int = 25,
    api_key: str | None = None,
    inst_token: str | None = None,
    progress_callback: Any = None,
) -> pd.DataFrame:
    """Execute a Scopus search with pagination, progress reporting, and ETA tracking.

    Parameters
    ----------
    query:
        Scopus advanced search string.
    count:
        Number of items per pagination request chunk (default: 25).
    api_key:
        Optional API key (falls back to SCOPUS_API_KEY environment variable).
    inst_token:
        Optional Elsevier Institutional Token.
    progress_callback:
        Optional callback accepting (chunk_idx: int, total_chunks: int,
        retrieved: int, total_available: int, message: str).

    Returns
    -------
    DataFrame conforming to _RESULT_COLUMNS.
    """
    resolved_key = (api_key or os.getenv("SCOPUS_API_KEY", "")).strip()

    # Attempt paginated REST requests if an API key is available
    if resolved_key:
        headers = {
            "X-ELS-APIKey": resolved_key,
            "Accept": "application/json",
        }
        if inst_token:
            headers["X-ELS-Insttoken"] = inst_token.strip()

        base_url = "https://api.elsevier.com/content/search/scopus"
        start_time = time.time()

        try:
            params = {"query": query, "start": 0, "count": count, "view": "COMPLETE"}
            resp = requests.get(base_url, headers=headers, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                sr = data.get("search-results", {})
                total_results = int(sr.get("opensearch:totalResults", 0))

                if total_results == 0:
                    if progress_callback:
                        progress_callback(1, 1, 0, 0, "No records found | ETA: 00:00")
                    return pd.DataFrame(columns=_RESULT_COLUMNS)

                total_chunks = max(1, math.ceil(total_results / count))
                all_rows: list[dict[str, Any]] = []

                # Process chunk 1
                entries = sr.get("entry", [])
                for e in entries:
                    all_rows.append(_parse_entry_to_row(e))

                elapsed = time.time() - start_time
                avg_time = elapsed / 1
                remaining_chunks = total_chunks - 1
                eta_sec = avg_time * remaining_chunks
                eta_str = format_eta(eta_sec)

                if progress_callback:
                    msg = (
                        f"Fetching chunk 1 of {total_chunks}... "
                        f"({len(all_rows)} / {total_results:,} papers) | ETA: {eta_str}"
                    )
                    progress_callback(1, total_chunks, len(all_rows), total_results, msg)

                # Fetch remaining chunks
                for chunk_idx in range(2, total_chunks + 1):
                    start = (chunk_idx - 1) * count
                    p = {"query": query, "start": start, "count": count, "view": "COMPLETE"}
                    chunk_resp = requests.get(base_url, headers=headers, params=p, timeout=20)

                    if chunk_resp.status_code == 200:
                        c_data = chunk_resp.json()
                        c_entries = c_data.get("search-results", {}).get("entry", [])
                        for e in c_entries:
                            all_rows.append(_parse_entry_to_row(e))

                    elapsed = time.time() - start_time
                    avg_time = elapsed / chunk_idx
                    rem_chunks = total_chunks - chunk_idx
                    eta_sec = avg_time * rem_chunks
                    eta_str = format_eta(eta_sec)

                    if progress_callback:
                        msg = (
                            f"Fetching chunk {chunk_idx} of {total_chunks}... "
                            f"({len(all_rows)} / {total_results:,} papers) | ETA: {eta_str}"
                        )
                        progress_callback(chunk_idx, total_chunks, len(all_rows), total_results, msg)

                return pd.DataFrame(all_rows, columns=_RESULT_COLUMNS)

        except Exception as exc:
            logger.warning("Paginated REST search exception: %s. Falling back to pybliometrics.", exc)

    # Fallback to pybliometrics.scopus.ScopusSearch
    try:
        from pybliometrics.scopus import ScopusSearch
    except Exception as exc:
        raise RuntimeError(f"Failed to import pybliometrics: {exc}") from exc

    try:
        search = ScopusSearch(query, refresh=True)
    except Exception as exc:
        raise RuntimeError(f"Scopus API error: {exc}") from exc

    results = search.results
    if not results:
        if progress_callback:
            progress_callback(1, 1, 0, 0, "No records found | ETA: 00:00")
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    total_results = len(results)
    if progress_callback:
        progress_callback(
            1,
            1,
            total_results,
            total_results,
            f"Retrieved {total_results:,} papers | ETA: 00:00",
        )

    rows: list[dict[str, Any]] = []
    for doc in results:
        raw_affil = getattr(doc, "affilname", None)
        raw_country = getattr(doc, "affiliation_country", None)
        insts, countries, details = _parse_institutions_and_countries(raw_affil, raw_country)
        authors = _parse_authors(
            getattr(doc, "author_names", None),
            getattr(doc, "creator", None),
        )

        rows.append(
            {
                "eid": getattr(doc, "eid", None),
                "title": getattr(doc, "title", None),
                "year": _parse_year(getattr(doc, "coverDate", None)),
                "affiliations": insts,
                "doi": getattr(doc, "doi", None),
                "abstract": getattr(doc, "description", None),
                "institutions": insts,
                "countries": countries,
                "affiliations_detail": details,
                "authors": authors,
            }
        )

    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)
