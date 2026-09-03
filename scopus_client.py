"""
scopus_client.py — Scopus API data-fetching layer.

Wraps pybliometrics to execute ScopusSearch queries and return clean
DataFrames with extracted year and affiliation lists.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


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

    # Initialize pybliometrics in-memory config
    try:
        import pybliometrics
        tokens = [clean_token] if clean_token else None
        pybliometrics.init(keys=[clean_key], inst_tokens=tokens)
    except Exception as exc:
        logger.debug("pybliometrics in-memory init notice: %s", exc)

    logger.info("pybliometrics config initialized for %s", _CONFIG_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_year(cover_date: Any) -> int | None:
    """Extract a 4-digit year from a coverDate string like '2023-06-15'."""
    if not cover_date or not isinstance(cover_date, str):
        return None
    try:
        return int(cover_date[:4])
    except (ValueError, IndexError):
        return None


def _parse_affiliations(affilname: Any) -> list[str]:
    """Split a semicolon-delimited affiliation string into a clean list."""
    if not affilname or not isinstance(affilname, str):
        return []
    return [a.strip() for a in affilname.split(";") if a.strip()]


def _parse_institutions_and_countries(
    affilname: Any,
    country_str: Any,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Defensively parse and pair institutions and countries from Scopus metadata.

    Guarantees zero KeyError or IndexError exceptions even when delimiter
    counts differ or fields are null/malformed.

    Parameters
    ----------
    affilname:
        Semicolon-delimited institutional names or None.
    country_str:
        Semicolon-delimited country names or None.

    Returns
    -------
    tuple of:
        - institutions: deduplicated list of institution names
        - countries: deduplicated list of country names
        - affiliations_detail: list of paired dicts {"institution": ..., "country": ...}
    """
    raw_insts: list[str] = []
    if affilname and isinstance(affilname, str):
        raw_insts = [i.strip() for i in affilname.split(";") if i.strip()]

    raw_countries: list[str] = []
    if country_str and isinstance(country_str, str):
        raw_countries = [c.strip() for c in country_str.split(";") if c.strip()]

    # Deduplicated lists preserving order
    unique_insts: list[str] = []
    for inst in raw_insts:
        if inst not in unique_insts:
            unique_insts.append(inst)

    unique_countries: list[str] = []
    for cntry in raw_countries:
        if cntry not in unique_countries:
            unique_countries.append(cntry)

    # Construct paired detail records defensively
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Define the schema so empty DataFrames have consistent columns.
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


def search_scopus(query: str) -> pd.DataFrame:
    """Execute a Scopus advanced search and return structured results.

    Parameters
    ----------
    query:
        Scopus advanced search string, e.g.
        ``"TITLE-ABS-KEY(machine learning) AND PUBYEAR > 2019"``.

    Returns
    -------
    DataFrame with columns ``eid``, ``title``, ``year`` (int | None),
    ``affiliations`` (list[str]), ``doi`` (str | None), ``abstract`` (str | None),
    ``institutions`` (list[str]), ``countries`` (list[str]),
    ``affiliations_detail`` (list[dict[str, str]]), and ``authors`` (list[str]).

    Raises
    ------
    RuntimeError
        If the Scopus API returns an error.
    """
    # Import here so the config file is already in place when pybliometrics
    # reads it at module-import time.
    try:
        from pybliometrics.scopus import ScopusSearch
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import pybliometrics. Is it installed? {exc}"
        ) from exc

    try:
        search = ScopusSearch(query, refresh=True)
    except Exception as exc:
        raise RuntimeError(f"Scopus API error: {exc}") from exc

    results = search.results
    if not results:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

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
