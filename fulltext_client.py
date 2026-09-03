"""
fulltext_client.py — Elsevier Article Retrieval with strict Abstract fallback.

Queries the Elsevier Article Retrieval API for full text given a paper's DOI.
If the article is paywalled (401/403), not in ScienceDirect (404), rate-limited (429),
missing a DOI, or encounters a network error, it automatically falls back
to the paper's Scopus abstract.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"


def clean_text(raw_text: Any) -> str:
    """Strip XML/HTML markup and collapse irregular whitespace.

    Parameters
    ----------
    raw_text:
        Raw string containing plain text, HTML, or XML tags.

    Returns
    -------
    Cleaned plain text string.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Fast path: check for XML/HTML tags
    if "<" in raw_text and ">" in raw_text:
        try:
            soup = BeautifulSoup(raw_text, "html.parser")
            text = soup.get_text(separator=" ")
        except Exception:
            # Regex fallback if parser fails
            text = re.sub(r"<[^>]+>", " ", raw_text)
    else:
        text = raw_text

    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def fetch_article_text(
    doi: str | None,
    api_key: str,
    abstract_fallback: str | None = None,
    timeout: int = 8,
    inst_token: str | None = None,
) -> tuple[str, str, str]:
    """Fetch article full text by DOI with strict fallback to abstract.

    Parameters
    ----------
    doi:
        Digital Object Identifier (e.g., '10.1016/j.cell.2023.01.001').
    api_key:
        Elsevier API key.
    abstract_fallback:
        Abstract text from Scopus Search API to use if full text is unavailable.
    timeout:
        Request timeout in seconds.
    inst_token:
        Optional Elsevier Institutional Token.

    Returns
    -------
    tuple of (text: str, text_source: str, status_detail: str)
        text_source is one of: 'Full Text', 'Abstract', 'None'.
    """
    cleaned_abstract = clean_text(abstract_fallback)

    def _fallback(reason: str) -> tuple[str, str, str]:
        if cleaned_abstract:
            return cleaned_abstract, "Abstract", reason
        return "", "None", f"{reason}; No Abstract Available"

    if not doi or not str(doi).strip():
        return _fallback("Missing DOI")

    clean_doi = str(doi).strip()
    # Normalize DOI by stripping URL prefix if present
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if clean_doi.lower().startswith(prefix):
            clean_doi = clean_doi[len(prefix):]
            break

    if not api_key or not api_key.strip():
        return _fallback("No API Key Provided")

    url = f"{_ELSEVIER_ARTICLE_URL}{clean_doi}"
    headers = {
        "X-ELS-APIKey": api_key.strip(),
        "Accept": "text/plain, text/xml, application/xml, application/json",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token.strip()

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"view": "FULL"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        logger.warning("Timeout retrieving full text for DOI %s", clean_doi)
        return _fallback("Request Timeout")
    except requests.exceptions.RequestException as exc:
        logger.warning("Network error retrieving full text for DOI %s: %s", clean_doi, exc)
        return _fallback(f"Connection Error: {exc.__class__.__name__}")

    status = response.status_code

    if status == 200:
        content_type = response.headers.get("Content-Type", "").lower()
        full_text = ""

        if "json" in content_type:
            try:
                data = response.json()
                resp_data = data.get("full-text-retrieval-response", {})
                full_text = resp_data.get("originalText", "") or ""
                if not full_text:
                    coredata = resp_data.get("coredata", {})
                    full_text = coredata.get("dc:description", "") or ""
            except Exception:
                full_text = ""
        else:
            full_text = response.text

        cleaned_full_text = clean_text(full_text)

        # Minimum substantive length check (avoid empty shells)
        if len(cleaned_full_text) >= 150:
            return cleaned_full_text, "Full Text", "Successfully Retrieved"

        # If payload was too short or empty, fallback
        return _fallback("Full Text Payload Incomplete")

    if status in (401, 403):
        return _fallback(f"Paywalled / Unauthorized (HTTP {status})")

    if status == 404:
        return _fallback("Not Found in ScienceDirect (HTTP 404)")

    if status == 429:
        return _fallback("API Rate Limit Exceeded (HTTP 429)")

    return _fallback(f"Elsevier HTTP Error {status}")


def enrich_dataset_with_text(
    df: pd.DataFrame,
    api_key: str,
    max_fulltext: int = 50,
    progress_callback: Any = None,
    inst_token: str | None = None,
    fetch_full_text: bool = True,
) -> pd.DataFrame:
    """Enrich a Scopus DataFrame with full text or fallback abstract.

    Parameters
    ----------
    df:
        DataFrame containing 'doi' and 'abstract' columns.
    api_key:
        Elsevier API key.
    max_fulltext:
        Maximum number of DOIs to query via the Elsevier API when full-text retrieval is active.
    progress_callback:
        Optional callback accepting (fraction: float, message: str).
    inst_token:
        Optional Elsevier Institutional Token.
    fetch_full_text:
        If True, query the Elsevier Article Retrieval API for each DOI with abstract fallback.
        If False, completely bypass the Elsevier API and populate text using Scopus abstracts.

    Returns
    -------
    DataFrame with added columns:
        'text', 'text_source', 'text_status_detail'.
    """
    if df.empty:
        df_out = df.copy()
        df_out["text"] = ""
        df_out["text_source"] = "None"
        df_out["text_status_detail"] = ""
        return df_out

    df_out = df.copy()
    texts: list[str] = []
    sources: list[str] = []
    details: list[str] = []

    total_rows = len(df_out)
    fulltext_attempts = 0

    for idx, (_, row) in enumerate(df_out.iterrows()):
        doi = row.get("doi")
        abstract = row.get("abstract")

        if progress_callback:
            progress_callback(
                (idx + 1) / total_rows,
                f"Processing text: article {idx + 1} of {total_rows}",
            )

        if fetch_full_text and doi and fulltext_attempts < max_fulltext:
            fulltext_attempts += 1
            text, source, detail = fetch_article_text(
                doi=doi,
                api_key=api_key,
                abstract_fallback=abstract,
                inst_token=inst_token,
            )
        else:
            # Use abstract directly (bypassed or limit exceeded or no DOI)
            clean_abs = clean_text(abstract)
            if clean_abs:
                text = clean_abs
                source = "Abstract"
                if not fetch_full_text:
                    detail = "Scopus Abstract (Full-Text Retrieval Bypassed)"
                elif not doi:
                    detail = "Scopus Abstract (No DOI)"
                else:
                    detail = "Scopus Abstract (Fetch Limit Reached)"
            else:
                text = ""
                source = "None"
                detail = "No Abstract Available"

        texts.append(text)
        sources.append(source)
        details.append(detail)

    df_out["text"] = texts
    df_out["text_source"] = sources
    df_out["text_status_detail"] = details

    return df_out
