"""
keyword_matcher.py — Case-insensitive keyword frequency and presence extraction.

Performs case-insensitive word-boundary matching across article text
(full text or abstract) for user-defined target keywords.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def parse_keywords(raw_keywords: str) -> list[str]:
    """Parse and clean a comma-separated string of user keywords.

    Parameters
    ----------
    raw_keywords:
        Comma-separated string, e.g. "deep learning, benchmark, neural network".

    Returns
    -------
    List of unique, non-empty, stripped keywords preserving insertion order.
    """
    if not raw_keywords or not isinstance(raw_keywords, str):
        return []

    tokens = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    seen: set[str] = set()
    cleaned: list[str] = []

    for t in tokens:
        lowered = t.lower()
        if lowered not in seen:
            seen.add(lowered)
            cleaned.append(t)

    return cleaned


def count_keyword(text: Any, keyword: str) -> int:
    """Count case-insensitive occurrences of a keyword in text with word boundaries.

    Parameters
    ----------
    text:
        Target string (full text or abstract).
    keyword:
        Keyword or multi-word term to search.

    Returns
    -------
    Integer count of occurrences.
    """
    if not text or not isinstance(text, str) or not keyword or not keyword.strip():
        return 0

    kw_clean = keyword.strip()
    # Word boundary regex respecting phrase boundaries
    pattern = rf"\b{re.escape(kw_clean)}\b"

    try:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return len(matches)
    except re.error:
        # Simple substring fallback if regex fails
        return text.lower().count(kw_clean.lower())


def match_keywords(text: Any, keywords: list[str]) -> dict[str, int]:
    """Calculate frequency counts for all specified keywords in text.

    Parameters
    ----------
    text:
        Target string.
    keywords:
        List of target keywords.

    Returns
    -------
    Dictionary mapping each keyword to its occurrence count.
    """
    return {kw: count_keyword(text, kw) for kw in keywords}


def add_keyword_features(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Enrich DataFrame with per-keyword boolean presence and frequency counts.

    Parameters
    ----------
    df:
        DataFrame containing a 'text' column.
    keywords:
        List of keywords to evaluate.

    Returns
    -------
    DataFrame with added columns:
        - 'kw_{keyword}_present' (bool)
        - 'kw_{keyword}_count' (int)
        - 'matched_keywords' (list[str])
        - 'total_keyword_hits' (int)
        - 'has_any_keyword' (bool)
    """
    df_out = df.copy()

    if df_out.empty or not keywords:
        df_out["matched_keywords"] = [[] for _ in range(len(df_out))]
        df_out["total_keyword_hits"] = 0
        df_out["has_any_keyword"] = False
        for kw in keywords:
            df_out[f"kw_{kw}_present"] = False
            df_out[f"kw_{kw}_count"] = 0
        return df_out

    text_series = df_out["text"].fillna("").astype(str)

    # Compute per-keyword columns
    counts_dict: dict[str, list[int]] = {}
    presence_dict: dict[str, list[bool]] = {}

    for kw in keywords:
        counts = [count_keyword(t, kw) for t in text_series]
        presences = [c > 0 for c in counts]
        counts_dict[kw] = counts
        presence_dict[kw] = presences
        df_out[f"kw_{kw}_count"] = counts
        df_out[f"kw_{kw}_present"] = presences

    # Aggregated metrics
    matched_list: list[list[str]] = []
    total_hits_list: list[int] = []
    has_any_list: list[bool] = []

    for i in range(len(df_out)):
        active_kws = [kw for kw in keywords if counts_dict[kw][i] > 0]
        total_hits = sum(counts_dict[kw][i] for kw in keywords)
        matched_list.append(active_kws)
        total_hits_list.append(total_hits)
        has_any_list.append(len(active_kws) > 0)

    df_out["matched_keywords"] = matched_list
    df_out["total_keyword_hits"] = total_hits_list
    df_out["has_any_keyword"] = has_any_list

    return df_out
