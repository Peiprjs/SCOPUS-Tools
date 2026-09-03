"""
classifier.py — Affiliation classification heuristics.

Classifies affiliation strings into Academia, Industry, or Unknown,
and classifies entire papers based on the mix of their authors' affiliations.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

# Each tuple: (pattern, is_regex)
# Non-regex patterns are matched as case-insensitive substrings.
# Regex patterns use word-boundary matching for short/ambiguous tokens.

_ACADEMIA_KEYWORDS: list[str] = [
    "university",
    "universität",
    "université",
    "universidad",
    "università",
    "universidade",
    "univ.",
    "college",
    "institute of technology",
    "polytechnic",
    "school of",
    "academy",
    # Major public research organisations
    "cnrs",
    "inria",
    "inserm",
    "max planck",
    "fraunhofer",
    "helmholtz",
    "leibniz",
    "csic",
    "csiro",
    "chinese academy of sciences",
    "russian academy of sciences",
    # Well-known abbreviated names
    "mit",
    "eth zurich",
    "epfl",
    "caltech",
    "nasa",
    "cern",
    "nih",
    # Generic research-facility markers
    "research center",
    "research centre",
    "research institute",
    "faculty of",
    "department of",
    "graduate school",
    "medical school",
    "hospital",
    "medical center",
    "medical centre",
    "national laboratory",
    "national lab",
]

_INDUSTRY_KEYWORDS: list[str] = [
    "inc.",
    "inc,",
    "corp.",
    "corp,",
    "corporation",
    "ltd",
    "llc",
    "gmbh",
    "s.a.",
    "s.r.l.",
    "plc",
    "company",
    "co.",
    "technologies",
    "pharma",
    "pharmaceutical",
    "biotech",
    "therapeutics",
    "solutions",
]

# Short/ambiguous industry tokens that need word-boundary matching to avoid
# false positives (e.g. "Aga Khan" matching "ag").
_INDUSTRY_REGEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bag\b", re.IGNORECASE),
    re.compile(r"\bse\b", re.IGNORECASE),
    re.compile(r"\bgroup\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Single-affiliation classifier
# ---------------------------------------------------------------------------

def classify_affiliation(affil_string: str) -> str:
    """Classify a single affiliation string.

    Parameters
    ----------
    affil_string:
        Raw affiliation name, e.g. ``"Massachusetts Institute of Technology"``.

    Returns
    -------
    ``"Academia"``, ``"Industry"``, or ``"Unknown"``.
    """
    if not affil_string or not affil_string.strip():
        return "Unknown"

    lowered = affil_string.lower().strip()

    is_academia = any(kw in lowered for kw in _ACADEMIA_KEYWORDS)
    is_industry = (
        any(kw in lowered for kw in _INDUSTRY_KEYWORDS)
        or any(pat.search(affil_string) for pat in _INDUSTRY_REGEX_PATTERNS)
    )

    if is_academia and is_industry:
        # When both match (rare edge-case, e.g. "University Hospital GmbH"),
        # prefer Academia since the primary institution is academic.
        return "Academia"
    if is_academia:
        return "Academia"
    if is_industry:
        return "Industry"

    return "Unknown"


# ---------------------------------------------------------------------------
# Paper-level classifier
# ---------------------------------------------------------------------------

def classify_paper(affiliations: list[str]) -> str:
    """Classify a paper based on all its authors' affiliations.

    Parameters
    ----------
    affiliations:
        List of affiliation name strings for a single paper.

    Returns
    -------
    ``"Academia"``, ``"Industry"``, ``"Mixed"``, or ``"Unknown"``.

    Rules
    -----
    - All resolved affiliations Academia → ``"Academia"``
    - All resolved affiliations Industry → ``"Industry"``
    - A mix of both → ``"Mixed"``
    - Empty list or all Unknown → ``"Unknown"``
    """
    if not affiliations:
        return "Unknown"

    categories = {classify_affiliation(a) for a in affiliations}
    # Discard Unknown for the purpose of deciding the mix
    resolved = categories - {"Unknown"}

    if not resolved:
        return "Unknown"
    if resolved == {"Academia"}:
        return "Academia"
    if resolved == {"Industry"}:
        return "Industry"
    return "Mixed"


# ---------------------------------------------------------------------------
# Geopolitical Classification (EU / EEC)
# ---------------------------------------------------------------------------

EU_EEC_COUNTRIES: set[str] = {
    # EU 27 Member States
    "austria",
    "belgium",
    "bulgaria",
    "croatia",
    "cyprus",
    "czech republic",
    "czechia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "ireland",
    "republic of ireland",
    "italy",
    "latvia",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "the netherlands",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
    # EEC / EEA EFTA Member States
    "iceland",
    "liechtenstein",
    "norway",
}


def classify_geography(countries: list[str]) -> str:
    """Classify a paper's geopolitical scope based on affiliation countries.

    Parameters
    ----------
    countries:
        List of country names for all author affiliations of a paper.

    Returns
    -------
    ``"EU/EEC"``, ``"Non-EU/EEC"``, ``"Mixed Geo"``, or ``"Unknown Geo"``.

    Rules
    -----
    - Only EU/EEC countries present -> ``"EU/EEC"``
    - Only countries outside EU/EEC present -> ``"Non-EU/EEC"``
    - Both EU/EEC and non-EU/EEC countries present -> ``"Mixed Geo"``
    - Empty list or all unresolved -> ``"Unknown Geo"``
    """
    if not countries:
        return "Unknown Geo"

    clean_countries = [c.strip().lower() for c in countries if c and str(c).strip()]
    if not clean_countries:
        return "Unknown Geo"

    is_eu_flags = [c in EU_EEC_COUNTRIES for c in clean_countries]
    has_eu = any(is_eu_flags)
    has_non_eu = any(not flag for flag in is_eu_flags)

    if has_eu and has_non_eu:
        return "Mixed Geo"
    if has_eu:
        return "EU/EEC"
    return "Non-EU/EEC"

