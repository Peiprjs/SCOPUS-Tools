"""
batch_dataset_exporter.py — Mass Dataset & Provenance ZIP Exporter Engine.

Executes a batch of Scopus search queries defined in a specification CSV,
enriches records with toggleable full text / abstract fallback, classifies
institutional affiliations and geopolitics, computes keyword metrics, applies
granular country and institution filters, calculates SHA-256 cryptographic
provenance manifests, and compiles everything into a structured ZIP archive.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from classifier import classify_geography, classify_paper
from fulltext_client import enrich_dataset_with_text
from keyword_matcher import add_keyword_features, parse_keywords
from prism_exporter import generate_prism_xml
from ris_exporter import dataframe_to_ris_entries, export_ris_string
from scopus_client import format_eta, search_scopus

logger = logging.getLogger(__name__)


@dataclass
class JobSpecification:
    """Specification parameters for a single batch retrieval job."""

    job_id: int
    job_name: str
    slug: str
    query: str
    keywords: list[str]
    countries: list[str]
    institutions: list[str]
    fetch_full_text: bool
    max_results: int


@dataclass
class JobResult:
    """Execution output, dataset, citations, and provenance for a completed job."""

    job_spec: JobSpecification
    df: pd.DataFrame
    csv_text: str
    ris_text: str
    provenance: dict[str, Any]
    provenance_json: str
    duration_seconds: float
    total_retrieved: int
    total_filtered: int


def _slugify(text: str) -> str:
    """Create a safe filesystem-friendly slug from text."""
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:40] if slug else "job"


def _parse_list_field(val: Any) -> list[str]:
    """Parse comma- or semicolon-delimited values into a list of non-empty strings."""
    if val is None or pd.isna(val):
        return []
    text = str(val).strip()
    if not text:
        return []
    # Split by semicolon or comma
    items = re.split(r"[;,]", text)
    cleaned = [i.strip() for i in items if i.strip()]
    return cleaned


def _parse_bool_field(val: Any, default: bool = False) -> bool:
    """Parse boolean values from strings, integers, or booleans."""
    if val is None or pd.isna(val):
        return default
    if isinstance(val, bool):
        return val
    text = str(val).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "enabled"):
        return True
    if text in ("0", "false", "f", "no", "n", "disabled"):
        return False
    return default


def parse_batch_spec_csv(source: str | io.StringIO | bytes | pd.DataFrame) -> list[JobSpecification]:
    """Parse a batch specification CSV into a list of JobSpecification objects.

    Recognized column aliases:
      - query: 'query', 'search_query', 'scopus_query', 'queries', 'pattern'
      - keywords: 'keywords', 'target_keywords', 'keyword_filter', 'keyword_list'
      - countries: 'countries', 'country', 'country_filter', 'country_list'
      - institutions: 'institutions', 'institution', 'institution_filter', 'affiliations'
      - fetch_full_text: 'fetch_full_text', 'full_text', 'enable_fulltext', 'fulltext', 'full_text_retrieval'
      - job_name: 'job_name', 'cohort_name', 'name', 'id', 'label'
      - max_results: 'max_results', 'count', 'limit', 'max_papers'

    Parameters
    ----------
    source:
        File path, string buffer, raw bytes, or existing DataFrame.

    Returns
    -------
    list of JobSpecification instances.
    """
    try:
        if isinstance(source, pd.DataFrame):
            df_spec = source.copy()
        elif isinstance(source, bytes):
            if not source.strip():
                return []
            df_spec = pd.read_csv(io.BytesIO(source))
        elif isinstance(source, io.StringIO):
            df_spec = pd.read_csv(source)
        else:
            src_str = str(source).strip()
            if not src_str:
                return []
            if os.path.exists(src_str):
                df_spec = pd.read_csv(src_str)
            else:
                df_spec = pd.read_csv(io.StringIO(src_str))
    except (pd.errors.EmptyDataError, Exception):
        return []

    if df_spec.empty:
        return []

    # Normalized column mapping
    col_map: dict[str, str] = {}
    for col in df_spec.columns:
        norm = re.sub(r"[\s_-]+", "", str(col).lower())
        col_map[norm] = col

    def _get_col_val(row: pd.Series, aliases: list[str], fallback: Any = None) -> Any:
        for alias in aliases:
            norm_alias = re.sub(r"[\s_-]+", "", alias.lower())
            if norm_alias in col_map:
                val = row[col_map[norm_alias]]
                if val is not None and not pd.isna(val):
                    return val
        return fallback

    jobs: list[JobSpecification] = []

    for idx, (_, row) in enumerate(df_spec.iterrows()):
        query_val = _get_col_val(
            row,
            ["query", "search_query", "scopus_query", "queries", "pattern"],
            fallback="",
        )
        query_str = str(query_val).strip()
        if not query_str:
            continue

        job_name_val = _get_col_val(
            row,
            ["job_name", "cohort_name", "name", "label", "id"],
            fallback=f"Cohort {idx + 1}",
        )
        job_name = str(job_name_val).strip() or f"Cohort {idx + 1}"

        kw_val = _get_col_val(row, ["keywords", "target_keywords", "keyword_filter", "keyword_list"])
        keywords = _parse_list_field(kw_val)

        countries_val = _get_col_val(row, ["countries", "country", "country_filter", "country_list"])
        countries = _parse_list_field(countries_val)

        inst_val = _get_col_val(row, ["institutions", "institution", "institution_filter", "affiliations"])
        institutions = _parse_list_field(inst_val)

        ft_val = _get_col_val(
            row,
            ["fetch_full_text", "full_text", "enable_fulltext", "fulltext", "full_text_retrieval"],
            fallback=False,
        )
        fetch_full_text = _parse_bool_field(ft_val, default=False)

        max_res_val = _get_col_val(row, ["max_results", "count", "limit", "max_papers"], fallback=50)
        try:
            max_results = max(1, int(max_res_val))
        except (ValueError, TypeError):
            max_results = 50

        job_id = idx + 1
        slug = f"job_{job_id:02d}_{_slugify(job_name)}"

        jobs.append(
            JobSpecification(
                job_id=job_id,
                job_name=job_name,
                slug=slug,
                query=query_str,
                keywords=keywords,
                countries=countries,
                institutions=institutions,
                fetch_full_text=fetch_full_text,
                max_results=max_results,
            )
        )

    return jobs


def generate_template_csv() -> str:
    """Generate a canonical sample specification CSV."""
    df_template = pd.DataFrame(
        [
            {
                "job_name": "Nanomaterials Safety",
                "query": 'TITLE-ABS-KEY("Safe-by-Design" AND "Nanomaterials")',
                "keywords": "toxicity; environmental safety; lifecycle assessment",
                "countries": "Germany; France; Netherlands",
                "institutions": "",
                "fetch_full_text": "False",
                "max_results": 25,
            },
            {
                "job_name": "Graphene Advanced Materials",
                "query": 'TITLE-ABS-KEY("Graphene" AND "Synthesis" AND PUBYEAR > 2022)',
                "keywords": "chemical vapor deposition; electrical conductivity",
                "countries": "",
                "institutions": "Max Planck; CNRS",
                "fetch_full_text": "False",
                "max_results": 25,
            },
            {
                "job_name": "Sustainable Batteries",
                "query": 'TITLE-ABS-KEY("Solid-State Battery" AND "Recycling")',
                "keywords": "cobalt; lithium; circular economy",
                "countries": "",
                "institutions": "",
                "fetch_full_text": "False",
                "max_results": 25,
            },
        ]
    )
    return df_template.to_csv(index=False)


def _matches_filter(row_items: Any, filter_criteria: list[str]) -> bool:
    """Check if any item in row_items matches any filter criterion (case-insensitive substring)."""
    if not filter_criteria:
        return True
    if not row_items or not isinstance(row_items, (list, set, tuple)):
        return False

    norm_criteria = [c.lower().strip() for c in filter_criteria if c.strip()]
    for item in row_items:
        if not item or not isinstance(item, str):
            continue
        item_lower = item.lower().strip()
        for crit in norm_criteria:
            if crit in item_lower:
                return True
    return False


def _compute_sha256(content: str) -> str:
    """Calculate SHA-256 cryptographic digest of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_provenance_manifest(
    job_spec: JobSpecification,
    df_filtered: pd.DataFrame,
    csv_text: str,
    ris_text: str,
    total_retrieved: int,
    duration_seconds: float,
) -> dict[str, Any]:
    """Generate a formal, comprehensive cryptographic and scientific provenance manifest."""
    total_cohort = len(df_filtered)

    # 1. Full-text telemetry
    ft_counts = df_filtered["text_source"].value_counts().to_dict() if "text_source" in df_filtered.columns else {}
    n_full = int(ft_counts.get("Full Text", 0))
    n_abs = int(ft_counts.get("Abstract", 0))
    n_none = int(ft_counts.get("None", 0))
    ft_rate = round((n_full / total_cohort * 100), 2) if total_cohort > 0 else 0.0

    status_details_dist = (
        df_filtered["text_status_detail"].value_counts().to_dict()
        if "text_status_detail" in df_filtered.columns
        else {}
    )

    # 2. Sectoral classification distribution
    cat_counts = df_filtered["category"].value_counts().to_dict() if "category" in df_filtered.columns else {}
    affil_dist = {
        k: {
            "count": int(v),
            "percentage": round(v / total_cohort * 100, 2) if total_cohort > 0 else 0.0,
        }
        for k, v in cat_counts.items()
    }

    # 3. Geopolitical classification distribution
    geo_counts = df_filtered["geo_category"].value_counts().to_dict() if "geo_category" in df_filtered.columns else {}
    geo_dist = {
        k: {
            "count": int(v),
            "percentage": round(v / total_cohort * 100, 2) if total_cohort > 0 else 0.0,
        }
        for k, v in geo_counts.items()
    }

    # 4. Keyword matches
    kw_stats: dict[str, Any] = {}
    if job_spec.keywords:
        for kw in job_spec.keywords:
            col_count = f"kw_count_{kw}"
            if col_count in df_filtered.columns:
                occurrences = int(df_filtered[col_count].sum())
                papers_with_kw = int((df_filtered[col_count] > 0).sum())
                kw_stats[kw] = {
                    "total_occurrences": occurrences,
                    "papers_matching": papers_with_kw,
                    "cohort_prevalence_pct": round(papers_with_kw / total_cohort * 100, 2) if total_cohort > 0 else 0.0,
                }

    # 5. File checksums
    csv_sha256 = _compute_sha256(csv_text)
    ris_sha256 = _compute_sha256(ris_text)

    # 6. Assemble complete provenance dictionary
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "provenance_type": "ScopusBibliometricDatasetManifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_metadata": {
            "job_id": job_spec.job_id,
            "job_name": job_spec.job_name,
            "slug": job_spec.slug,
            "search_query": job_spec.query,
            "configured_limit": job_spec.max_results,
        },
        "query_parameters": {
            "target_keywords": job_spec.keywords,
            "country_filters": job_spec.countries,
            "institution_filters": job_spec.institutions,
            "full_text_retrieval_enabled": job_spec.fetch_full_text,
        },
        "execution_metrics": {
            "duration_seconds": round(duration_seconds, 3),
            "records_retrieved_from_api": total_retrieved,
            "records_in_filtered_cohort": total_cohort,
            "filtered_out_count": total_retrieved - total_cohort,
        },
        "full_text_telemetry": {
            "mode": "Elsevier Article Retrieval API" if job_spec.fetch_full_text else "Bypassed (Abstract Only)",
            "full_text_count": n_full,
            "abstract_fallback_count": n_abs,
            "missing_text_count": n_none,
            "full_text_retrieval_rate_pct": ft_rate,
            "status_details_distribution": status_details_dist,
        },
        "classification_summary": {
            "affiliation_sectoral_distribution": affil_dist,
            "geopolitical_distribution": geo_dist,
        },
        "keyword_analytics": kw_stats,
        "artifact_verification": {
            "dataset_csv": {
                "file_name": "dataset.csv",
                "sha256_checksum": csv_sha256,
                "byte_size": len(csv_text.encode("utf-8")),
                "row_count": total_cohort,
            },
            "citations_ris": {
                "file_name": "citations.ris",
                "sha256_checksum": ris_sha256,
                "byte_size": len(ris_text.encode("utf-8")),
                "reference_count": total_cohort,
            },
        },
        "runtime_environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }

    return manifest


def execute_batch_job(
    job_spec: JobSpecification,
    api_key: str,
    inst_token: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> JobResult:
    """Execute a single Scopus batch job, apply filters, and build provenance.

    Parameters
    ----------
    job_spec:
        Specification for the retrieval job.
    api_key:
        Elsevier / Scopus API key.
    inst_token:
        Optional Elsevier Institutional Token.
    progress_callback:
        Optional callback accepting (fraction: float, message: str).

    Returns
    -------
    JobResult containing DataFrame, CSV text, RIS text, and provenance manifest.
    """
    t_start = time.time()
    logger.info("Executing batch job %d: '%s' (Query: %s)", job_spec.job_id, job_spec.job_name, job_spec.query)

    if progress_callback:
        progress_callback(0.05, f"Searching Scopus API: {job_spec.job_name}")

    # 1. Search Scopus
    raw_data = search_scopus(
        query=job_spec.query,
        count=25,
        api_key=api_key,
        inst_token=inst_token,
    )
    total_retrieved = len(raw_data)

    if raw_data.empty:
        logger.warning("Job %d returned 0 records for query: %s", job_spec.job_id, job_spec.query)
        empty_df = pd.DataFrame()
        duration = time.time() - t_start
        csv_text = ""
        ris_text = ""
        prov = generate_provenance_manifest(job_spec, empty_df, csv_text, ris_text, 0, duration)
        prov_json = json.dumps(prov, indent=2)
        return JobResult(
            job_spec=job_spec,
            df=empty_df,
            csv_text=csv_text,
            ris_text=ris_text,
            provenance=prov,
            provenance_json=prov_json,
            duration_seconds=duration,
            total_retrieved=0,
            total_filtered=0,
        )

    # Truncate to max_results if specified
    if job_spec.max_results and total_retrieved > job_spec.max_results:
        raw_data = raw_data.head(job_spec.max_results).copy()

    # 2. Enrich with text (full text or abstract fallback)
    if progress_callback:
        progress_callback(0.35, f"Enriching text payloads: {job_spec.job_name}")

    enriched = enrich_dataset_with_text(
        df=raw_data,
        api_key=api_key,
        max_fulltext=job_spec.max_results if job_spec.fetch_full_text else 0,
        inst_token=inst_token,
        fetch_full_text=job_spec.fetch_full_text,
    )

    # 3. Classify affiliations
    if progress_callback:
        progress_callback(0.60, f"Classifying affiliations & geopolitics: {job_spec.job_name}")

    enriched["category"] = enriched["affiliations"].apply(
        lambda aff: classify_paper(aff) if isinstance(aff, list) else classify_paper([])
    )

    # 4. Classify geopolitics
    enriched["geo_category"] = enriched.apply(
        lambda r: classify_geography(r.get("countries")),
        axis=1,
    )

    # 5. Add keyword features
    if job_spec.keywords:
        enriched = add_keyword_features(enriched, job_spec.keywords)
    else:
        enriched["has_any_keyword"] = True

    # 6. Apply Granular Filters (Country and Institution)
    filtered_df = enriched.copy()

    if job_spec.countries:
        filtered_df = filtered_df[
            filtered_df["countries"].apply(lambda c_list: _matches_filter(c_list, job_spec.countries))
        ]

    if job_spec.institutions:
        filtered_df = filtered_df[
            filtered_df["institutions"].apply(lambda i_list: _matches_filter(i_list, job_spec.institutions))
        ]

    total_filtered = len(filtered_df)

    # 7. Generate structured CSV & RIS
    if progress_callback:
        progress_callback(0.85, f"Formatting CSV and RIS outputs: {job_spec.job_name}")

    # Format list fields as readable strings for export
    export_df = filtered_df.copy()
    if "affiliations_detail" in export_df.columns:
        export_df["affiliations_detail"] = export_df["affiliations_detail"].apply(
            lambda v: json.dumps(v) if isinstance(v, (list, dict)) else str(v) if v is not None else ""
        )
    for col in ["authors", "institutions", "countries", "affiliations"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda v: "; ".join(v) if isinstance(v, list) else str(v) if v is not None else ""
            )

    csv_text = export_df.to_csv(index=False)
    ris_entries = dataframe_to_ris_entries(filtered_df)
    ris_text = export_ris_string(ris_entries)

    duration = time.time() - t_start

    # 8. Compute Provenance Manifest
    if progress_callback:
        progress_callback(0.95, f"Computing cryptographic provenance: {job_spec.job_name}")

    prov = generate_provenance_manifest(
        job_spec=job_spec,
        df_filtered=filtered_df,
        csv_text=csv_text,
        ris_text=ris_text,
        total_retrieved=total_retrieved,
        duration_seconds=duration,
    )
    prov_json = json.dumps(prov, indent=2)

    if progress_callback:
        progress_callback(1.0, f"Completed: {job_spec.job_name}")

    return JobResult(
        job_spec=job_spec,
        df=filtered_df,
        csv_text=csv_text,
        ris_text=ris_text,
        provenance=prov,
        provenance_json=prov_json,
        duration_seconds=duration,
        total_retrieved=total_retrieved,
        total_filtered=total_filtered,
    )


def build_batch_provenance_zip(
    results: list[JobResult],
    batch_metadata: dict[str, Any] | None = None,
    export_prism: bool = True,
) -> bytes:
    """Bundle all job datasets, citations, provenance manifests, and optional Prism file into a ZIP archive.

    Directory Structure:
      - batch_manifest.json
      - consolidated_dataset.csv
      - consolidated_citations.ris
      - scopus_mass_export.pzfx (when export_prism is True)
      - jobs/
        - job_01_<slug>/
          - dataset.csv
          - citations.ris
          - provenance.json
        - job_02_<slug>/
          ...

    Parameters
    ----------
    results:
        List of completed JobResult objects.
    batch_metadata:
        Optional overarching metadata (title, requester, description).
    export_prism:
        If True, compile and include a unified GraphPad Prism project (.pzfx) covering all jobs.

    Returns
    -------
    Bytes representing the compressed ZIP archive.
    """
    buffer = io.BytesIO()

    # 1. Compile consolidated dataset
    consolidated_rows: list[pd.DataFrame] = []
    consolidated_ris_entries: list[dict[str, Any]] = []

    for r in results:
        if not r.df.empty:
            df_c = r.df.copy()
            df_c.insert(0, "batch_job_id", r.job_spec.job_id)
            df_c.insert(1, "batch_job_name", r.job_spec.job_name)
            consolidated_rows.append(df_c)
            consolidated_ris_entries.extend(dataframe_to_ris_entries(r.df))

    if consolidated_rows:
        consolidated_df = pd.concat(consolidated_rows, ignore_index=True)
        # Format export columns
        c_export = consolidated_df.copy()
        if "affiliations_detail" in c_export.columns:
            c_export["affiliations_detail"] = c_export["affiliations_detail"].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else str(v) if v is not None else ""
            )
        for col in ["authors", "institutions", "countries", "affiliations"]:
            if col in c_export.columns:
                c_export[col] = c_export[col].apply(
                    lambda v: "; ".join(v) if isinstance(v, list) else str(v) if v is not None else ""
                )
        consolidated_csv_text = c_export.to_csv(index=False)
    else:
        consolidated_csv_text = ""

    consolidated_ris_text = export_ris_string(consolidated_ris_entries)

    # 2. Compile GraphPad Prism Project (.pzfx) if requested
    prism_xml = ""
    prism_sha256 = ""
    if export_prism and results:
        try:
            p_title = (batch_metadata or {}).get("title", "Scopus Bibliometric Mass Export")
            prism_xml = generate_prism_xml(results, project_title=p_title)
            prism_sha256 = _compute_sha256(prism_xml)
        except Exception as exc:
            logger.error("Failed generating GraphPad Prism XML for batch export: %s", exc)
            prism_xml = ""

    # 3. Compile Master Batch Manifest
    total_jobs = len(results)
    total_retrieved = sum(r.total_retrieved for r in results)
    total_filtered = sum(r.total_filtered for r in results)
    total_time = sum(r.duration_seconds for r in results)

    master_manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "provenance_type": "ScopusMasterBatchManifest",
        "batch_title": (batch_metadata or {}).get("title", "Scopus Mass Dataset Export Bundle"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_jobs_executed": total_jobs,
            "total_records_retrieved": total_retrieved,
            "total_records_in_cohorts": total_filtered,
            "total_processing_seconds": round(total_time, 2),
        },
        "consolidated_artifacts": {
            "consolidated_dataset_csv": {
                "file_path": "consolidated_dataset.csv",
                "sha256": _compute_sha256(consolidated_csv_text),
                "records": total_filtered,
            },
            "consolidated_citations_ris": {
                "file_path": "consolidated_citations.ris",
                "sha256": _compute_sha256(consolidated_ris_text),
                "references": len(consolidated_ris_entries),
            },
        },
        "jobs": [
            {
                "job_id": r.job_spec.job_id,
                "job_name": r.job_spec.job_name,
                "directory": f"jobs/{r.job_spec.slug}/",
                "search_query": r.job_spec.query,
                "target_keywords": r.job_spec.keywords,
                "country_filters": r.job_spec.countries,
                "institution_filters": r.job_spec.institutions,
                "fetch_full_text": r.job_spec.fetch_full_text,
                "records_retrieved": r.total_retrieved,
                "records_filtered": r.total_filtered,
                "duration_seconds": round(r.duration_seconds, 2),
                "dataset_sha256": r.provenance.get("artifact_verification", {}).get("dataset_csv", {}).get("sha256_checksum"),
                "citations_sha256": r.provenance.get("artifact_verification", {}).get("citations_ris", {}).get("sha256_checksum"),
            }
            for r in results
        ],
    }

    if prism_xml:
        master_manifest["consolidated_artifacts"]["graphpad_prism_project"] = {
            "file_path": "scopus_mass_export.pzfx",
            "sha256": prism_sha256,
            "byte_size": len(prism_xml.encode("utf-8")),
            "format": "GraphPad Prism XML 5.00",
        }

    # 4. Write ZIP Archive
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Root files
        zf.writestr("batch_manifest.json", json.dumps(master_manifest, indent=2))
        zf.writestr("consolidated_dataset.csv", consolidated_csv_text)
        zf.writestr("consolidated_citations.ris", consolidated_ris_text)
        if prism_xml:
            zf.writestr("scopus_mass_export.pzfx", prism_xml)

        # Per-job files
        for r in results:
            job_dir = f"jobs/{r.job_spec.slug}"
            zf.writestr(f"{job_dir}/dataset.csv", r.csv_text)
            zf.writestr(f"{job_dir}/citations.ris", r.ris_text)
            zf.writestr(f"{job_dir}/provenance.json", r.provenance_json)

    buffer.seek(0)
    return buffer.getvalue()
