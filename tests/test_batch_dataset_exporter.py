"""
tests/test_batch_dataset_exporter.py — Unit tests for mass dataset and provenance exporter.
"""

from __future__ import annotations

import io
import json
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd

from batch_dataset_exporter import (
    JobResult,
    JobSpecification,
    _matches_filter,
    build_batch_provenance_zip,
    execute_batch_job,
    generate_provenance_manifest,
    generate_template_csv,
    parse_batch_spec_csv,
)


class TestBatchDatasetExporter(unittest.TestCase):
    def test_parse_batch_spec_csv_standard(self):
        csv_data = (
            "job_name,query,keywords,countries,institutions,fetch_full_text,max_results\n"
            'Nano Safety,TITLE-ABS-KEY("nanoparticles"),toxicity; safety,Germany; France,Max Planck,True,20\n'
            'Battery Tech,TITLE-ABS-KEY("battery"),cobalt; lithium,,,False,10\n'
        )
        jobs = parse_batch_spec_csv(csv_data)
        self.assertEqual(len(jobs), 2)

        j1 = jobs[0]
        self.assertEqual(j1.job_id, 1)
        self.assertEqual(j1.job_name, "Nano Safety")
        self.assertEqual(j1.query, 'TITLE-ABS-KEY("nanoparticles")')
        self.assertEqual(j1.keywords, ["toxicity", "safety"])
        self.assertEqual(j1.countries, ["Germany", "France"])
        self.assertEqual(j1.institutions, ["Max Planck"])
        self.assertTrue(j1.fetch_full_text)
        self.assertEqual(j1.max_results, 20)

        j2 = jobs[1]
        self.assertEqual(j2.job_id, 2)
        self.assertEqual(j2.job_name, "Battery Tech")
        self.assertEqual(j2.keywords, ["cobalt", "lithium"])
        self.assertEqual(j2.countries, [])
        self.assertEqual(j2.institutions, [])
        self.assertFalse(j2.fetch_full_text)
        self.assertEqual(j2.max_results, 10)

    def test_parse_batch_spec_csv_aliases_and_booleans(self):
        csv_data = (
            "cohort_name,search_query,target_keywords,country_filter,institution_filter,fulltext\n"
            'Cohort A,TITLE("test"),term1,Spain,University of Madrid,1\n'
            'Cohort B,TITLE("test2"),term2,Italy,,yes\n'
        )
        jobs = parse_batch_spec_csv(csv_data)
        self.assertEqual(len(jobs), 2)
        self.assertTrue(jobs[0].fetch_full_text)
        self.assertTrue(jobs[1].fetch_full_text)
        self.assertEqual(jobs[0].countries, ["Spain"])
        self.assertEqual(jobs[0].institutions, ["University of Madrid"])

    def test_parse_batch_spec_csv_empty(self):
        jobs = parse_batch_spec_csv("")
        self.assertEqual(jobs, [])

        empty_df = pd.DataFrame(columns=["query"])
        self.assertEqual(parse_batch_spec_csv(empty_df), [])

    def test_generate_template_csv(self):
        template = generate_template_csv()
        self.assertIn("job_name", template)
        self.assertIn("query", template)
        self.assertIn("keywords", template)
        jobs = parse_batch_spec_csv(template)
        self.assertTrue(len(jobs) >= 2)

    def test_matches_filter(self):
        # Empty criteria should match everything
        self.assertTrue(_matches_filter(["Germany"], []))
        self.assertTrue(_matches_filter([], []))

        # Case-insensitive substring matching
        self.assertTrue(_matches_filter(["Federal Republic of Germany"], ["germany"]))
        self.assertTrue(_matches_filter(["Max Planck Institute"], ["max planck"]))
        self.assertFalse(_matches_filter(["Harvard University"], ["mit"]))
        self.assertFalse(_matches_filter(None, ["germany"]))

    def test_generate_provenance_manifest(self):
        job_spec = JobSpecification(
            job_id=1,
            job_name="Test Cohort",
            slug="job_01_test_cohort",
            query="TITLE(test)",
            keywords=["ai"],
            countries=["Germany"],
            institutions=["Max Planck"],
            fetch_full_text=False,
            max_results=50,
        )
        df = pd.DataFrame(
            [
                {
                    "title": "Paper 1",
                    "category": "Academia",
                    "geo_category": "EU/EEC",
                    "text_source": "Abstract",
                    "text_status_detail": "Scopus Abstract (Bypassed)",
                    "kw_count_ai": 2,
                }
            ]
        )
        csv_text = df.to_csv(index=False)
        ris_text = "TY  - JOUR\nTI  - Paper 1\nER  -\n"

        prov = generate_provenance_manifest(
            job_spec=job_spec,
            df_filtered=df,
            csv_text=csv_text,
            ris_text=ris_text,
            total_retrieved=1,
            duration_seconds=1.23,
        )

        self.assertEqual(prov["provenance_type"], "ScopusBibliometricDatasetManifest")
        self.assertEqual(prov["job_metadata"]["job_id"], 1)
        self.assertEqual(prov["execution_metrics"]["records_in_filtered_cohort"], 1)
        self.assertIn("dataset_csv", prov["artifact_verification"])
        self.assertIn("sha256_checksum", prov["artifact_verification"]["dataset_csv"])
        self.assertIn("citations_ris", prov["artifact_verification"])

    @patch("batch_dataset_exporter.search_scopus")
    @patch("batch_dataset_exporter.enrich_dataset_with_text")
    def test_execute_batch_job(self, mock_enrich, mock_search):
        mock_raw = pd.DataFrame(
            [
                {
                    "eid": "2-s2.0-001",
                    "doi": "10.1016/j.test.1",
                    "title": "Machine Learning in Nanomaterials",
                    "year": 2023,
                    "abstract": "Deep learning models for materials.",
                    "authors": ["Mueller, H.", "Smith, J."],
                    "affiliations": ["University of Berlin", "Siemens AG"],
                    "countries": ["Germany", "United States"],
                    "institutions": ["University of Berlin", "Siemens AG"],
                    "affiliations_detail": [
                        {"institution": "University of Berlin", "country": "Germany"},
                        {"institution": "Siemens AG", "country": "United States"},
                    ],
                },
                {
                    "eid": "2-s2.0-002",
                    "doi": "10.1016/j.test.2",
                    "title": "Polymer Synthesis",
                    "year": 2022,
                    "abstract": "Traditional synthesis methods.",
                    "authors": ["Dupont, P."],
                    "affiliations": ["CNRS Paris"],
                    "countries": ["France"],
                    "institutions": ["CNRS Paris"],
                    "affiliations_detail": [{"institution": "CNRS Paris", "country": "France"}],
                },
            ]
        )
        mock_search.return_value = mock_raw

        mock_enriched = mock_raw.copy()
        mock_enriched["text"] = mock_enriched["abstract"]
        mock_enriched["text_source"] = "Abstract"
        mock_enriched["text_status_detail"] = "Scopus Abstract"
        mock_enrich.return_value = mock_enriched

        job_spec = JobSpecification(
            job_id=1,
            job_name="German Nano Research",
            slug="job_01_german_nano",
            query='TITLE("nano")',
            keywords=["deep learning"],
            countries=["Germany"],
            institutions=[],
            fetch_full_text=False,
            max_results=50,
        )

        result = execute_batch_job(job_spec=job_spec, api_key="test_key")

        self.assertEqual(result.total_retrieved, 2)
        # Only row 1 has country Germany
        self.assertEqual(result.total_filtered, 1)
        self.assertEqual(len(result.df), 1)
        self.assertEqual(result.df.iloc[0]["title"], "Machine Learning in Nanomaterials")
        self.assertIn("citations_ris", result.provenance["artifact_verification"])
        self.assertEqual(result.provenance["artifact_verification"]["citations_ris"]["file_name"], "citations.ris")

    def test_build_batch_provenance_zip(self):
        job_spec = JobSpecification(
            job_id=1,
            job_name="Test Job",
            slug="job_01_test_job",
            query="TITLE(test)",
            keywords=["test"],
            countries=[],
            institutions=[],
            fetch_full_text=False,
            max_results=10,
        )
        df = pd.DataFrame(
            [
                {
                    "title": "Study 1",
                    "year": 2023,
                    "doi": "10.1016/j.test.study",
                    "abstract": "Study abstract.",
                    "authors": ["Author, A."],
                    "affiliations": ["Univ 1"],
                    "countries": ["UK"],
                    "institutions": ["Univ 1"],
                    "category": "Academia",
                    "geo_category": "Non-EU/EEC",
                    "text_source": "Abstract",
                    "text_status_detail": "Scopus Abstract",
                }
            ]
        )
        csv_text = df.to_csv(index=False)
        ris_text = "TY  - JOUR\nTI  - Study 1\nER  -\n"
        prov = generate_provenance_manifest(job_spec, df, csv_text, ris_text, 1, 0.5)
        prov_json = json.dumps(prov, indent=2)

        res = JobResult(
            job_spec=job_spec,
            df=df,
            csv_text=csv_text,
            ris_text=ris_text,
            provenance=prov,
            provenance_json=prov_json,
            duration_seconds=0.5,
            total_retrieved=1,
            total_filtered=1,
        )

        zip_bytes = build_batch_provenance_zip([res])
        self.assertTrue(len(zip_bytes) > 0)

        # Inspect ZIP structure
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            self.assertIn("batch_manifest.json", names)
            self.assertIn("consolidated_dataset.csv", names)
            self.assertIn("consolidated_citations.ris", names)
            self.assertIn("scopus_mass_export.pzfx", names)
            self.assertIn("jobs/job_01_test_job/dataset.csv", names)
            self.assertIn("jobs/job_01_test_job/citations.ris", names)
            self.assertIn("jobs/job_01_test_job/provenance.json", names)

            # Check batch manifest parsing
            manifest_data = json.loads(zf.read("batch_manifest.json").decode("utf-8"))
            self.assertEqual(manifest_data["provenance_type"], "ScopusMasterBatchManifest")
            self.assertEqual(manifest_data["summary"]["total_jobs_executed"], 1)
            self.assertIn("graphpad_prism_project", manifest_data["consolidated_artifacts"])


if __name__ == "__main__":
    unittest.main()
