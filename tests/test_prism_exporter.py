"""
tests/test_prism_exporter.py — Unit tests for GraphPad Prism (.pzfx) XML project exporter.
Zero emojis, formal academic design.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from batch_dataset_exporter import (
    JobResult,
    JobSpecification,
    build_batch_provenance_zip,
    generate_provenance_manifest,
)
from prism_exporter import (
    PRISM_XML_NAMESPACE,
    export_prism_file,
    generate_prism_xml,
)


class TestPrismExporter(unittest.TestCase):
    def setUp(self) -> None:
        self.spec1 = JobSpecification(
            job_id=1,
            job_name="Safe-by-Design Nanomaterials",
            slug="job_01_safe_by_design_nanomaterials",
            query='TITLE-ABS-KEY("safe-by-design" AND "nanomaterials")',
            keywords=["toxicity", "exposure"],
            countries=["Germany", "France"],
            institutions=["Max Planck Institute"],
            fetch_full_text=True,
            max_results=50,
        )
        self.df1 = pd.DataFrame(
            [
                {
                    "title": "Toxicity assessment of zinc oxide nanoparticles",
                    "year": 2021,
                    "doi": "10.1016/j.impact.2021.1001",
                    "authors": ["Mueller, H.", "Schmidt, A."],
                    "affiliations": ["Max Planck Institute, Germany"],
                    "countries": ["Germany"],
                    "institutions": ["Max Planck Institute"],
                    "category": "Academia",
                    "geo_category": "EU/EEC",
                    "text_source": "Elsevier Article Retrieval API",
                    "text_status_detail": "Full Text Retrieved",
                    "kw_count_toxicity": 5,
                    "kw_count_exposure": 2,
                    "target_keywords_present": True,
                },
                {
                    "title": "Industrial exposure monitoring in nanomaterial synthesis",
                    "year": 2022,
                    "doi": "10.1016/j.impact.2022.1002",
                    "authors": ["Weber, K."],
                    "affiliations": ["BASF SE, Germany"],
                    "countries": ["Germany"],
                    "institutions": ["BASF SE"],
                    "category": "Industry",
                    "geo_category": "EU/EEC",
                    "text_source": "Abstract",
                    "text_status_detail": "Scopus Abstract (Full-Text Fallback)",
                    "kw_count_toxicity": 0,
                    "kw_count_exposure": 4,
                    "target_keywords_present": True,
                },
            ]
        )
        self.csv1 = self.df1.to_csv(index=False)
        self.ris1 = "TY  - JOUR\nTI  - Paper 1\nER  -\nTY  - JOUR\nTI  - Paper 2\nER  -\n"
        self.prov1 = generate_provenance_manifest(self.spec1, self.df1, self.csv1, self.ris1, 2, 1.5)
        self.result1 = JobResult(
            job_spec=self.spec1,
            df=self.df1,
            csv_text=self.csv1,
            ris_text=self.ris1,
            provenance=self.prov1,
            provenance_json=json.dumps(self.prov1, indent=2),
            duration_seconds=1.5,
            total_retrieved=2,
            total_filtered=2,
        )

        self.spec2 = JobSpecification(
            job_id=2,
            job_name="Battery Recycling Chemistry",
            slug="job_02_battery_recycling_chemistry",
            query='TITLE-ABS-KEY("battery recycling")',
            keywords=["lithium", "cobalt"],
            countries=["United States"],
            institutions=["Argonne National Laboratory"],
            fetch_full_text=False,
            max_results=50,
        )
        self.df2 = pd.DataFrame(
            [
                {
                    "title": "Hydrometallurgical extraction of cobalt and lithium",
                    "year": 2021,
                    "doi": "10.1016/j.jpowsour.2021.2001",
                    "authors": ["Johnson, M.", "Davis, R."],
                    "affiliations": ["Argonne National Laboratory, United States"],
                    "countries": ["United States"],
                    "institutions": ["Argonne National Laboratory"],
                    "category": "Academia",
                    "geo_category": "Non-EU/EEC",
                    "text_source": "Abstract",
                    "text_status_detail": "Scopus Abstract (Bypassed)",
                    "kw_count_lithium": 3,
                    "kw_count_cobalt": 2,
                    "target_keywords_present": True,
                }
            ]
        )
        self.csv2 = self.df2.to_csv(index=False)
        self.ris2 = "TY  - JOUR\nTI  - Battery Paper 1\nER  -\n"
        self.prov2 = generate_provenance_manifest(self.spec2, self.df2, self.csv2, self.ris2, 1, 0.8)
        self.result2 = JobResult(
            job_spec=self.spec2,
            df=self.df2,
            csv_text=self.csv2,
            ris_text=self.ris2,
            provenance=self.prov2,
            provenance_json=json.dumps(self.prov2, indent=2),
            duration_seconds=0.8,
            total_retrieved=1,
            total_filtered=1,
        )

    def test_generate_prism_xml_validity(self) -> None:
        xml_str = generate_prism_xml([self.result1, self.result2], project_title="Comparative Bibliometrics Test")
        self.assertIsInstance(xml_str, str)
        self.assertTrue(xml_str.startswith('<?xml version="1.0" encoding="UTF-8"?>'))

        # Parse with xml.etree.ElementTree to ensure 100% compliant XML
        root = ET.fromstring(xml_str)
        ns = {"p": PRISM_XML_NAMESPACE}
        self.assertEqual(root.tag, f"{{{PRISM_XML_NAMESPACE}}}GraphPadPrismFile")
        self.assertEqual(root.attrib.get("PrismXMLVersion"), "5.00")

        # Verify Info sheet
        info = root.find("p:Info", ns)
        self.assertIsNotNone(info)
        info_title = info.find("p:Title", ns)
        self.assertIsNotNone(info_title)
        self.assertEqual(info_title.text, "Comparative Bibliometrics Test")

        # Verify TableSequence references
        table_seq = root.find("p:TableSequence", ns)
        self.assertIsNotNone(table_seq)
        refs = table_seq.findall("p:Ref", ns)
        self.assertGreaterEqual(len(refs), 7)

        # Verify all referenced tables exist
        tables = root.findall("p:Table", ns)
        self.assertEqual(len(tables), len(refs))

        # Check Table 1: Sectoral Affiliation Dynamics
        t1 = root.find(".//p:Table[@ID='Table0']", ns)
        self.assertIsNotNone(t1)
        self.assertEqual(t1.find("p:Title", ns).text, "Sectoral Affiliation Dynamics")
        self.assertEqual(t1.attrib.get("TableType"), "OneWay")

        # Check Table 2: Geopolitical Classification
        t2 = root.find(".//p:Table[@ID='Table1']", ns)
        self.assertIsNotNone(t2)
        self.assertEqual(t2.find("p:Title", ns).text, "Geopolitical Classification")
        self.assertEqual(t2.attrib.get("TableType"), "OneWay")

        # Check Table 3: Annual Publication Dynamics (XY)
        t3 = root.find(".//p:Table[@ID='Table2']", ns)
        self.assertIsNotNone(t3)
        self.assertEqual(t3.find("p:Title", ns).text, "Annual Publication Dynamics")
        self.assertEqual(t3.attrib.get("TableType"), "XY")
        self.assertEqual(t3.attrib.get("XFormat"), "numbers")
        self.assertIsNotNone(t3.find("p:XColumn", ns))

    def test_export_prism_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_project.pzfx")
            export_prism_file([self.result1, self.result2], out_file, "Disk Export Test")
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(os.path.getsize(out_file), 1000)

            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Disk Export Test", content)
            self.assertIn("GraphPadPrismFile", content)

    def test_build_batch_provenance_zip_with_prism_enabled(self) -> None:
        zip_bytes = build_batch_provenance_zip(
            [self.result1, self.result2],
            batch_metadata={"title": "Zip Test Project"},
            export_prism=True,
        )
        self.assertGreater(len(zip_bytes), 0)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("scopus_mass_export.pzfx", namelist)
            self.assertIn("batch_manifest.json", namelist)

            manifest = json.loads(zf.read("batch_manifest.json").decode("utf-8"))
            self.assertIn("consolidated_artifacts", manifest)
            self.assertIn("graphpad_prism_project", manifest["consolidated_artifacts"])
            prism_meta = manifest["consolidated_artifacts"]["graphpad_prism_project"]
            self.assertEqual(prism_meta["file_path"], "scopus_mass_export.pzfx")
            self.assertEqual(prism_meta["format"], "GraphPad Prism XML 5.00")
            self.assertTrue(len(prism_meta["sha256"]) == 64)

            # Check that the pzfx file inside ZIP is valid XML
            pzfx_bytes = zf.read("scopus_mass_export.pzfx")
            root = ET.fromstring(pzfx_bytes.decode("utf-8"))
            self.assertEqual(root.attrib.get("PrismXMLVersion"), "5.00")

    def test_build_batch_provenance_zip_with_prism_disabled(self) -> None:
        zip_bytes = build_batch_provenance_zip(
            [self.result1, self.result2],
            batch_metadata={"title": "No Prism Test"},
            export_prism=False,
        )
        self.assertGreater(len(zip_bytes), 0)

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            self.assertNotIn("scopus_mass_export.pzfx", namelist)

            manifest = json.loads(zf.read("batch_manifest.json").decode("utf-8"))
            self.assertNotIn("graphpad_prism_project", manifest.get("consolidated_artifacts", {}))

    def test_generate_prism_xml_empty_results(self) -> None:
        empty_spec = JobSpecification(
            job_id=3,
            job_name="Empty Job",
            slug="job_03_empty",
            query="TITLE(none)",
            keywords=[],
            countries=[],
            institutions=[],
            fetch_full_text=False,
            max_results=10,
        )
        empty_res = JobResult(
            job_spec=empty_spec,
            df=pd.DataFrame(),
            csv_text="",
            ris_text="",
            provenance={},
            provenance_json="{}",
            duration_seconds=0.1,
            total_retrieved=0,
            total_filtered=0,
        )
        xml_str = generate_prism_xml([empty_res])
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, f"{{{PRISM_XML_NAMESPACE}}}GraphPadPrismFile")


if __name__ == "__main__":
    unittest.main()
