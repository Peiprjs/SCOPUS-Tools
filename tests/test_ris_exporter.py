"""
Unit tests for ris_exporter.py.
"""

import unittest
from unittest.mock import patch
import pandas as pd
import rispy

from ris_exporter import (
    row_to_ris_entry,
    dataframe_to_ris_entries,
    export_ris_string,
    process_query_batch,
)


class TestRisExporter(unittest.TestCase):
    def test_row_to_ris_entry_complete(self):
        row = {
            "title": "Machine Learning for Materials Science",
            "year": 2024,
            "doi": "https://doi.org/10.1016/j.matsci.2024.01",
            "abstract": "A comprehensive survey of machine learning techniques in materials discovery.",
            "eid": "2-s2.0-85123456789",
            "affiliations": ["MIT", "Stanford University"],
            "category": "Academia",
        }
        entry = row_to_ris_entry(row)
        self.assertEqual(entry["type_of_reference"], "JOUR")
        self.assertEqual(entry["primary_title"], "Machine Learning for Materials Science")
        self.assertEqual(entry["publication_year"], "2024")
        self.assertEqual(entry["doi"], "10.1016/j.matsci.2024.01")
        self.assertEqual(entry["accession_number"], "2-s2.0-85123456789")
        self.assertEqual(entry["custom1"], "Affiliation: Academia")
        self.assertIn("MIT; Stanford University", entry["notes"][0])

    def test_dataframe_to_ris_entries_and_export(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Paper One",
                    "year": 2023,
                    "doi": "10.1000/1",
                    "abstract": "First abstract.",
                    "eid": "eid-1",
                    "affiliations": ["Harvard"],
                    "category": "Academia",
                },
                {
                    "title": "Paper Two",
                    "year": 2022,
                    "doi": "10.1000/2",
                    "abstract": "Second abstract.",
                    "eid": "eid-2",
                    "affiliations": ["Google LLC"],
                    "category": "Industry",
                },
            ]
        )
        entries = dataframe_to_ris_entries(df)
        self.assertEqual(len(entries), 2)

        ris_text = export_ris_string(entries)
        self.assertIn("TY  - JOUR", ris_text)
        self.assertIn("T1  - Paper One", ris_text)
        self.assertIn("T1  - Paper Two", ris_text)
        self.assertIn("DO  - 10.1000/1", ris_text)
        self.assertIn("ER  - ", ris_text)

        # Verify parsed back via rispy
        parsed = rispy.loads(ris_text)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["primary_title"], "Paper One")
        self.assertEqual(parsed[1]["primary_title"], "Paper Two")

    def test_empty_dataframe(self):
        self.assertEqual(dataframe_to_ris_entries(pd.DataFrame()), [])
        self.assertEqual(export_ris_string([]), "")

    @patch("ris_exporter.search_scopus")
    def test_process_query_batch(self, mock_search):
        mock_df = pd.DataFrame(
            [
                {
                    "title": "Batch Paper",
                    "year": 2025,
                    "doi": "10.1000/batch",
                    "abstract": "Batch abstract.",
                    "eid": "eid-batch",
                    "affiliations": ["CNRS"],
                    "category": "Academia",
                }
            ]
        )
        mock_search.return_value = mock_df

        queries = ["TITLE(quantum)", "TITLE(nanomaterials)"]
        ris_text, q_count, ref_count, summaries = process_query_batch(queries)

        self.assertEqual(q_count, 2)
        # Deduplication check: same eid-batch should be included once
        self.assertEqual(ref_count, 1)
        self.assertIn("T1  - Batch Paper", ris_text)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["status"], "Success")


if __name__ == "__main__":
    unittest.main()
