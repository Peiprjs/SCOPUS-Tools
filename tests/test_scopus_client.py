"""
Unit tests for scopus_client.py parsing helpers.
"""

import unittest
from scopus_client import _parse_year, _parse_affiliations


class TestScopusClientHelpers(unittest.TestCase):
    def test_parse_year_valid(self):
        self.assertEqual(_parse_year("2024-03-15"), 2024)
        self.assertEqual(_parse_year("1999-12-31"), 1999)
        self.assertEqual(_parse_year("2020"), 2020)

    def test_parse_year_invalid(self):
        self.assertIsNone(_parse_year(None))
        self.assertIsNone(_parse_year(""))
        self.assertIsNone(_parse_year("not-a-year"))
        self.assertIsNone(_parse_year(2023))

    def test_parse_affiliations_valid(self):
        raw = "Stanford University; Google LLC ; MIT "
        expected = ["Stanford University", "Google LLC", "MIT"]
        self.assertEqual(_parse_affiliations(raw), expected)

    def test_parse_affiliations_empty_or_invalid(self):
        self.assertEqual(_parse_affiliations(""), [])
        self.assertEqual(_parse_affiliations(" ; ; "), [])
        self.assertEqual(_parse_affiliations(None), [])
    def test_result_columns_schema(self):
        from scopus_client import _RESULT_COLUMNS
        expected = [
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
        self.assertEqual(_RESULT_COLUMNS, expected)

    def test_parse_institutions_and_countries_defensive(self):
        from scopus_client import _parse_institutions_and_countries

        # Normal balanced input
        insts, countries, details = _parse_institutions_and_countries(
            "MIT; Harvard University", "United States; United States"
        )
        self.assertEqual(insts, ["MIT", "Harvard University"])
        self.assertEqual(countries, ["United States"])
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0], {"institution": "MIT", "country": "United States"})

        # Mismatched lengths (3 insts, 1 country)
        insts, countries, details = _parse_institutions_and_countries(
            "Univ A; Univ B; Univ C", "Germany"
        )
        self.assertEqual(len(insts), 3)
        self.assertEqual(len(countries), 1)
        self.assertEqual(len(details), 3)
        self.assertEqual(details[0]["country"], "Germany")
        self.assertEqual(details[1]["country"], "Unknown Country")
        self.assertEqual(details[2]["country"], "Unknown Country")

    def test_format_eta(self):
        from scopus_client import format_eta

        self.assertEqual(format_eta(135), "02:15")
        self.assertEqual(format_eta(330), "05:30")
        self.assertEqual(format_eta(0), "00:00")
        self.assertEqual(format_eta(3665), "01:01:05")
        self.assertEqual(format_eta(None), "--:--")
        self.assertEqual(format_eta(-10), "--:--")

    def test_search_scopus_pagination_mock(self):
        from unittest.mock import patch, MagicMock
        from scopus_client import search_scopus

        chunk1_data = {
            "search-results": {
                "opensearch:totalResults": "3",
                "entry": [
                    {
                        "eid": "2-s2.0-001",
                        "dc:title": "Paper One",
                        "prism:coverDate": "2024-01-01",
                        "affiliation": [{"affilname": "Inst A", "affiliation-country": "Germany"}],
                        "author": [{"surname": "Smith", "given-name": "John"}],
                    },
                    {
                        "eid": "2-s2.0-002",
                        "dc:title": "Paper Two",
                        "prism:coverDate": "2023-01-01",
                        "affiliation": [{"affilname": "Inst B", "affiliation-country": "France"}],
                        "author": [{"surname": "Dupont", "given-name": "Jean"}],
                    },
                ],
            }
        }
        chunk2_data = {
            "search-results": {
                "opensearch:totalResults": "3",
                "entry": [
                    {
                        "eid": "2-s2.0-003",
                        "dc:title": "Paper Three",
                        "prism:coverDate": "2022-01-01",
                        "affiliation": [{"affilname": "Inst C", "affiliation-country": "Italy"}],
                        "author": [{"surname": "Rossi", "given-name": "Mario"}],
                    },
                ],
            }
        }

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = chunk1_data

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = chunk2_data

        callback_messages: list[str] = []

        def mock_callback(chunk_idx, total_chunks, curr, total, msg):
            callback_messages.append(msg)

        with patch("requests.get", side_effect=[resp1, resp2]):
            df = search_scopus(
                query="TITLE(test)",
                count=2,
                api_key="dummy_key",
                progress_callback=mock_callback,
            )

        self.assertEqual(len(df), 3)
        self.assertEqual(len(callback_messages), 2)
        self.assertIn("chunk 1 of 2", callback_messages[0])
        self.assertIn("ETA: ", callback_messages[0])
        self.assertIn("chunk 2 of 2", callback_messages[1])


if __name__ == "__main__":
    unittest.main()
