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

        # None / empty / malformed
        insts, countries, details = _parse_institutions_and_countries(None, None)
        self.assertEqual(insts, [])
        self.assertEqual(countries, [])
        self.assertEqual(details, [])


if __name__ == "__main__":
    unittest.main()
