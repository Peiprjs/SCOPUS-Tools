"""
Unit tests for fulltext_client.py with mocked HTTP responses.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from fulltext_client import clean_text, fetch_article_text, enrich_dataset_with_text


class TestFulltextClient(unittest.TestCase):
    def test_clean_text(self):
        xml_input = "<ce:para>This is a <b>substantive</b> scientific finding.</ce:para>"
        cleaned = clean_text(xml_input)
        self.assertEqual(cleaned, "This is a substantive scientific finding.")
        self.assertEqual(clean_text("   Regular   text   "), "Regular text")
        self.assertEqual(clean_text(None), "")

    def test_fetch_missing_doi_fallback(self):
        text, source, detail = fetch_article_text(
            doi=None,
            api_key="mock_key",
            abstract_fallback="This is the fallback abstract.",
        )
        self.assertEqual(text, "This is the fallback abstract.")
        self.assertEqual(source, "Abstract")
        self.assertEqual(detail, "Missing DOI")

    def test_fetch_missing_doi_and_abstract(self):
        text, source, detail = fetch_article_text(
            doi="",
            api_key="mock_key",
            abstract_fallback=None,
        )
        self.assertEqual(text, "")
        self.assertEqual(source, "None")
        self.assertIn("No Abstract Available", detail)

    @patch("requests.get")
    def test_fetch_article_success_200(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = (
            "This is a comprehensive full text article describing our findings in detail. "
            "It meets the minimum length requirement for substantive extraction. "
            "We evaluate deep learning algorithms across multiple benchmark datasets."
        )
        mock_get.return_value = mock_response

        text, source, detail = fetch_article_text(
            doi="10.1016/j.mock.2023.01",
            api_key="mock_key",
            abstract_fallback="Fallback abstract",
        )
        self.assertEqual(source, "Full Text")
        self.assertIn("comprehensive full text", text)
        self.assertEqual(detail, "Successfully Retrieved")

    @patch("requests.get")
    def test_fetch_article_paywalled_401_403(self, mock_get):
        for code in (401, 403):
            with self.subTest(status_code=code):
                mock_response = MagicMock()
                mock_response.status_code = code
                mock_get.return_value = mock_response

                text, source, detail = fetch_article_text(
                    doi="10.1016/j.cell.2023.01.001",
                    api_key="mock_key",
                    abstract_fallback="Paywalled paper abstract text.",
                )
                self.assertEqual(source, "Abstract")
                self.assertEqual(text, "Paywalled paper abstract text.")
                self.assertIn(str(code), detail)

    @patch("requests.get")
    def test_fetch_article_not_found_404(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        text, source, detail = fetch_article_text(
            doi="10.1109/NONELSEVIER.2023.01",
            api_key="mock_key",
            abstract_fallback="IEEE paper abstract.",
        )
        self.assertEqual(source, "Abstract")
        self.assertEqual(text, "IEEE paper abstract.")
        self.assertIn("404", detail)

    @patch("requests.get")
    def test_fetch_article_rate_limit_429(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        text, source, detail = fetch_article_text(
            doi="10.1016/j.cell.2023.01.001",
            api_key="mock_key",
            abstract_fallback="Rate limited paper abstract.",
        )
        self.assertEqual(source, "Abstract")
        self.assertEqual(text, "Rate limited paper abstract.")
        self.assertIn("429", detail)

    @patch("requests.get")
    def test_enrich_dataset_with_text(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 403  # Paywalled
        mock_get.return_value = mock_response

        df = pd.DataFrame(
            {
                "eid": ["2-s2.0-001", "2-s2.0-002"],
                "doi": ["10.1016/j.test.1", None],
                "abstract": ["Abstract for paper 1", "Abstract for paper 2"],
            }
        )
        res = enrich_dataset_with_text(df, api_key="test_key", max_fulltext=10)

        self.assertEqual(len(res), 2)
        self.assertEqual(res.loc[0, "text_source"], "Abstract")
        self.assertEqual(res.loc[0, "text"], "Abstract for paper 1")
        self.assertEqual(res.loc[1, "text_source"], "Abstract")
        self.assertEqual(res.loc[1, "text"], "Abstract for paper 2")

    @patch("requests.get")
    def test_enrich_dataset_with_text_bypassed(self, mock_get):
        df = pd.DataFrame(
            {
                "eid": ["2-s2.0-001"],
                "doi": ["10.1016/j.test.1"],
                "abstract": ["Abstract without full text."],
            }
        )
        res = enrich_dataset_with_text(
            df,
            api_key="test_key",
            max_fulltext=10,
            fetch_full_text=False,
        )

        mock_get.assert_not_called()
        self.assertEqual(res.loc[0, "text_source"], "Abstract")
        self.assertEqual(res.loc[0, "text"], "Abstract without full text.")
        self.assertIn("Bypassed", res.loc[0, "text_status_detail"])


if __name__ == "__main__":
    unittest.main()
