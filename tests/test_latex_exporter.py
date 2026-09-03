"""
Unit tests for latex_exporter.py.
"""

import io
import unittest
import zipfile
import pandas as pd

from latex_exporter import (
    escape_latex,
    build_monochrome_figures,
    generate_latex_document,
    create_latex_bundle,
)


class TestLatexExporter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "eid": "2-s2.0-001",
                    "title": "Study on Advanced Materials & Safety",
                    "year": 2023,
                    "category": "Academia",
                    "geo_category": "EU/EEC",
                    "text_source": "Full Text",
                    "kw_safety_present": True,
                    "kw_safety_count": 3,
                    "kw_nano_present": False,
                    "kw_nano_count": 0,
                },
                {
                    "eid": "2-s2.0-002",
                    "title": "Industrial Applications in 50% Yield",
                    "year": 2024,
                    "category": "Industry",
                    "geo_category": "Non-EU/EEC",
                    "text_source": "Abstract",
                    "kw_safety_present": False,
                    "kw_safety_count": 0,
                    "kw_nano_present": True,
                    "kw_nano_count": 2,
                },
            ]
        )
        self.kws = ["safety", "nano"]

    def test_escape_latex(self):
        self.assertEqual(escape_latex(""), "")
        self.assertEqual(escape_latex(None), "")
        raw = "Cost is $100 & 50% #1 _test_ {bracket} ^caret ~tilde \\slash"
        escaped = escape_latex(raw)
        self.assertIn(r"\$", escaped)
        self.assertIn(r"\&", escaped)
        self.assertIn(r"\%", escaped)
        self.assertIn(r"\#", escaped)
        self.assertIn(r"\_", escaped)
        self.assertIn(r"\{bracket\}", escaped)
        self.assertIn(r"\textasciicircum{}", escaped)
        self.assertIn(r"\textasciitilde{}", escaped)
        self.assertIn(r"\textbackslash{}", escaped)

    def test_build_monochrome_figures(self):
        figs = build_monochrome_figures(self.df, self.kws)
        self.assertIn("fig_affiliation_distribution", figs)
        self.assertIn("fig_institutional_trends", figs)
        self.assertIn("fig_geopolitical_correlation", figs)
        self.assertIn("fig_keyword_prevalence", figs)
        self.assertIn("fig_citation_network", figs)
        self.assertIn("fig_top_authors", figs)
        self.assertIn("fig_top_institutions", figs)

        # Check template
        self.assertEqual(figs["fig_affiliation_distribution"].layout.template.layout.margin.l or 50, 50)

    def test_generate_latex_document(self):
        metadata = {
            "query": 'TITLE("Carbon Nanotubes & Safety")',
            "year_start": 2020,
            "year_end": 2024,
            "selected_countries": ["Germany", "France"],
            "selected_institutions": ["Max Planck"],
            "fulltext_enabled": True,
        }
        image_filenames = {
            "fig_affiliation_distribution": "fig_affiliation_distribution.png",
            "fig_institutional_trends": "fig_institutional_trends.png",
            "fig_geopolitical_correlation": "fig_geopolitical_correlation.png",
            "fig_keyword_prevalence": "fig_keyword_prevalence.png",
            "fig_citation_network": "fig_citation_network.png",
            "fig_top_authors": "fig_top_authors.png",
            "fig_top_institutions": "fig_top_institutions.png",
        }
        tex = generate_latex_document(self.df, self.kws, metadata, image_filenames)
        self.assertIn(r"\documentclass[11pt,a4paper]{article}", tex)
        self.assertIn(r"\begin{document}", tex)
        self.assertIn(r"\end{document}", tex)
        self.assertIn(r"Carbon Nanotubes \& Safety", tex)
        self.assertIn(r"\includegraphics", tex)
        self.assertIn("fig_affiliation_distribution.png", tex)
        self.assertIn("fig_citation_network.png", tex)
        self.assertIn("fig_top_authors.png", tex)
        self.assertIn("fig_top_institutions.png", tex)

    def test_create_latex_bundle(self):
        metadata = {
            "query": 'TITLE("Test")',
            "year_start": 2022,
            "year_end": 2024,
            "selected_countries": [],
            "selected_institutions": [],
            "fulltext_enabled": False,
        }
        zip_bytes = create_latex_bundle(self.df, self.kws, metadata)
        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 1000)

        # Verify ZIP contents
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            names = z.namelist()
            self.assertIn("report.tex", names)
            self.assertIn("fig_affiliation_distribution.png", names)
            self.assertIn("fig_institutional_trends.png", names)
            self.assertIn("fig_geopolitical_correlation.png", names)
            self.assertIn("fig_keyword_prevalence.png", names)
            self.assertIn("fig_citation_network.png", names)
            self.assertIn("fig_top_authors.png", names)
            self.assertIn("fig_top_institutions.png", names)


if __name__ == "__main__":
    unittest.main()
