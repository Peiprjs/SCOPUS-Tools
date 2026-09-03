"""
Unit tests for classifier.py.
"""

import unittest
from classifier import classify_affiliation, classify_paper


class TestClassifier(unittest.TestCase):
    def test_classify_affiliation_academia(self):
        cases = [
            "Massachusetts Institute of Technology",
            "Stanford University",
            "Université Paris-Saclay",
            "Max Planck Institute for Informatics",
            "CNRS",
            "ETH Zurich",
            "Caltech",
            "NASA Jet Propulsion Laboratory",
            "Charité - Universitätsmedizin Berlin",
        ]
        for name in cases:
            with self.subTest(name=name):
                self.assertEqual(classify_affiliation(name), "Academia")

    def test_classify_affiliation_industry(self):
        cases = [
            "Google LLC",
            "Microsoft Corp.",
            "Pfizer Ltd",
            "Siemens AG",
            "BioNTech SE",
            "Novartis Pharma AG",
            "DeepMind Technologies Ltd",
        ]
        for name in cases:
            with self.subTest(name=name):
                self.assertEqual(classify_affiliation(name), "Industry")

    def test_classify_affiliation_boundary_cases(self):
        # "Aga Khan" should not trigger regex \bag\b
        self.assertEqual(classify_affiliation("Aga Khan Foundation"), "Unknown")
        # Dual keyword (e.g. University Hospital GmbH -> Academia)
        self.assertEqual(classify_affiliation("University Hospital GmbH"), "Academia")

    def test_classify_affiliation_empty_and_unknown(self):
        self.assertEqual(classify_affiliation(""), "Unknown")
        self.assertEqual(classify_affiliation("   "), "Unknown")
        self.assertEqual(classify_affiliation("Unknown Institute of Mystics"), "Unknown")

    def test_classify_paper(self):
        self.assertEqual(classify_paper([]), "Unknown")
        self.assertEqual(
            classify_paper(["Harvard University", "MIT"]), "Academia"
        )
        self.assertEqual(
            classify_paper(["Google LLC", "Apple Inc."]), "Industry"
        )
        self.assertEqual(
            classify_paper(["Harvard University", "Google LLC"]), "Mixed"
        )
        self.assertEqual(
            classify_paper(["Harvard University", "Unrecognized Body"]), "Academia"
        )
        self.assertEqual(
            classify_paper(["Google LLC", "Unrecognized Body"]), "Industry"
        )
        self.assertEqual(
            classify_paper(["Unrecognized Body", "Another Mystery Org"]), "Unknown"
        )

    def test_classify_geography(self):
        from classifier import classify_geography

        self.assertEqual(classify_geography([]), "Unknown Geo")
        self.assertEqual(classify_geography(None), "Unknown Geo")
        self.assertEqual(classify_geography(["", "   "]), "Unknown Geo")

        # Pure EU/EEC
        self.assertEqual(classify_geography(["Germany", "France", "Spain"]), "EU/EEC")
        self.assertEqual(classify_geography(["Norway"]), "EU/EEC")
        self.assertEqual(classify_geography(["The Netherlands", "Italy"]), "EU/EEC")

        # Pure Non-EU/EEC
        self.assertEqual(classify_geography(["United States", "Japan"]), "Non-EU/EEC")
        self.assertEqual(classify_geography(["China"]), "Non-EU/EEC")

        # Mixed Geo
        self.assertEqual(classify_geography(["Germany", "United States"]), "Mixed Geo")
        self.assertEqual(classify_geography(["Italy", "China", "Spain"]), "Mixed Geo")


if __name__ == "__main__":
    unittest.main()
