"""
Unit tests for network_analyzer.py.
"""

import unittest
import networkx as nx
import pandas as pd

from network_analyzer import (
    build_citation_graph,
    build_network_plotly_figure,
    build_descriptive_frequency_figures,
)


class TestNetworkAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "eid": "2-s2.0-001",
                    "title": "Fundamental Study on SSbD Materials",
                    "year": 2022,
                    "category": "Academia",
                    "geo_category": "EU/EEC",
                    "authors": ["Dupont, Marie", "Smith, John"],
                    "institutions": ["CNRS", "Sorbonne"],
                    "doi": "10.1000/1",
                },
                {
                    "eid": "2-s2.0-002",
                    "title": "Industrial Applications of Safe Design",
                    "year": 2023,
                    "category": "Industry",
                    "geo_category": "Non-EU/EEC",
                    "authors": ["Smith, John", "Tanaka, Ken"],
                    "institutions": ["BASF SE"],
                    "doi": "10.1000/2",
                },
                {
                    "eid": "2-s2.0-003",
                    "title": "Comprehensive Review of Advanced Materials",
                    "year": 2024,
                    "category": "Mixed",
                    "geo_category": "Mixed Geo",
                    "authors": ["Dupont, Marie", "Müller, Hans"],
                    "institutions": ["CNRS", "Fraunhofer"],
                    "doi": "10.1000/3",
                },
                {
                    "eid": "2-s2.0-004",
                    "title": "Isolated Emerging Methodology",
                    "year": 2024,
                    "category": "Academia",
                    "geo_category": "EU/EEC",
                    "authors": ["Rossi, Elena"],
                    "institutions": ["Politecnico di Milano"],
                    "doi": "10.1000/4",
                },
            ]
        )
        # Edges: 003 cites 001 and 002; 002 cites 001. 004 is isolated (0 citations).
        self.edges = [
            ("2-s2.0-003", "2-s2.0-001"),
            ("2-s2.0-003", "2-s2.0-002"),
            ("2-s2.0-002", "2-s2.0-001"),
        ]

    def test_build_citation_graph(self):
        G, df_metrics = build_citation_graph(self.df, self.edges)

        # Node count must equal total papers (preserving isolated nodes)
        self.assertEqual(len(G.nodes()), 4)
        self.assertEqual(len(G.edges()), 3)

        # Check metric columns present
        self.assertIn("in_degree", df_metrics.columns)
        self.assertIn("out_degree", df_metrics.columns)
        self.assertIn("betweenness_centrality", df_metrics.columns)
        self.assertIn("pagerank", df_metrics.columns)

        # Paper 001 has in_degree 2 (cited by 003 and 002)
        row_001 = df_metrics[df_metrics["eid"] == "2-s2.0-001"].iloc[0]
        self.assertEqual(row_001["in_degree"], 2)
        self.assertEqual(row_001["out_degree"], 0)

        # Paper 004 is isolated (in=0, out=0)
        row_004 = df_metrics[df_metrics["eid"] == "2-s2.0-004"].iloc[0]
        self.assertEqual(row_004["in_degree"], 0)
        self.assertEqual(row_004["out_degree"], 0)

    def test_build_network_plotly_figure(self):
        G, df_metrics = build_citation_graph(self.df, self.edges)

        # Test interactive color figure
        fig_color = build_network_plotly_figure(G, df_metrics, monochrome=False)
        self.assertIsNotNone(fig_color)
        self.assertGreater(len(fig_color.data), 1)

        # Test monochrome figure
        fig_mono = build_network_plotly_figure(G, df_metrics, monochrome=True)
        self.assertIsNotNone(fig_mono)
        self.assertEqual(len(fig_mono.data), 2)  # 1 edge trace + 1 node trace

    def test_build_descriptive_frequency_figures(self):
        fig_authors, fig_insts = build_descriptive_frequency_figures(
            self.df, top_n=10, monochrome=False
        )
        self.assertIsNotNone(fig_authors)
        self.assertIsNotNone(fig_insts)

        # Test monochrome mode
        fig_authors_m, fig_insts_m = build_descriptive_frequency_figures(
            self.df, top_n=10, monochrome=True
        )
        self.assertIsNotNone(fig_authors_m)
        self.assertIsNotNone(fig_insts_m)


if __name__ == "__main__":
    unittest.main()
