"""
Unit tests for network_analyzer.py.
"""

import unittest
import networkx as nx
import pandas as pd

from network_analyzer import (
    apply_adaptive_theme,
    build_citation_graph,
    build_coauthorship_graph,
    build_coauthorship_plotly_figure,
    build_coinstitution_graph,
    build_coinstitution_plotly_figure,
    build_descriptive_frequency_figures,
    build_network_plotly_figure,
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


    def test_build_coauthorship_graph(self):
        G, df_auth = build_coauthorship_graph(self.df)

        # Expected unique authors: Dupont, Marie; Smith, John; Tanaka, Ken; Müller, Hans; Rossi, Elena (5 total)
        self.assertEqual(len(G.nodes()), 5)
        self.assertIn("Dupont, Marie", G.nodes())
        self.assertIn("Smith, John", G.nodes())

        # Check edge: Dupont and Smith co-authored paper 001
        self.assertTrue(G.has_edge("Dupont, Marie", "Smith, John"))
        # Rossi is isolated (sole author on paper 004)
        self.assertEqual(G.degree("Rossi, Elena"), 0)

        # Check metrics DataFrame columns
        expected_cols = [
            "author", "publications", "collaborators_count",
            "collaboration_volume", "betweenness_centrality", "pagerank", "community"
        ]
        for col in expected_cols:
            self.assertIn(col, df_auth.columns)

        # Dupont authored 2 papers (001 and 003)
        row_dupont = df_auth[df_auth["author"] == "Dupont, Marie"].iloc[0]
        self.assertEqual(row_dupont["publications"], 2)
        self.assertEqual(row_dupont["collaborators_count"], 2)  # Smith, John and Müller, Hans

        # Test empty dataframe
        G_empty, df_empty = build_coauthorship_graph(pd.DataFrame())
        self.assertEqual(len(G_empty), 0)
        self.assertTrue(df_empty.empty)

    def test_build_coinstitution_graph(self):
        G, df_inst = build_coinstitution_graph(self.df)

        # Institutions: CNRS, Sorbonne, BASF SE, Fraunhofer, Politecnico di Milano
        self.assertEqual(len(G.nodes()), 5)
        self.assertIn("CNRS", G.nodes())
        self.assertIn("BASF SE", G.nodes())

        # Check sectoral classification
        self.assertEqual(G.nodes["CNRS"]["sector"], "Academia")
        self.assertEqual(G.nodes["BASF SE"]["sector"], "Industry")

        # Check edge: CNRS and Sorbonne co-affiliated on paper 001
        self.assertTrue(G.has_edge("CNRS", "Sorbonne"))
        # Check edge: CNRS and Fraunhofer on paper 003
        self.assertTrue(G.has_edge("CNRS", "Fraunhofer"))

        # Check metrics DataFrame columns
        expected_cols = [
            "institution", "sector", "publications", "partners_count",
            "collaboration_volume", "betweenness_centrality", "pagerank"
        ]
        for col in expected_cols:
            self.assertIn(col, df_inst.columns)

        row_cnrs = df_inst[df_inst["institution"] == "CNRS"].iloc[0]
        self.assertEqual(row_cnrs["publications"], 2)
        self.assertEqual(row_cnrs["partners_count"], 2)

        # Test empty dataframe
        G_empty, df_empty = build_coinstitution_graph(pd.DataFrame())
        self.assertEqual(len(G_empty), 0)
        self.assertTrue(df_empty.empty)

    def test_build_coauthorship_plotly_figure(self):
        G, df_auth = build_coauthorship_graph(self.df)

        fig_color = build_coauthorship_plotly_figure(G, df_auth, monochrome=False)
        self.assertIsNotNone(fig_color)
        self.assertGreater(len(fig_color.data), 1)

        fig_mono = build_coauthorship_plotly_figure(G, df_auth, monochrome=True)
        self.assertIsNotNone(fig_mono)
        self.assertEqual(len(fig_mono.data), 2)  # 1 edge trace + 1 node trace

        # Test empty graph
        fig_empty = build_coauthorship_plotly_figure(nx.Graph(), pd.DataFrame())
        self.assertIsNotNone(fig_empty)

    def test_build_coinstitution_plotly_figure(self):
        G, df_inst = build_coinstitution_graph(self.df)

        fig_color = build_coinstitution_plotly_figure(G, df_inst, monochrome=False)
        self.assertIsNotNone(fig_color)
        self.assertGreater(len(fig_color.data), 1)

        fig_mono = build_coinstitution_plotly_figure(G, df_inst, monochrome=True)
        self.assertIsNotNone(fig_mono)
        self.assertEqual(len(fig_mono.data), 2)

        # Test empty graph
        fig_empty = build_coinstitution_plotly_figure(nx.Graph(), pd.DataFrame())
        self.assertIsNotNone(fig_empty)

    def test_apply_adaptive_theme(self):
        G, df_metrics = build_citation_graph(self.df, self.edges)
        fig = build_network_plotly_figure(G, df_metrics, monochrome=False)

        # In interactive mode, background must be transparent to adapt to theme
        self.assertEqual(fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(fig.layout.plot_bgcolor, "rgba(0,0,0,0)")

        # In monochrome mode, background must be solid white for print
        fig_m = build_network_plotly_figure(G, df_metrics, monochrome=True)
        self.assertEqual(fig_m.layout.paper_bgcolor, "#ffffff")
        self.assertEqual(fig_m.layout.plot_bgcolor, "#ffffff")


if __name__ == "__main__":
    unittest.main()
