"""
Unit tests for keyword_matcher.py.
"""

import unittest
import pandas as pd
from keyword_matcher import (
    parse_keywords,
    count_keyword,
    match_keywords,
    add_keyword_features,
)


class TestKeywordMatcher(unittest.TestCase):
    def test_parse_keywords(self):
        raw = " neural network , Deep Learning, benchmark, NEURAL NETWORK,  "
        expected = ["neural network", "Deep Learning", "benchmark"]
        self.assertEqual(parse_keywords(raw), expected)
        self.assertEqual(parse_keywords(""), [])
        self.assertEqual(parse_keywords(None), [])

    def test_count_keyword(self):
        text = (
            "We propose a deep neural network for natural language processing. "
            "Our neural network achieves a new benchmark, outperforming previous neural network architectures."
        )
        self.assertEqual(count_keyword(text, "neural network"), 3)
        self.assertEqual(count_keyword(text, "benchmark"), 1)
        self.assertEqual(count_keyword(text, "Deep Neural Network"), 1)
        self.assertEqual(count_keyword(text, "convolutional"), 0)
        # Word boundary test: "net" should not match "network"
        self.assertEqual(count_keyword(text, "net"), 0)

    def test_count_keyword_empty_or_none(self):
        self.assertEqual(count_keyword("", "test"), 0)
        self.assertEqual(count_keyword(None, "test"), 0)
        self.assertEqual(count_keyword("some text", ""), 0)

    def test_match_keywords(self):
        text = "Transformer models and attention mechanisms are effective."
        kws = ["transformer", "attention", "reinforcement learning"]
        res = match_keywords(text, kws)
        self.assertEqual(res["transformer"], 1)
        self.assertEqual(res["attention"], 1)
        self.assertEqual(res["reinforcement learning"], 0)

    def test_add_keyword_features(self):
        df = pd.DataFrame(
            {
                "eid": ["1", "2"],
                "text": [
                    "Transformers are widely used in deep learning applications.",
                    "Traditional methods rely on rule-based systems.",
                ],
            }
        )
        kws = ["deep learning", "transformers", "quantum"]
        res = add_keyword_features(df, kws)

        self.assertTrue(res.loc[0, "kw_deep learning_present"])
        self.assertEqual(res.loc[0, "kw_deep learning_count"], 1)
        self.assertTrue(res.loc[0, "kw_transformers_present"])
        self.assertFalse(res.loc[0, "kw_quantum_present"])
        self.assertTrue(res.loc[0, "has_any_keyword"])
        self.assertCountEqual(
            res.loc[0, "matched_keywords"], ["deep learning", "transformers"]
        )
        self.assertEqual(res.loc[0, "total_keyword_hits"], 2)

        self.assertFalse(res.loc[1, "has_any_keyword"])
        self.assertEqual(res.loc[1, "total_keyword_hits"], 0)


if __name__ == "__main__":
    unittest.main()
