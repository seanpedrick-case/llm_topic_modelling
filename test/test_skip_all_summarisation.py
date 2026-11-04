"""Tests for SKIP_ALL_SUMMARISATION configuration and helpers."""

import unittest

import pandas as pd

from tools.config import SKIP_ALL_SUMMARISATION, do_summaries_from_config
from tools.dedup_summaries import _pass_through_summarisation_outputs
from tools.helper_functions import blank_summary_columns
from tools.prompts import initial_table_prompt, omit_per_row_summary_instruction


class TestSkipAllSummarisation(unittest.TestCase):
    def test_omit_per_row_summary_instruction(self):
        modified = omit_per_row_summary_instruction(initial_table_prompt)
        self.assertIn("leave every cell in that column empty", modified.lower())
        self.assertNotIn("write a summary of the subtopic", modified.lower())

    def test_blank_summary_columns(self):
        df = pd.DataFrame(
            {
                "General topic": ["A"],
                "Summary": ["Some text"],
                "Revised summary": ["Other"],
            }
        )
        out = blank_summary_columns(df)
        self.assertEqual(out["Summary"].iloc[0], "")
        self.assertEqual(out["Revised summary"].iloc[0], "")

    def test_do_summaries_from_config_matches_flag(self):
        expected = "No" if SKIP_ALL_SUMMARISATION else "Yes"
        self.assertEqual(do_summaries_from_config(), expected)

    def test_pass_through_summarisation_outputs(self):
        reference = pd.DataFrame(
            {
                "Response ID": ["1"],
                "General topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "Sentiment": ["Negative"],
                "Summary": ["should not appear"],
                "Group": ["All"],
            }
        )
        topics = pd.DataFrame(
            {
                "General topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "Sentiment": ["Negative"],
                "Group": ["All"],
                "Number of responses": [1],
                "Summary": ["ignored"],
            }
        )
        _, topic_out, ref_out, *_ = _pass_through_summarisation_outputs(
            reference, topics, reference, []
        )
        self.assertFalse(topic_out.empty)
        self.assertFalse(ref_out.empty)
        self.assertEqual(topic_out["Revised summary"].iloc[0], "")
        self.assertNotIn("Summary", ref_out.columns)


if __name__ == "__main__":
    unittest.main()
