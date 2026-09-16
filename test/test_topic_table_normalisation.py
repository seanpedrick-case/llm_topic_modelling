"""Unit tests for topic-analysis markdown table column normalisation."""

import unittest

import pandas as pd

from tools.llm_api_call import (
    TOPIC_TABLE_EXPECTED_COLS,
    _ensure_standard_topic_table_columns,
)


class TestEnsureStandardTopicTableColumns(unittest.TestCase):
    def test_six_column_table_with_name_collision_is_unique(self):
        """Positional rename used to create duplicate 'General topic' labels."""
        df = pd.DataFrame(
            {
                "Topic": ["Housing"],
                "Detail": ["Repairs delay"],
                "General topic": ["Services"],
                "Response ID": ["1"],
                "Summary": ["Delayed repairs"],
                "Extra": ["noise"],
            }
        )

        out = _ensure_standard_topic_table_columns(df, batch_size_number=1)

        self.assertEqual(list(out.columns), TOPIC_TABLE_EXPECTED_COLS)
        self.assertFalse(out.columns.duplicated().any())
        self.assertEqual(out.loc[0, "General topic"], "Services")
        self.assertEqual(out.loc[0, "Response ID"], "1")
        # Concat must not raise the historical reindexing error
        other = pd.DataFrame(
            {
                "General topic": ["Other"],
                "Subtopic": ["X"],
                "Sentiment": ["Not assessed"],
            }
        )
        combined = pd.concat(
            [out[["General topic", "Subtopic", "Sentiment"]], other],
            ignore_index=True,
        )
        self.assertEqual(len(combined), 2)

    def test_float_nan_general_topic_becomes_string(self):
        df = pd.DataFrame(
            {
                "General topic": [float("nan"), float("nan")],
                "Subtopic": ["A", "B"],
                "Sentiment": ["Neutral", "Positive"],
                "Response ID": [1.0, 2.0],
                "Summary": ["s1", "s2"],
            }
        )

        out = _ensure_standard_topic_table_columns(df, batch_size_number=5)

        self.assertEqual(out["General topic"].dtype, object)
        # .str accessor must work after normalisation
        stripped = out["General topic"].str.strip()
        self.assertTrue((stripped == "").all())
        self.assertEqual(out.loc[0, "Response ID"], "1.0")

    def test_empty_named_general_topic_prefers_positional_values(self):
        """Blank 'General topic' header should not block real topic text in col0."""
        df = pd.DataFrame(
            {
                "Topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "General topic": [float("nan")],
                "Response ID": ["1"],
                "Summary": ["Delayed repairs"],
                "Extra": ["noise"],
            }
        )

        out = _ensure_standard_topic_table_columns(df, batch_size_number=5)

        self.assertEqual(out.loc[0, "General topic"], "Housing")
        self.assertEqual(out.loc[0, "Subtopic"], "Repairs")
        self.assertFalse(out.columns.duplicated().any())

    def test_three_column_table_gets_defaults(self):
        df = pd.DataFrame(
            {
                "General topic": ["Transport"],
                "Subtopic": ["Buses"],
                "Summary": ["Need more buses"],
            }
        )

        out = _ensure_standard_topic_table_columns(df, batch_size_number=1)

        self.assertEqual(list(out.columns), TOPIC_TABLE_EXPECTED_COLS)
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")
        self.assertEqual(out.loc[0, "Response ID"], "1")
        self.assertEqual(out.loc[0, "General topic"], "Transport")

    def test_empty_dataframe(self):
        out = _ensure_standard_topic_table_columns(pd.DataFrame(), batch_size_number=5)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), TOPIC_TABLE_EXPECTED_COLS)

    def test_duplicate_input_columns_are_deduped(self):
        df = pd.DataFrame(
            [["Housing", "Repairs", "Negative", "1", "summary"]],
            columns=[
                "General topic",
                "Subtopic",
                "Sentiment",
                "Response ID",
                "Summary",
            ],
        )
        # Force duplicate labels the way a bad select used to
        df = pd.concat([df, df[["General topic"]]], axis=1)

        out = _ensure_standard_topic_table_columns(df, batch_size_number=5)

        self.assertEqual(list(out.columns), TOPIC_TABLE_EXPECTED_COLS)
        self.assertFalse(out.columns.duplicated().any())
        self.assertEqual(out.loc[0, "General topic"], "Housing")


if __name__ == "__main__":
    unittest.main()
