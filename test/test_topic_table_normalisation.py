"""Unit tests for topic-analysis markdown table column normalisation."""

import unittest

import pandas as pd

from tools.llm_api_call import (
    TOPIC_TABLE_EXPECTED_COLS,
    _ensure_standard_topic_table_columns,
    _four_column_table_has_sentiment,
    reconstruct_markdown_table_from_reference_df,
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


class TestNoSentimentTableLayouts(unittest.TestCase):
    def test_named_four_column_preserves_response_ids(self):
        df = pd.DataFrame(
            {
                "General topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "Response ID": ["3, 4"],
                "Summary": ["Delayed repairs"],
            }
        )

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=False
        )

        self.assertEqual(out.loc[0, "Response ID"], "3, 4")
        self.assertEqual(out.loc[0, "Summary"], "Delayed repairs")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_unnamed_four_column_maps_third_column_as_response_id(self):
        df = pd.DataFrame([["Housing", "Repairs", "3, 4", "Delayed repairs"]])

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=False
        )

        self.assertEqual(out.loc[0, "Response ID"], "3, 4")
        self.assertEqual(out.loc[0, "Summary"], "Delayed repairs")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_five_column_trailing_extra_does_not_steal_response_id(self):
        df = pd.DataFrame(
            {
                "General topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "Response ID": ["3, 4"],
                "Summary": ["Delayed repairs"],
                "Extra": [""],
            }
        )

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=False
        )

        self.assertEqual(out.loc[0, "Response ID"], "3, 4")
        self.assertEqual(out.loc[0, "Summary"], "Delayed repairs")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_unnamed_five_column_trailing_extra_without_sentiment(self):
        df = pd.DataFrame([["Housing", "Repairs", "3, 4", "Delayed repairs", ""]])

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=False
        )

        self.assertEqual(out.loc[0, "Response ID"], "3, 4")
        self.assertEqual(out.loc[0, "Summary"], "Delayed repairs")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_three_column_summary_is_not_used_as_response_id(self):
        df = pd.DataFrame(
            {
                "General topic": ["Transport"],
                "Subtopic": ["Buses"],
                "Summary": ["Need more buses"],
            }
        )

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=False
        )

        self.assertEqual(out.loc[0, "Summary"], "Need more buses")
        self.assertEqual(out.loc[0, "Response ID"], "")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_sentiment_layout_still_maps_when_assessing_sentiment(self):
        df = pd.DataFrame([["Housing", "Repairs", "Negative", "3", "summary"]])

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=True
        )

        self.assertEqual(out.loc[0, "Sentiment"], "Negative")
        self.assertEqual(out.loc[0, "Response ID"], "3")
        self.assertEqual(out.loc[0, "Summary"], "summary")

    def test_three_column_placeholder_table_maps_ids_not_sentiment(self):
        """Force-zero-shot tables often have Placeholder, Subtopic, Response ID."""
        df = pd.DataFrame(
            {
                "Placeholder": ["Not assessed", "Not assessed"],
                "Subtopics": ["Housing repairs", "Buses"],
                "Response refs": ["1, 2", "3"],
            }
        )

        out = _ensure_standard_topic_table_columns(
            df, batch_size_number=5, assess_sentiment=True
        )

        self.assertEqual(out.loc[0, "Response ID"], "1, 2")
        self.assertEqual(out.loc[1, "Response ID"], "3")
        self.assertEqual(out.loc[0, "Subtopic"], "Housing repairs")
        self.assertEqual(out.loc[0, "Sentiment"], "Not assessed")

    def test_four_column_with_response_id_header_is_not_sentiment(self):
        df = pd.DataFrame(
            {
                "General topic": ["Housing"],
                "Subtopic": ["Repairs"],
                "Response ID": ["3, 4"],
                "Summary": ["Delayed repairs"],
            }
        )

        self.assertFalse(_four_column_table_has_sentiment(df, assess_sentiment=False))

    def test_four_column_numeric_ids_are_not_treated_as_sentiment(self):
        df = pd.DataFrame([["Housing", "Repairs", "1, 2", "Delayed repairs"]])

        self.assertFalse(_four_column_table_has_sentiment(df, assess_sentiment=False))

    def test_reconstruct_omits_sentiment_column_when_not_assessed(self):
        reference_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs"],
                "Sentiment": ["Not assessed", "Not assessed"],
                "Response ID": [1, 2],
                "Summary": ["Delayed repairs", "Delayed repairs"],
            }
        )

        markdown, out_df = reconstruct_markdown_table_from_reference_df(
            reference_df, sentiment_checkbox="Do not assess sentiment"
        )

        self.assertIn("| General topic | Subtopic | Response ID | Summary |", markdown)
        self.assertNotIn("| Sentiment |", markdown)
        self.assertEqual(out_df.loc[0, "Sentiment"], "Not assessed")
        self.assertIn("1", out_df.loc[0, "Response ID"])
        self.assertIn("2", out_df.loc[0, "Response ID"])


if __name__ == "__main__":
    unittest.main()
