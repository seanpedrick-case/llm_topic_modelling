"""Tests for topic-summary aggregation from reference tables."""

import time
import unittest

import pandas as pd

from tools.helper_functions import (
    convert_reference_table_to_pivot_table,
    create_topic_summary_df_from_reference_table,
    parse_topic_confidence_value,
)


class TestCreateTopicSummaryDfFromReferenceTable(unittest.TestCase):
    def test_summaries_sort_by_start_row_and_dedupe(self):
        reference_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs", "Repairs"],
                "Sentiment": ["Negative", "Negative", "Negative"],
                "Group": ["All", "All", "All"],
                "Response ID": [3, 1, 2],
                "Summary": ["Later issue", "First issue", "First issue"],
                "Start row of group": [11, 1, 6],
            }
        )

        out = create_topic_summary_df_from_reference_table(reference_df)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "Summary"], "First issue<br>Later issue")
        self.assertEqual(out.loc[0, "Number of responses"], 3)

    def test_prejoined_summary_segments_are_split(self):
        reference_df = pd.DataFrame(
            {
                "General topic": ["Transport", "Transport"],
                "Subtopic": ["Buses", "Buses"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Response ID": [1, 2],
                "Summary": ["Need more buses<br>Late evenings", "Late evenings"],
                "Start row of group": [1, 6],
            }
        )

        out = create_topic_summary_df_from_reference_table(reference_df)

        self.assertEqual(out.loc[0, "Summary"], "Need more buses<br>Late evenings")

    def test_without_sentiment_groups_by_topic_only(self):
        reference_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs"],
                "Sentiment": ["Negative", "Positive"],
                "Group": ["All", "All"],
                "Response ID": [1, 2],
                "Summary": ["Damp", "Fixed quickly"],
                "Start row of group": [1, 6],
            }
        )

        out = create_topic_summary_df_from_reference_table(
            reference_df, sentiment_checkbox="Do not assess sentiment"
        )

        self.assertEqual(len(out), 1)
        self.assertNotIn("Sentiment", out.columns)
        self.assertIn("Damp", out.loc[0, "Summary"])
        self.assertIn("Fixed quickly", out.loc[0, "Summary"])

    def test_large_reference_table_aggregates_quickly(self):
        n_topics = 40
        n_rows_per_topic = 80
        rows = []
        for topic_i in range(n_topics):
            for row_i in range(n_rows_per_topic):
                rows.append(
                    {
                        "General topic": f"Topic {topic_i}",
                        "Subtopic": f"Subtopic {topic_i}",
                        "Sentiment": "Negative",
                        "Group": "All",
                        "Response ID": row_i + 1,
                        "Summary": f"Issue {topic_i} batch {row_i} extra detail",
                        "Start row of group": row_i * 5 + 1,
                    }
                )
        reference_df = pd.DataFrame(rows)

        started = time.perf_counter()
        out = create_topic_summary_df_from_reference_table(reference_df)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(out), n_topics)
        self.assertLess(elapsed, 2.0)


class TestConvertReferenceTableToPivotTable(unittest.TestCase):
    def test_not_assessed_strip_does_not_create_duplicate_columns(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 2],
                "General topic": ["Not assessed", "Housing"],
                "Subtopic": ["Housing - Repairs", "Repairs"],
                "Sentiment": ["Not assessed", "Not assessed"],
            }
        )

        pivot = convert_reference_table_to_pivot_table(reference_df)

        self.assertFalse(pivot.columns.duplicated().any())
        pivot["Group"] = "All"
        self.assertEqual(pivot.loc[0, "Group"], "All")

    def test_unassessed_general_topic_is_omitted_from_headers(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 2],
                "General topic": ["Not Assessed", "Not assessed"],
                "Subtopic": ["Parking", "Buses"],
                "Sentiment": ["Negative", "Not assessed"],
            }
        )

        pivot = convert_reference_table_to_pivot_table(reference_df)
        headers = [str(col) for col in pivot.columns if col != "Response ID"]

        self.assertIn("Parking - Negative", headers)
        self.assertIn("Buses", headers)
        self.assertFalse(any("not assessed" in col.casefold() for col in headers))

    def test_confidence_replaces_presence_counts(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 1, 2],
                "General topic": ["Housing", "Transport", "Housing"],
                "Subtopic": ["Repairs", "Buses", "Repairs"],
                "Sentiment": ["Negative", "Negative", "Negative"],
                "Confidence": [0.9, 0.2, 0.55],
            }
        )
        basic = pd.DataFrame(
            {
                "Response ID": [1, 2, 3],
                "Original Response ID": [1, 2, 3],
                "Response": ["a", "b", "c"],
            }
        )

        pivot = convert_reference_table_to_pivot_table(
            reference_df, basic, include_confidence=True
        )

        housing_col = [
            col
            for col in pivot.columns
            if "Housing" in str(col) and "Repairs" in str(col)
        ][0]
        buses_col = [col for col in pivot.columns if "Buses" in str(col)][0]

        self.assertAlmostEqual(float(pivot.loc[0, housing_col]), 0.9)
        self.assertAlmostEqual(float(pivot.loc[0, buses_col]), 0.2)
        self.assertAlmostEqual(float(pivot.loc[1, housing_col]), 0.55)
        self.assertTrue(pd.isna(pivot.loc[2, housing_col]))
        self.assertEqual(int(pivot.loc[0, "All"]), 2)
        self.assertEqual(int(pivot.loc[1, "All"]), 1)

    def test_without_confidence_still_uses_counts(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 2],
                "General topic": ["Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs"],
                "Sentiment": ["Negative", "Negative"],
            }
        )

        pivot = convert_reference_table_to_pivot_table(
            reference_df, include_confidence=False
        )
        topic_col = [col for col in pivot.columns if col not in {"Response ID", "All"}][
            0
        ]
        self.assertEqual(int(pivot.loc[0, topic_col]), 1)


class TestParseTopicConfidenceValue(unittest.TestCase):
    def test_zero_to_one_and_percentages(self):
        self.assertEqual(parse_topic_confidence_value("0.82"), 0.82)
        self.assertEqual(parse_topic_confidence_value("80%"), 0.8)
        self.assertEqual(parse_topic_confidence_value(75), 0.75)
        self.assertEqual(parse_topic_confidence_value("1"), 1.0)
        self.assertIsNone(parse_topic_confidence_value(""))
        self.assertIsNone(parse_topic_confidence_value("not sure"))
        self.assertEqual(parse_topic_confidence_value("-0.2"), 0.0)
        self.assertEqual(parse_topic_confidence_value("150"), 1.0)


class TestTopicSummaryConfidenceAggregation(unittest.TestCase):
    def test_mean_and_min_confidence_are_added(self):
        reference_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Response ID": [1, 2],
                "Summary": ["Damp", "Damp"],
                "Start row of group": [1, 6],
                "Confidence": [0.9, 0.4],
            }
        )

        out = create_topic_summary_df_from_reference_table(reference_df)

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.loc[0, "Mean confidence"]), 0.65)
        self.assertAlmostEqual(float(out.loc[0, "Min confidence"]), 0.4)

    def test_missing_confidence_values_do_not_raise(self):
        """Confidence column present but empty must not break summary aggregation."""
        reference_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Housing"],
                "Subtopic": ["Repairs", "Repairs"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Response ID": [1, 2],
                "Summary": ["Damp", "Damp"],
                "Start row of group": [1, 6],
                "Confidence": [None, ""],
            }
        )

        out = create_topic_summary_df_from_reference_table(reference_df)

        self.assertEqual(len(out), 1)
        self.assertIn("Mean confidence", out.columns)
        self.assertIn("Min confidence", out.columns)
        self.assertTrue(pd.isna(out.loc[0, "Mean confidence"]))
        self.assertTrue(pd.isna(out.loc[0, "Min confidence"]))


if __name__ == "__main__":
    unittest.main()
