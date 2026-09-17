"""Tests for topic-summary aggregation from reference tables."""

import time
import unittest

import pandas as pd

from tools.helper_functions import (
    convert_reference_table_to_pivot_table,
    create_topic_summary_df_from_reference_table,
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


if __name__ == "__main__":
    unittest.main()
