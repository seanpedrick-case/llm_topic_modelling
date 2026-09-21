import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.helper_functions import (
    create_candidate_topics_df_from_improved_names,
    identify_pivot_id_column,
    identify_pivot_response_column,
    identify_pivot_topic_columns,
    load_topic_response_pivot,
    responses_assigned_to_topic_mask,
    sample_responses_for_topic,
    write_improved_topics_csv,
)
from tools.llm_api_call import _parse_improve_topic_name_response


class TestPivotColumnDetection(unittest.TestCase):
    def test_identify_response_and_topic_columns(self):
        columns = [
            "Original Response ID",
            "Response",
            "Parking - Congestion - Negative",
            "Housing - Affordability - Positive",
            "All",
        ]
        response_col = identify_pivot_response_column(columns)
        id_col = identify_pivot_id_column(columns)
        topic_cols = identify_pivot_topic_columns(columns, response_col, id_col)

        self.assertEqual(response_col, "Response")
        self.assertEqual(id_col, "Original Response ID")
        self.assertEqual(
            topic_cols,
            ["Parking - Congestion - Negative", "Housing - Affordability - Positive"],
        )

    def test_response_text_alias(self):
        columns = ["Response ID", "Response text", "Topic A", "All"]
        self.assertEqual(identify_pivot_response_column(columns), "Response text")
        self.assertEqual(identify_pivot_id_column(columns), "Response ID")


class TestPivotLoadAndSample(unittest.TestCase):
    def setUp(self):
        self.pivot_df = pd.DataFrame(
            {
                "Original Response ID": [101, 102, 103, 104],
                "Response": [
                    "Too many cars on the high street.",
                    "Need more affordable homes.",
                    "Traffic jams every morning.",
                    "Unrelated comment about parks.",
                ],
                "Parking congestion": [1, 0, 1.0, 0],
                "Affordable housing": [0, 0.9, 0, 0],
                "All": [1, 1, 1, 0],
            }
        )

    def test_assignment_mask(self):
        mask = responses_assigned_to_topic_mask(self.pivot_df["Parking congestion"])
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_sample_responses_for_topic(self):
        sample_df, assigned_count = sample_responses_for_topic(
            self.pivot_df,
            topic_column="Parking congestion",
            response_column="Response",
            sample_size=5,
            random_seed=42,
            id_column="Original Response ID",
        )
        self.assertEqual(assigned_count, 2)
        self.assertEqual(len(sample_df), 2)
        self.assertListEqual(list(sample_df.columns), ["Response ID", "Response"])
        self.assertTrue(set(sample_df["Response ID"]).issubset({"101", "103"}))

    def test_sample_empty_when_unassigned(self):
        empty_pivot = self.pivot_df.copy()
        empty_pivot["Unused topic"] = 0
        sample_df, assigned_count = sample_responses_for_topic(
            empty_pivot,
            topic_column="Unused topic",
            response_column="Response",
            sample_size=5,
            random_seed=1,
        )
        self.assertEqual(assigned_count, 0)
        self.assertTrue(sample_df.empty)

    def test_load_topic_response_pivot_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pivot.csv")
            self.pivot_df.to_csv(path, index=False)
            loaded, topic_cols, response_col, id_col = load_topic_response_pivot(path)
            self.assertEqual(response_col, "Response")
            self.assertEqual(id_col, "Original Response ID")
            self.assertIn("Parking congestion", topic_cols)
            self.assertNotIn("All", topic_cols)
            self.assertEqual(len(loaded), 4)

    def test_load_topic_response_pivot_prefers_named_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "analysis.xlsx")
            other = pd.DataFrame({"A": [1]})
            with pd.ExcelWriter(path) as writer:
                other.to_excel(writer, sheet_name="Cover sheet", index=False)
                self.pivot_df.to_excel(
                    writer, sheet_name="Topic response pivot table", index=False
                )
            loaded, topic_cols, response_col, id_col = load_topic_response_pivot(path)
            self.assertEqual(response_col, "Response")
            self.assertEqual(len(loaded), 4)
            self.assertIn("Affordable housing", topic_cols)


class TestImprovedTopicsCsv(unittest.TestCase):
    def test_accepted_and_rejected_rows(self):
        review_df = pd.DataFrame(
            {
                "Current topic": ["Park", "Housing - Aff"],
                "Suggested General topic": ["Transport", "Housing"],
                "Suggested Subtopic": [
                    "On Street Parking Congestion",
                    "Affordable Housing Supply",
                ],
                "Rationale": ["Cars mentioned", "Homes mentioned"],
                "Sample size": [2, 1],
                "Accept": ["Yes", "No"],
            }
        )
        out = create_candidate_topics_df_from_improved_names(review_df)
        self.assertEqual(len(out), 2)
        accepted = out[out["Subtopic"] == "On Street Parking Congestion"]
        self.assertEqual(accepted.iloc[0]["General topic"], "Transport")
        rejected = out[out["Subtopic"] == "Housing - Aff"]
        self.assertEqual(rejected.iloc[0]["General topic"], "")

    def test_write_improved_topics_csv(self):
        review_df = pd.DataFrame(
            {
                "Current topic": ["Park"],
                "Suggested General topic": ["Transport"],
                "Suggested Subtopic": ["Parking Congestion"],
                "Rationale": ["ok"],
                "Sample size": [3],
                "Accept": ["Yes"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            written = write_improved_topics_csv(review_df, path)
            self.assertEqual(written, path)
            loaded = pd.read_csv(path)
            self.assertListEqual(list(loaded.columns), ["General topic", "Subtopic"])
            self.assertEqual(loaded.iloc[0]["General topic"], "Transport")
            self.assertEqual(loaded.iloc[0]["Subtopic"], "Parking Congestion")


class TestParseImproveTopicNameResponse(unittest.TestCase):
    def test_parse_markdown_table(self):
        response_text = """| Current topic | Suggested General topic | Suggested Subtopic | Rationale |
| --- | --- | --- | --- |
| Park | Transport | Parking Congestion | Mentions cars and traffic |
"""
        with patch(
            "tools.llm_api_call.convert_response_text_to_dataframe"
        ) as mock_convert:
            mock_convert.return_value = (
                pd.DataFrame(
                    [
                        {
                            "Current topic": "Park",
                            "Suggested General topic": "Transport",
                            "Suggested Subtopic": "Parking Congestion",
                            "Rationale": "Mentions cars and traffic",
                        }
                    ]
                ),
                False,
            )
            general, subtopic, rationale = _parse_improve_topic_name_response(
                response_text, "Park"
            )
            self.assertEqual(general, "Transport")
            self.assertEqual(subtopic, "Parking Congestion")
            self.assertIn("cars", rationale.lower())


if __name__ == "__main__":
    unittest.main()
