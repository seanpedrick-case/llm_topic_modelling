"""Tests for memory-conscious fuzzy topic deduplication."""

import unittest

import pandas as pd

from tools.dedup_summaries import deduplicate_topics


class TestDeduplicateTopicsMemory(unittest.TestCase):
    def _reference_df(self):
        return pd.DataFrame(
            {
                "Response ID": [1, 2, 3, 4],
                "General topic": ["Transport"] * 4,
                "Subtopic": [
                    "Parking availability",
                    "Parking availablity",
                    "Bus frequency",
                    "Parking availability",
                ],
                "Sentiment": ["Negative"] * 4,
                "Summary": [
                    "Hard to park",
                    "No spaces",
                    "Buses are rare",
                    "More spaces needed",
                ],
                "Start row of group": [1, 1, 1, 6],
                "Group": ["All"] * 4,
            }
        )

    def test_batch_dedup_without_source_file_merges_typos(self):
        reference_df = self._reference_df()
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Transport", "Transport"],
                "Subtopic": ["Parking availability", "Bus frequency"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Topic number": [1, 2],
            }
        )

        out_ref, out_topics, output_files, log_files, markdown = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            score_threshold=90,
            output_files="False",
            in_data_files=None,
        )

        self.assertEqual(output_files, [])
        self.assertEqual(markdown, "")
        parking_rows = out_ref[out_ref["Subtopic"].str.contains("Parking", case=False)]
        self.assertGreaterEqual(parking_rows["Subtopic"].nunique(), 1)
        self.assertEqual(parking_rows["Subtopic"].nunique(), 1)
        self.assertEqual(out_ref["Response ID"].nunique(), 4)

    def test_duplicate_topic_response_rows_collapse_summaries(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 1, 2],
                "General topic": ["Housing", "Housing", "Transport"],
                "Subtopic": ["Repairs", "Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative", "Negative"],
                "Summary": ["Damp in kitchen", "Broken boiler", "Need more buses"],
                "Start row of group": [1, 1, 6],
                "Group": ["All", "All", "All"],
            }
        )
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Transport"],
                "Subtopic": ["Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Topic number": [1, 2],
            }
        )

        out_ref, _, _, _, _ = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            output_files="False",
            in_data_files=None,
        )

        repair_rows = out_ref[
            (out_ref["Response ID"] == 1) & (out_ref["Subtopic"] == "Repairs")
        ]
        self.assertEqual(len(repair_rows), 1)
        summary = repair_rows.iloc[0]["Summary"]
        self.assertIn("Damp in kitchen", summary)
        self.assertIn("Broken boiler", summary)

    def test_confidence_column_kept_when_no_topic_remaps(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 2],
                "General topic": ["Housing", "Transport"],
                "Subtopic": ["Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative"],
                "Summary": ["Damp in kitchen", "Need more buses"],
                "Start row of group": [1, 6],
                "Group": ["All", "All"],
                "Confidence": [0.82, 0.31],
            }
        )
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Transport"],
                "Subtopic": ["Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Topic number": [1, 2],
            }
        )

        out_ref, _, _, _, _ = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            output_files="False",
            in_data_files=None,
        )

        self.assertIn("Confidence", out_ref.columns)
        repairs = out_ref[out_ref["Subtopic"] == "Repairs"]
        buses = out_ref[out_ref["Subtopic"] == "Buses"]
        self.assertAlmostEqual(float(repairs.iloc[0]["Confidence"]), 0.82)
        self.assertAlmostEqual(float(buses.iloc[0]["Confidence"]), 0.31)

    def test_confidence_column_survives_row_collapse(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 1, 2],
                "General topic": ["Housing", "Housing", "Transport"],
                "Subtopic": ["Repairs", "Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative", "Negative"],
                "Summary": ["Damp in kitchen", "Broken boiler", "Need more buses"],
                "Start row of group": [1, 1, 6],
                "Group": ["All", "All", "All"],
                "Confidence": [0.4, 0.9, 0.75],
            }
        )
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Housing", "Transport"],
                "Subtopic": ["Repairs", "Buses"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Topic number": [1, 2],
            }
        )

        out_ref, _, _, _, _ = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            output_files="False",
            in_data_files=None,
        )

        self.assertIn("Confidence", out_ref.columns)
        repair_rows = out_ref[
            (out_ref["Response ID"] == 1) & (out_ref["Subtopic"] == "Repairs")
        ]
        self.assertEqual(len(repair_rows), 1)
        self.assertAlmostEqual(float(repair_rows.iloc[0]["Confidence"]), 0.9)


class TestLightNearDuplicateTopicMerge(unittest.TestCase):
    def _run(self, subtopics, threshold=92):
        reference_df = pd.DataFrame(
            {
                "Response ID": list(range(1, len(subtopics) + 1)),
                "General topic": ["Housing"] * len(subtopics),
                "Subtopic": subtopics,
                "Sentiment": ["Negative"] * len(subtopics),
                "Summary": [f"Summary {i}" for i in range(len(subtopics))],
                "Start row of group": [1] * len(subtopics),
                "Group": ["All"] * len(subtopics),
            }
        )
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Housing"] * len(set(subtopics)),
                "Subtopic": list(dict.fromkeys(subtopics)),
                "Sentiment": ["Negative"] * len(set(subtopics)),
                "Group": ["All"] * len(set(subtopics)),
                "Topic number": list(range(1, len(set(subtopics)) + 1)),
            }
        )
        out_ref, _, _, _, _ = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            score_threshold=threshold,
            merge_general_topics="No",
            output_files="False",
            in_data_files=None,
        )
        return out_ref

    def test_case_and_apostrophe_variants_merge(self):
        out_ref = self._run(
            ["Councils housing", "Council's Housing", "COUNCIL'S housing"]
        )
        self.assertEqual(out_ref["Subtopic"].nunique(), 1)

    def test_title_case_and_sentence_case_variants_merge(self):
        out_ref = self._run(
            [
                "General Disagree",
                "General disagree",
                "Impact On Businesses",
                "Impact on businesses",
                "Unaware Of Business Or Carer Or Health Permits",
                "Unaware of business or carer or health permits",
            ]
        )
        names = sorted(out_ref["Subtopic"].str.lower().unique())
        self.assertEqual(len(names), 3)
        self.assertEqual(
            names,
            [
                "general disagree",
                "impact on businesses",
                "unaware of business or carer or health permits",
            ],
        )

    def test_need_vs_needing_near_duplicate_merges(self):
        out_ref = self._run(
            ["Penalty For Need A Car", "Penalty for needing a car"],
            threshold=92,
        )
        self.assertEqual(out_ref["Subtopic"].nunique(), 1)

    def test_case_variants_merge_across_general_topic_casing(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [1, 2],
                "General topic": ["Not Assessed", "Not assessed"],
                "Subtopic": ["Impact On Businesses", "Impact on businesses"],
                "Sentiment": ["Negative", "Negative"],
                "Summary": ["s1", "s2"],
                "Start row of group": [1, 1],
                "Group": ["All", "All"],
            }
        )
        topic_summary_df = pd.DataFrame(
            {
                "General topic": ["Not Assessed", "Not assessed"],
                "Subtopic": ["Impact On Businesses", "Impact on businesses"],
                "Sentiment": ["Negative", "Negative"],
                "Group": ["All", "All"],
                "Topic number": [1, 2],
            }
        )
        out_ref, out_topics, _, _, _ = deduplicate_topics(
            reference_df=reference_df,
            topic_summary_df=topic_summary_df,
            reference_table_file_name="test_ref",
            unique_topics_table_file_name="test_topics",
            score_threshold=92,
            merge_general_topics="No",
            output_files="False",
            in_data_files=None,
        )
        self.assertEqual(out_ref["Subtopic"].nunique(), 1)
        self.assertEqual(out_ref["General topic"].nunique(), 1)
        self.assertEqual(out_topics["Subtopic"].nunique(), 1)

    def test_light_plural_variants_merge(self):
        out_ref = self._run(["Repair", "Repairs"])
        self.assertEqual(out_ref["Subtopic"].nunique(), 1)

    def test_distinct_topics_are_not_merged(self):
        out_ref = self._run(["Housing repairs", "Housing wait times"])
        self.assertEqual(out_ref["Subtopic"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
