import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.helper_functions import (
    create_candidate_topics_df_from_topic_summary,
    subsample_responses_for_topic_discovery,
    write_candidate_topics_csv,
    write_topic_discovery_manifest_csv,
)


class TestSubsampleResponsesForTopicDiscovery(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Group": ["A"] * 5 + ["B"] * 5,
                "Response text": [f"response {i}" for i in range(10)],
            }
        )

    def test_global_sample_respects_fraction_and_seed(self):
        sampled_a, meta_a = subsample_responses_for_topic_discovery(
            self.df, sample_fraction=0.2, random_seed=42
        )
        sampled_b, meta_b = subsample_responses_for_topic_discovery(
            self.df, sample_fraction=0.2, random_seed=42
        )

        self.assertEqual(len(sampled_a), 2)
        self.assertEqual(meta_a["original_rows"], 10)
        self.assertEqual(meta_a["sampled_rows"], 2)
        pd.testing.assert_frame_equal(
            sampled_a.drop(columns=["_discovery_original_row_index"]),
            sampled_b.drop(columns=["_discovery_original_row_index"]),
        )

    def test_stratified_sample_includes_each_group(self):
        sampled, meta = subsample_responses_for_topic_discovery(
            self.df,
            sample_fraction=0.2,
            random_seed=7,
            group_col="Group",
        )

        self.assertGreaterEqual(len(sampled), 2)
        self.assertIn("A", sampled["Group"].values)
        self.assertIn("B", sampled["Group"].values)
        self.assertEqual(meta["per_group_counts"]["A"]["original"], 5)
        self.assertEqual(meta["per_group_counts"]["B"]["original"], 5)

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError):
            subsample_responses_for_topic_discovery(pd.DataFrame(), 0.2, 42)

    def test_invalid_fraction_raises(self):
        with self.assertRaises(ValueError):
            subsample_responses_for_topic_discovery(self.df, 0, 42)


class TestCandidateTopicsCsv(unittest.TestCase):
    def test_create_and_write_candidate_topics_csv(self):
        topic_summary_df = pd.DataFrame(
            {
                "Group": ["G1", "G1", "G2"],
                "Sentiment": ["Positive", "Negative", "Positive"],
                "General topic": ["Housing", "Housing", "Transport"],
                "Subtopic": ["Rent", "Rent", "Buses"],
            }
        )

        topics_df = create_candidate_topics_df_from_topic_summary(topic_summary_df)
        self.assertEqual(len(topics_df), 2)
        self.assertListEqual(list(topics_df.columns), ["General topic", "Subtopic"])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "topics.csv")
            written = write_candidate_topics_csv(topic_summary_df, csv_path)
            self.assertEqual(written, csv_path)
            loaded = pd.read_csv(csv_path)
            self.assertEqual(len(loaded), 2)

    def test_manifest_csv_lists_sampled_rows(self):
        sampled_df = pd.DataFrame(
            {
                "_discovery_original_row_index": [0, 4, 7],
                "Group": ["A", "A", "B"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.csv")
            written = write_topic_discovery_manifest_csv(
                sampled_df, manifest_path, group_col="Group"
            )
            self.assertEqual(written, manifest_path)
            loaded = pd.read_csv(manifest_path)
            self.assertListEqual(list(loaded.columns), ["Original row index", "Group"])
            self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
