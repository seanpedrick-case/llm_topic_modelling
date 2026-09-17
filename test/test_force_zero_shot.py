import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from tools.helper_functions import (
    effective_force_zero_shot_radio,
    generate_zero_shot_topics_df,
    has_submitted_candidate_topics,
)


class TestHasSubmittedCandidateTopics(unittest.TestCase):
    def test_none_and_empty_values_are_false(self):
        self.assertFalse(has_submitted_candidate_topics(None))
        self.assertFalse(has_submitted_candidate_topics(""))
        self.assertFalse(has_submitted_candidate_topics("   "))
        self.assertFalse(has_submitted_candidate_topics([]))
        self.assertFalse(has_submitted_candidate_topics([""]))
        self.assertFalse(has_submitted_candidate_topics(SimpleNamespace(name="")))
        self.assertFalse(has_submitted_candidate_topics(SimpleNamespace(name="  ")))

    def test_submitted_file_or_list_is_true(self):
        self.assertTrue(has_submitted_candidate_topics("topics.csv"))
        self.assertTrue(
            has_submitted_candidate_topics(SimpleNamespace(name="topics.csv"))
        )
        self.assertTrue(has_submitted_candidate_topics(["topics.csv"]))
        self.assertTrue(
            has_submitted_candidate_topics([SimpleNamespace(name="topics.csv")])
        )


class TestEffectiveForceZeroShotRadio(unittest.TestCase):
    def test_yes_without_candidate_topics_is_ignored(self):
        self.assertEqual(effective_force_zero_shot_radio("Yes", None), "No")
        self.assertEqual(effective_force_zero_shot_radio("Yes", ""), "No")
        self.assertEqual(effective_force_zero_shot_radio("Yes", []), "No")

    def test_yes_with_candidate_topics_is_kept(self):
        self.assertEqual(
            effective_force_zero_shot_radio("Yes", "topics.csv"),
            "Yes",
        )
        self.assertEqual(
            effective_force_zero_shot_radio("Yes", SimpleNamespace(name="topics.csv")),
            "Yes",
        )

    def test_no_is_unchanged_even_with_candidate_topics(self):
        self.assertEqual(effective_force_zero_shot_radio("No", "topics.csv"), "No")
        self.assertEqual(effective_force_zero_shot_radio("No", None), "No")


class TestGenerateZeroShotTopicsKeepsForceFallback(unittest.TestCase):
    def test_no_relevant_topic_added_only_when_force_is_yes(self):
        topics = pd.DataFrame({"Subtopic": ["Housing"]})
        forced = generate_zero_shot_topics_df(
            topics.copy(), force_zero_shot_radio="Yes"
        )
        unforced = generate_zero_shot_topics_df(
            topics.copy(), force_zero_shot_radio="No"
        )

        self.assertIn("No relevant topic", forced["Subtopic"].tolist())
        self.assertNotIn("No relevant topic", unforced["Subtopic"].tolist())


if __name__ == "__main__":
    unittest.main()
