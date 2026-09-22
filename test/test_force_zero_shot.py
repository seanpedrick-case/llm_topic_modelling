import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from tools.helper_functions import (
    apply_forced_unassessed_general_topics,
    effective_force_zero_shot_radio,
    generate_zero_shot_topics_df,
    has_submitted_candidate_topics,
    order_topics_for_batch_prompt,
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


class TestForcedUnassessedGeneralTopics(unittest.TestCase):
    def test_invented_general_topics_are_overwritten_when_force_zero_shot(self):
        df = pd.DataFrame(
            {
                "General topic": ["Accessibility", "Not assessed", "Cost"],
                "Subtopic": ["Online only", "Cashless parking", "Higher charges"],
            }
        )
        out = apply_forced_unassessed_general_topics(df, "Yes")
        self.assertTrue((out["General topic"] == "Not assessed").all())
        self.assertEqual(
            out["Subtopic"].tolist(),
            ["Online only", "Cashless parking", "Higher charges"],
        )

    def test_general_topics_are_left_alone_when_force_zero_shot_is_no(self):
        df = pd.DataFrame(
            {
                "General topic": ["Accessibility", "Cost"],
                "Subtopic": ["Online only", "Higher charges"],
            }
        )
        out = apply_forced_unassessed_general_topics(df, "No")
        self.assertEqual(out["General topic"].tolist(), ["Accessibility", "Cost"])


class TestOrderTopicsForBatchPrompt(unittest.TestCase):
    def setUp(self):
        self.topics = pd.DataFrame(
            {
                "General topic": [f"Topic {i}" for i in range(12)],
                "Subtopic": [f"Subtopic {i}" for i in range(12)],
            }
        )

    def test_shuffle_false_leaves_order_unchanged(self):
        out = order_topics_for_batch_prompt(self.topics, shuffle=False)
        pd.testing.assert_frame_equal(out, self.topics)

    def test_shuffle_is_reproducible_with_seed(self):
        first = order_topics_for_batch_prompt(self.topics, shuffle=True, random_seed=42)
        second = order_topics_for_batch_prompt(
            self.topics, shuffle=True, random_seed=42
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            sorted(first["Subtopic"].tolist()),
            sorted(self.topics["Subtopic"].tolist()),
        )

    def test_different_seeds_can_change_order(self):
        first = order_topics_for_batch_prompt(self.topics, shuffle=True, random_seed=1)
        second = order_topics_for_batch_prompt(self.topics, shuffle=True, random_seed=2)
        self.assertEqual(
            sorted(first["Subtopic"].tolist()),
            sorted(second["Subtopic"].tolist()),
        )
        self.assertFalse(first["Subtopic"].tolist() == second["Subtopic"].tolist())

    def test_empty_dataframe_is_returned_unchanged(self):
        empty = pd.DataFrame(columns=["General topic", "Subtopic"])
        out = order_topics_for_batch_prompt(empty, shuffle=True, random_seed=42)
        self.assertTrue(out.empty)


class TestWriteLlmOutputForcesUnassessedGeneralTopic(unittest.TestCase):
    def test_invented_general_topic_is_replaced_in_reference_table(self):
        import os
        import tempfile

        from tools.llm_api_call import write_llm_output_and_logs

        batch_df = pd.DataFrame(
            {
                "Response ID": ["1"],
                "Response": ["The parking charges are too high."],
                "Original Response ID": [1],
            }
        )
        response_text = """| Placeholder | Subtopics | Sentiment | Summary |
|---|---|---|---|
| Accessibility | Online only | Negative | The respondent cannot pay by cash. |
"""
        with tempfile.TemporaryDirectory() as tmp:
            (
                _topic_path,
                _ref_path,
                _summary_path,
                _topic_df,
                reference_df,
                summary_df,
                _details,
                is_error,
                _incomplete,
            ) = write_llm_output_and_logs(
                response_text=response_text,
                whole_conversation=[],
                all_metadata_content=[],
                batch_file_path_details="force_unassessed",
                start_row=0,
                end_row=0,
                model_choice_clean="test-model",
                log_files_output_paths=[],
                existing_reference_df=pd.DataFrame(),
                existing_topics_df=pd.DataFrame(),
                batch_size_number=1,
                batch_basic_response_df=batch_df,
                group_name="All",
                produce_structured_summary_radio="No",
                output_folder=tmp + os.sep,
                force_zero_shot_radio="Yes",
            )
        self.assertFalse(is_error)
        self.assertFalse(reference_df.empty)
        self.assertTrue((reference_df["General topic"] == "Not assessed").all())
        self.assertEqual(reference_df.iloc[0]["Subtopic"], "Online only")
        self.assertTrue((summary_df["General topic"] == "Not assessed").all())


if __name__ == "__main__":
    unittest.main()
