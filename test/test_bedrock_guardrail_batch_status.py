"""Bedrock guardrail batch status CSV helpers."""

import tempfile
import unittest

import pandas as pd

from tools.bedrock_guardrail_batch_status import (
    GUARDRAIL_BATCH_STATUS_COLUMNS,
    GUARDRAIL_INTERVENTIONS_SHEET_NAME,
    append_guardrail_batch_status,
    clear_guardrail_batch_status,
    extract_guardrail_intervention_categories,
    load_guardrail_batch_status,
)


class TestBedrockGuardrailBatchStatus(unittest.TestCase):
    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = tmp + "/"
            append_guardrail_batch_status(
                folder,
                batch_number=2,
                start_row=9,
                end_row=18,
                task_type="Topic extraction",
            )
            df = load_guardrail_batch_status(folder)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["Batch"], 2)
            self.assertEqual(df.iloc[0]["Rows"], "Rows 10 to 19")
            self.assertEqual(df.iloc[0]["Status"], "Guardrail rejected")
            self.assertEqual(df.iloc[0]["Intervention categories"], "")

    def test_append_includes_intervention_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = tmp + "/"
            append_guardrail_batch_status(
                folder,
                batch_number=1,
                start_row=0,
                end_row=4,
                intervention_categories="Content (HATE); Topic (Council business)",
            )
            df = load_guardrail_batch_status(folder)
            self.assertEqual(
                df.iloc[0]["Intervention categories"],
                "Content (HATE); Topic (Council business)",
            )

    def test_summarize_uses_policy_units_when_filters_missing(self):
        from tools.bedrock_guardrail_batch_status import (
            summarize_guardrail_intervention_for_export,
        )

        api_response = {
            "trace": {
                "guardrail": {
                    "inputAssessment": {
                        "gr-1": {
                            "invocationMetrics": {
                                "usage": {
                                    "topicPolicyUnits": 2,
                                    "contentPolicyUnits": 0,
                                }
                            }
                        }
                    }
                }
            }
        }
        summary = summarize_guardrail_intervention_for_export(api_response)
        self.assertIn("Topic policy", summary)

    def test_extract_categories_from_converse_trace(self):
        api_response = {
            "stopReason": "guardrail_intervened",
            "trace": {
                "guardrail": {
                    "inputAssessment": {
                        "gr-1": {
                            "contentPolicy": {
                                "filters": [
                                    {
                                        "type": "HATE",
                                        "action": "BLOCKED",
                                    }
                                ]
                            },
                            "topicPolicy": {
                                "topics": [
                                    {
                                        "name": "Heavy metal",
                                        "type": "DENY",
                                        "action": "BLOCKED",
                                    }
                                ]
                            },
                        }
                    }
                }
            },
        }
        categories = extract_guardrail_intervention_categories(api_response)
        self.assertIn("Content (HATE)", categories)
        self.assertIn("Topic (Heavy metal)", categories)

    def test_clear_removes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = tmp + "/"
            append_guardrail_batch_status(
                folder, batch_number=1, start_row=0, end_row=4
            )
            clear_guardrail_batch_status(folder)
            df = load_guardrail_batch_status(folder)
            self.assertTrue(df.empty)
            self.assertEqual(list(df.columns), GUARDRAIL_BATCH_STATUS_COLUMNS)

    def test_reference_table_has_assignments_empty_schema_only(self):
        from tools.llm_api_call import _reference_table_has_topic_assignments

        empty_with_columns = pd.DataFrame(
            columns=[
                "Response ID",
                "General topic",
                "Subtopic",
                "Sentiment",
                "Summary",
            ]
        )
        self.assertFalse(_reference_table_has_topic_assignments(empty_with_columns))
        self.assertFalse(_reference_table_has_topic_assignments(None))

        with_rows = pd.DataFrame(
            [{"Response ID": 1, "General topic": "A", "Subtopic": "B"}]
        )
        self.assertTrue(_reference_table_has_topic_assignments(with_rows))

    def test_guardrail_no_assignments_ui_text_all_batches(self):
        from tools.llm_api_call import _guardrail_no_assignments_ui_text

        md, plain = _guardrail_no_assignments_ui_text(6, 6)
        self.assertIn("All 6 extraction batches were rejected", plain)
        self.assertIn("All 6 extraction batches were rejected", md)
        self.assertIn(GUARDRAIL_INTERVENTIONS_SHEET_NAME, md)

        md_partial, plain_partial = _guardrail_no_assignments_ui_text(2, 6)
        self.assertIn("2 batch(es)", plain_partial)
        self.assertNotIn("All 6 extraction batches", plain_partial)


if __name__ == "__main__":
    unittest.main()
