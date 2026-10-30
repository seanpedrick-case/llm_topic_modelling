"""Bedrock guardrail batch status CSV helpers."""

import tempfile
import unittest

from tools.bedrock_guardrail_batch_status import (
    GUARDRAIL_BATCH_STATUS_COLUMNS,
    append_guardrail_batch_status,
    clear_guardrail_batch_status,
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


if __name__ == "__main__":
    unittest.main()
