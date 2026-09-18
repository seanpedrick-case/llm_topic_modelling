"""Remainder batches (last batch of 1) must still map topics to that response."""

import os
import tempfile
import unittest

import pandas as pd

from tools.llm_api_call import (
    _effective_parse_batch_size,
    data_file_to_markdown_table,
    reconstruct_markdown_table_from_reference_df,
    write_llm_output_and_logs,
)


def _consultation_df(n_rows: int = 41) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Response": [
                f"This is a substantial consultation response number {i} "
                "about parking charges, housing repairs, and local services."
                for i in range(1, n_rows + 1)
            ]
        }
    )


class TestRemainderBatchSlicing(unittest.TestCase):
    def test_effective_parse_size_uses_actual_row_count(self):
        batch_df = pd.DataFrame({"Response ID": ["1"], "Original Response ID": [41]})
        self.assertEqual(_effective_parse_batch_size(batch_df, 10), 1)
        self.assertEqual(_effective_parse_batch_size(pd.DataFrame(), 10), 10)

    def test_last_of_41_rows_is_a_batch_of_one(self):
        file_data = _consultation_df(41)
        _, markdown, start_row, end_row, batch_df = data_file_to_markdown_table(
            file_data, "consultation.csv", ["Response"], batch_number=4, batch_size=10
        )

        self.assertEqual(start_row, 40)
        self.assertEqual(end_row, 40)
        self.assertEqual(len(batch_df), 1)
        self.assertEqual(str(batch_df.iloc[0]["Response ID"]), "1")
        self.assertEqual(int(batch_df.iloc[0]["Original Response ID"]), 41)
        self.assertIn("response number 41", markdown)

    def test_all_41_rows_are_covered_once(self):
        file_data = _consultation_df(41)
        original_ids = []
        for batch_number in range(5):
            _, _, _, _, batch_df = data_file_to_markdown_table(
                file_data,
                "consultation.csv",
                ["Response"],
                batch_number=batch_number,
                batch_size=10,
            )
            original_ids.extend(batch_df["Original Response ID"].astype(int).tolist())

        self.assertEqual(original_ids, list(range(1, 42)))


class TestRemainderBatchOutputMapping(unittest.TestCase):
    def _one_row_batch_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Response ID": ["1"],
                "Response": ["The parking charges are too high and unfair."],
                "Original Response ID": [41],
            }
        )

    def _parse_table(self, response_text: str) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            output_folder = tmp + os.sep
            (
                _topic_path,
                _ref_path,
                _summary_path,
                _topic_df,
                reference_df,
                _summary_df,
                _details,
                is_error,
                _incomplete,
            ) = write_llm_output_and_logs(
                response_text=response_text,
                whole_conversation=[],
                all_metadata_content=[],
                batch_file_path_details="remainder_batch",
                start_row=40,
                end_row=40,
                model_choice_clean="test-model",
                log_files_output_paths=[],
                existing_reference_df=pd.DataFrame(),
                existing_topics_df=pd.DataFrame(),
                batch_size_number=10,
                batch_basic_response_df=self._one_row_batch_df(),
                group_name="All",
                produce_structured_summary_radio="No",
                output_folder=output_folder,
            )
        self.assertFalse(is_error)
        return reference_df

    def test_four_column_table_without_response_id_keeps_last_response(self):
        reference_df = self._parse_table(
            """| General topic | Subtopic | Sentiment | Summary |
|---|---|---|---|
| Transport | Parking charges | Negative | The respondent objects to higher parking fees. |
"""
        )
        self.assertFalse(reference_df.empty)
        self.assertEqual(int(reference_df.iloc[0]["Response ID"]), 41)
        self.assertEqual(reference_df.iloc[0]["Subtopic"], "Parking charges")

    def test_five_column_table_with_blank_response_id_keeps_last_response(self):
        reference_df = self._parse_table(
            """| General topic | Subtopic | Sentiment | Response ID | Summary |
|---|---|---|---|---|
| Transport | Parking charges | Negative |  | The respondent objects to higher parking fees. |
"""
        )
        self.assertFalse(reference_df.empty)
        self.assertEqual(int(reference_df.iloc[0]["Response ID"]), 41)

    def test_placeholder_table_without_response_id_keeps_last_response(self):
        reference_df = self._parse_table(
            """| Placeholder | Subtopics | Sentiment | Summary |
|---|---|---|---|
| Not assessed | Parking charges | Negative | The respondent objects to higher parking fees. |
"""
        )
        self.assertFalse(reference_df.empty)
        self.assertEqual(int(reference_df.iloc[0]["Response ID"]), 41)

    def test_reconstruct_includes_last_response_with_inclusive_end_row(self):
        reference_df = pd.DataFrame(
            {
                "Response ID": [41],
                "General topic": ["Transport"],
                "Subtopic": ["Parking charges"],
                "Sentiment": ["Negative"],
                "Summary": ["The respondent objects to higher parking fees."],
            }
        )
        markdown, reconstructed = reconstruct_markdown_table_from_reference_df(
            reference_df, start_row=40, end_row=40
        )
        self.assertFalse(reconstructed.empty)
        self.assertIn("Parking charges", markdown)


class TestResponseIdZeroNotEmitted(unittest.TestCase):
    def _batch_df(self, original_ids: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Response ID": [str(i) for i in range(1, len(original_ids) + 1)],
                "Response": [f"Response {oid}" for oid in original_ids],
                "Original Response ID": original_ids,
            }
        )

    def _parse(
        self,
        response_text: str,
        batch_df: pd.DataFrame,
        start_row: int,
        batch_size_number: int,
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            (
                _topic_path,
                _ref_path,
                _summary_path,
                _topic_df,
                reference_df,
                _summary_df,
                _details,
                is_error,
                _incomplete,
            ) = write_llm_output_and_logs(
                response_text=response_text,
                whole_conversation=[],
                all_metadata_content=[],
                batch_file_path_details="zero_id_batch",
                start_row=start_row,
                end_row=start_row + len(batch_df) - 1,
                model_choice_clean="test-model",
                log_files_output_paths=[],
                existing_reference_df=pd.DataFrame(),
                existing_topics_df=pd.DataFrame(),
                batch_size_number=batch_size_number,
                batch_basic_response_df=batch_df,
                group_name="All",
                produce_structured_summary_radio="No",
                output_folder=tmp + os.sep,
            )
        self.assertFalse(is_error)
        return reference_df

    def test_out_of_range_response_id_is_not_written_as_zero(self):
        reference_df = self._parse(
            """| General topic | Subtopic | Sentiment | Response ID | Summary |
|---|---|---|---|---|
| Transport | Parking charges | Negative | 99 | Fees are too high. |
""",
            self._batch_df([11, 12, 13]),
            start_row=10,
            batch_size_number=10,
        )
        if not reference_df.empty:
            ids = pd.to_numeric(reference_df["Response ID"], errors="coerce")
            self.assertFalse((ids.fillna(0) == 0).any())
            self.assertFalse((ids == 99).any())

    def test_global_original_id_in_later_batch_is_mapped(self):
        reference_df = self._parse(
            """| General topic | Subtopic | Sentiment | Response ID | Summary |
|---|---|---|---|---|
| Transport | Parking charges | Negative | 12 | Fees are too high. |
""",
            self._batch_df([11, 12, 13]),
            start_row=10,
            batch_size_number=10,
        )
        self.assertFalse(reference_df.empty)
        self.assertEqual(int(reference_df.iloc[0]["Response ID"]), 12)

    def test_blank_response_id_in_multi_row_batch_is_not_zero(self):
        reference_df = self._parse(
            """| General topic | Subtopic | Sentiment | Response ID | Summary |
|---|---|---|---|---|
| Transport | Parking charges | Negative |  | Fees are too high. |
""",
            self._batch_df([11, 12, 13]),
            start_row=10,
            batch_size_number=10,
        )
        if not reference_df.empty:
            ids = pd.to_numeric(reference_df["Response ID"], errors="coerce")
            self.assertFalse((ids.fillna(0) == 0).any())


if __name__ == "__main__":
    unittest.main()
