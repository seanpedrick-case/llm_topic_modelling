import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tools.helper_functions import (
    load_in_data_file,
    normalise_excel_sheet_arg,
    read_file,
)


class TestNormaliseExcelSheetArg(unittest.TestCase):
    def test_list_uses_first_sheet(self):
        self.assertEqual(normalise_excel_sheet_arg(["Data", "Other"]), "Data")

    def test_empty_list_and_none(self):
        self.assertEqual(normalise_excel_sheet_arg([]), "")
        self.assertEqual(normalise_excel_sheet_arg(None), "")

    def test_gradio_update_dict(self):
        self.assertEqual(
            normalise_excel_sheet_arg({"value": "Responses", "choices": ["Responses"]}),
            "Responses",
        )

    def test_int_sheet_index_preserved(self):
        self.assertEqual(normalise_excel_sheet_arg(0), 0)


class TestReadExcelWithSheetList(unittest.TestCase):
    def test_read_file_accepts_sheet_name_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample.xlsx"
            pd.DataFrame({"Response": ["hello"]}).to_excel(
                path, sheet_name="Data", index=False
            )
            df = read_file(path, ["Data"])
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(list(df.columns), ["Response"])
            self.assertEqual(df["Response"].iloc[0], "hello")

    def test_load_in_data_file_accepts_sheet_name_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/sample.xlsx"
            pd.DataFrame({"Response": ["hello"], "Id": [1]}).to_excel(
                path, sheet_name="Data", index=False
            )
            with patch("tools.helper_functions.ENABLE_INPUT_REDACTION", False):
                file_data, file_name, num_batches = load_in_data_file(
                    [path], ["Response"], batch_size=5, in_excel_sheets=["Data"]
                )
            self.assertFalse(file_data.empty)
            self.assertIn("Response", file_data.columns)
            self.assertEqual(file_data["Response"].iloc[0], "hello")
            self.assertEqual(file_name, "sample")
            self.assertEqual(num_batches, 1)


if __name__ == "__main__":
    unittest.main()
