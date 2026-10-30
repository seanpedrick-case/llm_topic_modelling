"""Tests for eval prep ID join validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PREP_PATH = REPO_ROOT / "evals" / "1. eval_prep.py"


def _load_eval_prep():
    spec = importlib.util.spec_from_file_location("eval_prep_test", EVAL_PREP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_prep_test"] = module
    spec.loader.exec_module(module)
    return module


eval_prep = _load_eval_prep()


class TestEvalPrepIdJoin(unittest.TestCase):
    def test_raises_when_no_response_ids_join(self):
        with self.assertRaises(ValueError) as raised:
            eval_prep._raise_response_id_join_failure(
                matched_rows=0,
                total_rows=100,
                original_id_col="Respondent ID",
                data_id_col="Respondent ID",
                original_ids={"118249103552", "118249101344"},
                llm_lookup_ids={"118247000000", "118248000000"},
                llm_excel_name="llm.xlsx",
                original_excel_name="original.xlsx",
                original_sheet="Sheet1",
                llm_lookup_rows=9,
                join_original_by="id",
            )
        message = str(raised.exception)
        self.assertIn("Eval prep failed", message)
        self.assertIn("Overlapping ID(s): 0", message)
        self.assertIn("118249103552", message)
        self.assertIn("118247000000", message)
        self.assertIn("row_order", message)

    def test_unique_id_values_normalizes_float_suffix(self):
        series = pd.Series([118249103552.0, "118249103552.0", pd.NA])
        values = eval_prep._unique_id_values(series)
        self.assertEqual(values, {"118249103552"})

    def test_detect_pivot_only_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "themefinder_pivot.xlsx"
            pivot = pd.DataFrame(
                {
                    "Original Response ID": [1, 2],
                    "Response": ["a", "b"],
                    "Topic A": [1, 0],
                }
            )
            with pd.ExcelWriter(path) as writer:
                pivot.to_excel(
                    writer, sheet_name="Topic response pivot table", index=False
                )
            self.assertEqual(
                eval_prep._detect_llm_workbook_format(path),
                "pivot_only",
            )

    def test_map_pivot_via_themefinder_response_id(self):
        pivot = pd.DataFrame(
            {
                "Response ID": ["10", "20"],
                "Response": ["a", "b"],
                "Topic": [1, 0],
            }
        )
        original_data = pd.DataFrame(
            {
                "Response ID": [100, 200],
                "ThemeFinder Response ID": [10, 20],
                "ID": ["uuid-a", "uuid-b"],
            }
        )
        mapped = eval_prep._map_pivot_to_canonical_response_ids(pivot, original_data)
        self.assertEqual(mapped["Response ID"].tolist(), ["100", "200"])

    def test_row_order_join_for_pivot_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            intermediate = root / "intermediate"
            inputs.mkdir()
            intermediate.mkdir()

            llm_path = inputs / "external_pivot.xlsx"
            pivot = pd.DataFrame(
                {
                    "Original Response ID": [1, 2],
                    "Response": ["alpha", "beta"],
                    "LLM topic": [1, 0],
                }
            )
            with pd.ExcelWriter(llm_path) as writer:
                pivot.to_excel(
                    writer, sheet_name="Topic response pivot table", index=False
                )

            original_path = inputs / "human.xlsx"
            original = pd.DataFrame(
                {
                    "ID": ["uuid-a", "uuid-b"],
                    "text": ["x", "y"],
                    "Human topic": [1, 0],
                }
            )
            with pd.ExcelWriter(original_path) as writer:
                original.to_excel(writer, sheet_name="Sheet1", index=False)

            stem = eval_prep.run_eval_prep(
                llm_path.name,
                original_path.name,
                "Sheet1",
                pivot_col_start=2,
                join_original_by="row_order",
                inputs_dir=inputs,
                intermediate_dir=intermediate,
            )
            out = pd.read_csv(
                intermediate / f"{stem}_original_analysis_data_with_id_pivot.csv"
            )
            self.assertEqual(
                out["Response ID"].astype("string").tolist(),
                ["1", "2"],
            )


if __name__ == "__main__":
    unittest.main()
