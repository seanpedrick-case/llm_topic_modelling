"""Tests for eval prep ID join validation."""

from __future__ import annotations

import importlib.util
import sys
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
            )
        message = str(raised.exception)
        self.assertIn("Eval prep failed", message)
        self.assertIn("Overlapping ID(s): 0", message)
        self.assertIn("118249103552", message)
        self.assertIn("118247000000", message)
        self.assertIn("--data_id_col", message)

    def test_unique_id_values_normalizes_float_suffix(self):
        series = pd.Series([118249103552.0, "118249103552.0", pd.NA])
        values = eval_prep._unique_id_values(series)
        self.assertEqual(values, {"118249103552"})


if __name__ == "__main__":
    unittest.main()
