"""Tests for LLM workbook cover metadata used in eval summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_METADATA_PATH = REPO_ROOT / "evals" / "eval_metadata.py"


def _load_eval_metadata():
    spec = importlib.util.spec_from_file_location(
        "eval_metadata_test", EVAL_METADATA_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_metadata_test"] = module
    spec.loader.exec_module(module)
    return module


eval_metadata = _load_eval_metadata()


class TestEvalMetadata(unittest.TestCase):
    def test_parse_cover_key_values(self):
        cover = pd.DataFrame(
            [
                ["Large Language Model thematic analysis", None],
                ["File name", "Consultation export.xlsx"],
                ["Column name", "Please tell us more"],
                ["Model name", "gemma-4-31b"],
            ]
        )
        pairs = eval_metadata.parse_cover_key_values(cover)
        self.assertEqual(pairs["File name"], "Consultation export.xlsx")
        self.assertEqual(pairs["Column name"], "Please tell us more")
        self.assertEqual(pairs["Model name"], "gemma-4-31b")

    def test_read_llm_workbook_metadata_extracts_column_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_theme_analysis.xlsx"
            cover = pd.DataFrame(
                [
                    ["File name", "data.xlsx"],
                    ["Column name", "NATURE OF REPRESENTATION"],
                    ["Model name", "test-model"],
                ]
            )
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                cover.to_excel(
                    writer, sheet_name="Cover sheet", index=False, header=False
                )
            meta = eval_metadata.read_llm_workbook_metadata(path)
            self.assertEqual(meta["column_analysed"], "NATURE OF REPRESENTATION")

    def test_llm_excel_stem_matches_eval_prep_truncation(self):
        name = "Changes_to_emissions_col_Please_tell_us_more__Gemma_4_31B_theme_analysis.xlsx"
        self.assertEqual(
            eval_metadata.llm_excel_stem(name),
            "Changes_to_emissions_col_Please_tell_us_more__Gemma_4_3",
        )

    def test_metadata_for_stem_reads_json_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            intermediate = Path(tmp) / "intermediate"
            intermediate.mkdir()
            stem = "demo_run"
            record = {
                "stem": stem,
                "source_data_file": "from-json.xlsx",
                "model_name": "claude-test",
                "llm_theme_analysis_excel": "demo.xlsx",
            }
            path = intermediate / f"{stem}{eval_metadata.RUN_METADATA_SUFFIX}"
            path.write_text(json.dumps(record), encoding="utf-8")
            meta = eval_metadata.metadata_for_stem(
                stem, intermediate_dir=intermediate, inputs_dir=Path(tmp)
            )
            self.assertEqual(meta["source_data_file"], "from-json.xlsx")
            self.assertEqual(meta["model_name"], "claude-test")


if __name__ == "__main__":
    unittest.main()
