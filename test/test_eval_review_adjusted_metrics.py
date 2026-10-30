"""Tests for review-adjusted metrics discovery and run-summary join."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "evals" / "5. review_adjusted_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "review_adjusted_metrics_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["review_adjusted_metrics_test"] = module
    spec.loader.exec_module(module)
    return module


review_adjusted = _load_module()


class TestDiscoverAndProvenance(unittest.TestCase):
    def test_discover_annotated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "a_topic_disagreements_annotated.csv").write_text("x", encoding="utf-8")
            (out / "b_topic_disagreements_annotated.csv").write_text("x", encoding="utf-8")
            (out / "not_a_match.csv").write_text("x", encoding="utf-8")

            found = review_adjusted.discover_annotated_files(out)
            self.assertEqual(
                [p.name for p in found],
                [
                    "a_topic_disagreements_annotated.csv",
                    "b_topic_disagreements_annotated.csv",
                ],
            )

    def test_attach_run_summary_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame(
                {
                    "stem": ["run_a", "run_a"],
                    "scenario": ["observed_corrected_gold", "extrapolated_corrected_gold"],
                    "adjusted_f1": [0.9, 0.95],
                }
            ).to_csv(out / "partial_summary.csv", index=False)
            pd.DataFrame(
                {
                    "stem": ["run_a"],
                    "source_data_file": ["data.xlsx"],
                    "column_analysed": ["Q1"],
                    "model_name": ["test-model"],
                }
            ).to_csv(out / review_adjusted.RUN_SUMMARY_FILENAME, index=False)

            summary = pd.read_csv(out / "partial_summary.csv")
            enriched = review_adjusted.attach_run_summary_provenance(summary, out)
            self.assertEqual(enriched.loc[0, "model_name"], "test-model")
            self.assertEqual(enriched.loc[0, "source_data_file"], "data.xlsx")
            self.assertEqual(enriched.loc[1, "column_analysed"], "Q1")


if __name__ == "__main__":
    unittest.main()
