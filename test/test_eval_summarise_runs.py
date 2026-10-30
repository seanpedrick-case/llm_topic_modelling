"""Tests for eval run summary topic catalog stats."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARISE_PATH = REPO_ROOT / "evals" / "4. summarise_runs.py"


def _load_summarise_runs():
    spec = importlib.util.spec_from_file_location("summarise_runs_test", SUMMARISE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarise_runs_test"] = module
    spec.loader.exec_module(module)
    return module


summarise_runs = _load_summarise_runs()


class TestTopicCatalogStats(unittest.TestCase):
    def test_counts_original_and_matched_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            intermediate = Path(tmp)
            stem = "demo_run"
            pd.DataFrame(
                {
                    "Original topics": ["Topic A", "Topic B", "Topic C", pd.NA],
                    "LLM topics": ["LLM A", "LLM B", "LLM C", "LLM D"],
                }
            ).to_csv(intermediate / f"{stem}_topic_match_initial.csv", index=False)
            pd.DataFrame(
                {
                    "Original topics": ["Topic A", "Topic B", pd.NA, pd.NA],
                    "LLM topics": ["LLM A", "LLM B", pd.NA, "LLM D"],
                    "similarity": [1.0, 0.95, pd.NA, pd.NA],
                }
            ).to_csv(intermediate / f"{stem}_topic_match.csv", index=False)

            stats = summarise_runs._topic_catalog_stats(stem, intermediate)
            self.assertEqual(stats["original_topics"], 3)
            self.assertEqual(stats["matched_topics"], 2)
            self.assertEqual(stats["pct_matched_topics"], 66.7)


if __name__ == "__main__":
    unittest.main()
