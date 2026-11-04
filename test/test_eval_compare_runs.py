"""Tests for paired eval run comparison (McNemar + bootstrap)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "evals" / "compare_runs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_runs_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_runs_test"] = module
    spec.loader.exec_module(module)
    return module


compare_runs = _load_module()


class TestCompareRuns(unittest.TestCase):
    def test_bonferroni_multiplies_by_family_size(self):
        adjusted = compare_runs.bonferroni_adjust(
            {"a": 0.04, "b": 0.01, "c": None},
        )
        self.assertEqual(adjusted["a"], 0.08)
        self.assertEqual(adjusted["b"], 0.02)
        self.assertIsNone(adjusted["c"])

    def test_holm_less_conservative_than_bonferroni(self):
        raw = {"a": 0.04, "b": 0.02, "c": 0.01}
        holm = compare_runs.holm_adjust(raw)
        bonf = compare_runs.bonferroni_adjust(raw)
        self.assertLessEqual(holm["a"], bonf["a"])
        self.assertEqual(holm["c"], 0.03)

    def test_row_jaccard_and_f1(self):
        gold = np.array([1, 1, 0])
        pred = np.array([1, 0, 0])
        self.assertAlmostEqual(compare_runs.row_jaccard_similarity(gold, pred), 0.5)
        self.assertAlmostEqual(compare_runs.row_f1_score(gold, pred), 2 / 3)

    def test_paired_response_bootstrap_detects_difference(self):
        scores = {
            "jaccard_delta": np.array([0.2, 0.3, 0.25]),
            "f1_delta": np.array([0.1, 0.1, 0.1]),
        }
        out = compare_runs.paired_response_bootstrap(scores, n_bootstrap=400, seed=0)
        self.assertLess(out["jaccard"]["p_two_sided"], 0.05)

    def test_jaccard_distance_identical_rows(self):
        X = np.array([[1, 1, 0], [1, 1, 0]], dtype=float)
        D = compare_runs.jaccard_distance_matrix(X)
        self.assertAlmostEqual(D[0, 1], 0.0)

    def test_paired_permanova_runs(self):
        pred_a = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
        pred_b = np.array([[0, 0], [1, 0], [0, 0]], dtype=bool)
        out = compare_runs.paired_permanova_jaccard(
            pred_a, pred_b, permutations=200, seed=0
        )
        self.assertEqual(out["permanova_n_responses"], 3)
        self.assertIsNotNone(out["permanova_p_value"])

    def test_mcnemar_exact_no_discordant(self):
        stats = compare_runs.mcnemar_exact(0, 0)
        self.assertEqual(stats["mcnemar_p_value"], 1.0)

    def test_mcnemar_exact_balanced_discordant(self):
        stats = compare_runs.mcnemar_exact(5, 5)
        self.assertGreater(stats["mcnemar_p_value"], 0.9)

    def test_micro_metrics_matches_topic_compare_style(self):
        gold = [1, 1, 0, 0]
        pred = [1, 0, 0, 1]
        m = compare_runs.micro_metrics(gold, pred)
        self.assertEqual(m.tp, 1)
        self.assertEqual(m.original_pos, 2)
        self.assertEqual(m.llm_pos, 2)
        self.assertEqual(m.precision, 0.5)
        self.assertEqual(m.recall, 0.5)

    def test_paired_units_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write_combined(stem: str, llm_flip_row0_topic0: bool):
                rows = []
                for analysis_type, t1, t2 in (
                    ("Original", 1, 0),
                    ("LLM", 1 if not llm_flip_row0_topic0 else 0, 1),
                ):
                    rows.append(
                        {
                            "Response ID": 1,
                            "ID": "uuid-1",
                            "analysis_type": analysis_type,
                            "T1": t1,
                            "T2": t2,
                        }
                    )
                for analysis_type, t1, t2 in (
                    ("Original", 0, 1),
                    ("LLM", 0, 1),
                ):
                    rows.append(
                        {
                            "Response ID": 2,
                            "ID": "uuid-2",
                            "analysis_type": analysis_type,
                            "T1": t1,
                            "T2": t2,
                        }
                    )
                pd.DataFrame(rows).to_csv(
                    out / f"{stem}_combined_analysis_data.csv", index=False
                )

            write_combined("run_a", llm_flip_row0_topic0=False)
            write_combined("run_b", llm_flip_row0_topic0=True)

            merged, meta = compare_runs.paired_units_for_stems(
                "run_a", "run_b", outputs_dir=out
            )
            self.assertEqual(meta["n_units_paired"], 4)
            self.assertEqual(len(merged), 4)

            result = compare_runs.compare_runs(
                "run_a",
                "run_b",
                outputs_dir=out,
                n_bootstrap=500,
                seed=1,
            )
            self.assertEqual(result["mcnemar_n_discordant"], 1)
            self.assertIn("jaccard", result["response_bootstrap"])
            self.assertIn("f1", result["bootstrap_unit_level"])
            self.assertEqual(result["multiple_testing"], "holm")


if __name__ == "__main__":
    unittest.main()
