"""Tests for eval topic matching and compare-step column handling."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"


def _load_eval_module(filename: str, module_name: str) -> ModuleType:
    path = EVALS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


group_topics = _load_eval_module("2. group_topics.py", "eval_group_topics_test")
topic_compare = _load_eval_module("3. topic_compare.py", "eval_topic_compare_test")


class TestExactAndFuzzyMatching(unittest.TestCase):
    def test_exact_prepass_prefers_atomic_label_over_blob(self):
        original = ["Shorter Hours ", "Speeding"]
        llm = [
            "Negative Impact On Shops, Generating Revenue, Shorter Hours",
            "Shorter Hours",
            "Bus Lanes, Speeding",
            "Speeding",
        ]
        pairs, rem_original, rem_llm = group_topics.exact_prepass(original, llm)
        matched = {orig: llm_label for orig, llm_label, _ in pairs}
        self.assertEqual(matched["Shorter Hours "], "Shorter Hours")
        self.assertEqual(matched["Speeding"], "Speeding")
        self.assertEqual(rem_original, [])
        self.assertIn(
            "Negative Impact On Shops, Generating Revenue, Shorter Hours", rem_llm
        )
        self.assertIn("Bus Lanes, Speeding", rem_llm)

    def test_slash_and_ampersand_fold_into_exact_matches(self):
        original = [
            "Shorter -Other 1/2hr variants, school pickup/drop off or non specified ",
            "Additional free bays for High Street & Rosendale Road businesses",
        ]
        llm = [
            "Shorter -Other 1 Or 2Hr Variants, School Pickup Or Drop Off Or Non Specified",
            "Additional Free Bays For High Street And Rosendale Road Businesses",
        ]
        pairs, rem_original, rem_llm = group_topics.exact_prepass(original, llm)
        self.assertEqual(rem_original, [])
        self.assertEqual(rem_llm, [])
        matched = {orig: llm_label for orig, llm_label, _ in pairs}
        self.assertEqual(matched[original[0]], llm[0])
        self.assertEqual(matched[original[1]], llm[1])

    def test_compound_original_still_matches_compound_llm_label(self):
        original = ["Cost of Living, Permit Prices, Generating Revenue"]
        llm = [
            "Generating Revenue",
            "Permit Prices",
            "Cost Of Living, Permit Prices, Generating Revenue",
            "Cost Of Living, Permit Prices, Generating Revenue, Operational Hours",
        ]
        pairs, rem_original, _rem_llm = group_topics.exact_prepass(original, llm)
        self.assertEqual(rem_original, [])
        self.assertEqual(
            pairs[0][1], "Cost Of Living, Permit Prices, Generating Revenue"
        )

    def test_fuzzy_does_not_pair_simple_original_to_joined_blob(self):
        original = ["Shorter Hours"]
        llm = [
            "Negative Impact On Shops And Businesses, Shorter Hours",
            "Shorter Hours Extra Words Nearby",
        ]
        catalog = llm + ["Shorter Hours"]
        pairs, rem_original, rem_llm = group_topics.fuzzy_prepass(
            original, llm, threshold=90, catalog=catalog
        )
        self.assertEqual(pairs, [])
        self.assertEqual(rem_original, original)
        self.assertEqual(rem_llm, llm)

    def test_fuzzy_matches_punctuation_variants_not_blobs(self):
        original = ["Remove / relocate cycle hangar"]
        llm = [
            "Criticism Of Consultation, Remove Or Relocate Cycle Hangar",
            "Remove Or Relocate Cycle Hangar",
        ]
        pairs, rem_original, rem_llm = group_topics.fuzzy_prepass(
            original, llm, threshold=90, catalog=llm
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], "Remove Or Relocate Cycle Hangar")
        self.assertEqual(rem_original, [])
        self.assertEqual(
            rem_llm, ["Criticism Of Consultation, Remove Or Relocate Cycle Hangar"]
        )

    def test_build_match_table_locks_atomic_columns_without_embeddings(self):
        original = [
            "Shorter Hours",
            "Speeding",
            "Cost of Living, Permit Prices, Generating Revenue",
        ]
        llm = [
            "Shorter Hours",
            "Negative Impact, Shorter Hours",
            "Speeding",
            "Bus Lanes, Speeding",
            "Cost Of Living, Permit Prices, Generating Revenue",
            "Generating Revenue",
        ]
        matched = group_topics.build_match_table(
            original, llm, fuzzy_threshold=90, embed_threshold=0.99
        )
        paired = matched[
            matched["Original topics"].notna() & matched["LLM topics"].notna()
        ]
        by_original = dict(zip(paired["Original topics"], paired["LLM topics"]))
        self.assertEqual(by_original["Shorter Hours"], "Shorter Hours")
        self.assertEqual(by_original["Speeding"], "Speeding")
        self.assertEqual(
            by_original["Cost of Living, Permit Prices, Generating Revenue"],
            "Cost Of Living, Permit Prices, Generating Revenue",
        )
        unmatched_llm = set(
            matched.loc[matched["Original topics"].isna(), "LLM topics"]
        )
        self.assertIn("Negative Impact, Shorter Hours", unmatched_llm)
        self.assertIn("Bus Lanes, Speeding", unmatched_llm)


class TestBlobFolding(unittest.TestCase):
    def test_map_unmatched_blob_splits_to_matched_originals(self):
        unmatched = [
            "Bus Lanes, Speeding",
            "Negative Impact, Shorter Hours",
            "No Relevant Topic",
        ]
        originals = ["Bus Lanes", "Speeding", "Shorter Hours"]
        catalog = unmatched + ["Bus Lanes", "Speeding", "Shorter Hours"]
        mapped = group_topics.map_unmatched_blob_columns(
            unmatched, originals, llm_catalog=catalog
        )
        self.assertIn("Bus Lanes, Speeding", mapped["Bus Lanes"])
        self.assertIn("Bus Lanes, Speeding", mapped["Speeding"])
        self.assertIn("Negative Impact, Shorter Hours", mapped["Shorter Hours"])
        self.assertNotIn(
            "No Relevant Topic", {c for cols in mapped.values() for c in cols}
        )

    def test_compare_folds_matched_blob_segments_into_llm_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            intermediate = Path(tmp) / "intermediate"
            outputs = Path(tmp) / "outputs"
            intermediate.mkdir()
            stem = "blob_fold_case"
            pd.DataFrame(
                {
                    "Original topics": ["Shorter Hours", pd.NA],
                    "LLM topics": ["Shorter Hours", "Trade, Shorter Hours"],
                    "similarity": [1.0, pd.NA],
                }
            ).to_csv(intermediate / f"{stem}_topic_match.csv", index=False)
            pd.DataFrame({"Response ID": [1, 2, 3], "Shorter Hours": [1, 0, 1]}).to_csv(
                intermediate / f"{stem}_original_analysis_data_with_id_pivot.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "Response ID": [1, 2, 3],
                    "Shorter Hours": [1, 0, 0],
                    "Trade, Shorter Hours": [0, 1, 1],
                }
            ).to_csv(
                intermediate / f"{stem}_llm_pivot_table_analysis_with_id.csv",
                index=False,
            )
            topic_compare.run_topic_compare(
                stem, intermediate_dir=intermediate, outputs_dir=outputs
            )
            combined = pd.read_csv(outputs / f"{stem}_combined_analysis_data.csv")
            llm_rows = combined[combined["analysis_type"] == "LLM"].sort_values(
                "Response ID"
            )
            self.assertEqual(llm_rows["Shorter Hours"].tolist(), [1.0, 1.0, 1.0])


class TestTopicCompareColumns(unittest.TestCase):
    def test_to_binary_flags_explains_duplicate_columns(self):
        frame = pd.DataFrame([[1, 0], [0, 1]])
        frame.columns = ["Speeding", "Speeding"]
        duplicate = frame.loc[:, frame.columns == "Speeding"]
        with self.assertRaises(TypeError) as raised:
            topic_compare._to_binary_flags(duplicate)
        self.assertIn("duplicated after rename", str(raised.exception))

    def test_collapse_duplicate_columns_ors_flags(self):
        frame = pd.DataFrame(
            [["r1", 1, 0], ["r2", 0, 1], ["r3", 0, 0]],
        )
        frame.columns = ["Response ID", "Speeding", "Speeding"]
        collapsed = topic_compare._collapse_duplicate_columns(frame)
        self.assertEqual(list(collapsed.columns), ["Response ID", "Speeding"])
        self.assertEqual(collapsed["Speeding"].tolist(), [1.0, 1.0, 0.0])

    def test_compare_keeps_matched_source_column_not_name_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            intermediate = Path(tmp) / "intermediate"
            outputs = Path(tmp) / "outputs"
            intermediate.mkdir()
            stem = "collision_case"
            pd.DataFrame(
                {
                    "Original topics": ["Speeding", pd.NA],
                    "LLM topics": ["Speeding", "Bus Lanes, Speeding"],
                    "similarity": [1.0, pd.NA],
                }
            ).to_csv(intermediate / f"{stem}_topic_match.csv", index=False)
            pd.DataFrame(
                {
                    "Response ID": [1, 2],
                    "Speeding": [1, 0],
                }
            ).to_csv(
                intermediate / f"{stem}_original_analysis_data_with_id_pivot.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "Response ID": [1, 2],
                    "Speeding": [0, 1],
                    "Bus Lanes, Speeding": [1, 0],
                }
            ).to_csv(
                intermediate / f"{stem}_llm_pivot_table_analysis_with_id.csv",
                index=False,
            )
            pd.DataFrame({"Response ID": [1, 2], "Response": ["a", "b"]}).to_csv(
                intermediate / f"{stem}_response_text.csv", index=False
            )

            topic_compare.run_topic_compare(
                stem, intermediate_dir=intermediate, outputs_dir=outputs
            )
            combined = pd.read_csv(outputs / f"{stem}_combined_analysis_data.csv")
            llm_rows = combined[combined["analysis_type"] == "LLM"].sort_values(
                "Response ID"
            )
            # Atomic column plus unmatched blob segment matched to Speeding.
            self.assertEqual(llm_rows["Speeding"].tolist(), [1.0, 1.0])
            grouped = pd.read_csv(
                outputs / f"{stem}_combined_analysis_data_grouped.csv"
            )
            self.assertIn("Speeding", grouped.columns)


if __name__ == "__main__":
    unittest.main()
