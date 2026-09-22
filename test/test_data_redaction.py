"""Optional Presidio/spaCy redaction of input text and Original data export."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tools.config import REDACTION_ENTITIES, _parse_entity_list
from tools.helper_functions import load_in_data_file


def _spacy_model_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


requires_spacy = unittest.skipUnless(
    _spacy_model_available(), "en_core_web_sm is not installed"
)


def _fake_redact(df, columns=None, **kwargs):
    out = df.copy()
    cols = list(out.columns) if columns is None else list(columns)
    replacements = {
        "jane@example.com": "<EMAIL_ADDRESS>",
        "07700 900123": "<PHONE_NUMBER>",
        "4111111111111111": "<CREDIT_CARD>",
    }
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col].astype(str)
        for original, replacement in replacements.items():
            series = series.str.replace(original, replacement, regex=False)
        out[col] = series
    return out


class TestRedactionConfig(unittest.TestCase):
    def test_default_entities_include_email_phone_and_number_types(self):
        self.assertIn("EMAIL_ADDRESS", REDACTION_ENTITIES)
        self.assertIn("PHONE_NUMBER", REDACTION_ENTITIES)
        self.assertIn("CREDIT_CARD", REDACTION_ENTITIES)
        self.assertNotIn("PERSON", REDACTION_ENTITIES)
        self.assertNotIn("CARDINAL", REDACTION_ENTITIES)

    def test_parse_entity_list_accepts_comma_separated_values(self):
        self.assertEqual(
            _parse_entity_list("EMAIL_ADDRESS, PHONE_NUMBER"),
            ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        )
        self.assertEqual(
            _parse_entity_list("['EMAIL_ADDRESS', 'PHONE_NUMBER']"),
            ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        )


class TestInputRedactionFlag(unittest.TestCase):
    def _write_csv(self, directory: str) -> str:
        path = os.path.join(directory, "comments.csv")
        pd.DataFrame(
            {
                "Response": ["Contact jane@example.com — I waited 3 weeks."],
                "Notes": ["Also email jane@example.com"],
            }
        ).to_csv(path, index=False)
        return path

    def test_flags_off_leaves_text_unchanged_and_does_not_load_spacy(self):
        sys.modules.pop("tools.data_redaction", None)
        sys.modules.pop("tools.load_spacy_model_custom_recognisers", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp)
            with patch("tools.helper_functions.ENABLE_INPUT_REDACTION", False):
                file_data, _, _ = load_in_data_file([path], ["Response"], batch_size=5)

        self.assertIn("jane@example.com", file_data["Response"].iloc[0])
        self.assertIn("3 weeks", file_data["Response"].iloc[0])
        self.assertNotIn("tools.data_redaction", sys.modules)
        self.assertNotIn("tools.load_spacy_model_custom_recognisers", sys.modules)

    def test_flags_on_redacts_only_chosen_text_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp)
            with (
                patch("tools.helper_functions.ENABLE_INPUT_REDACTION", True),
                patch(
                    "tools.data_redaction.redact_dataframe", side_effect=_fake_redact
                ),
            ):
                file_data, _, _ = load_in_data_file([path], ["Response"], batch_size=5)

        self.assertIn("<EMAIL_ADDRESS>", file_data["Response"].iloc[0])
        self.assertNotIn("jane@example.com", file_data["Response"].iloc[0])
        self.assertIn("3 weeks", file_data["Response"].iloc[0])
        self.assertIn("jane@example.com", file_data["Notes"].iloc[0])


class TestOriginalDataRedaction(unittest.TestCase):
    def test_unredacted_csv_reuses_source_path(self):
        from tools.combine_sheets_into_xlsx import stage_original_data_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "original.csv")
            pd.DataFrame({"Response": ["hello"], "Email": ["a@b.com"]}).to_csv(
                path, index=False
            )
            csv_path, is_temp = stage_original_data_csv(path, tmp, redact=False)

        self.assertEqual(csv_path, path)
        self.assertFalse(is_temp)

    def test_redacts_all_columns_without_mutating_source_file(self):
        from tools.combine_sheets_into_xlsx import stage_original_data_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "original.csv")
            source = pd.DataFrame(
                {
                    "Response": ["I waited 3 weeks."],
                    "Contact": ["jane@example.com"],
                }
            )
            source.to_csv(path, index=False)

            with patch(
                "tools.data_redaction.redact_dataframe", side_effect=_fake_redact
            ):
                csv_path, is_temp = stage_original_data_csv(path, tmp, redact=True)

            self.assertTrue(is_temp)
            self.assertNotEqual(csv_path, path)
            staged = pd.read_csv(csv_path)
            self.assertEqual(staged["Contact"].iloc[0], "<EMAIL_ADDRESS>")
            self.assertIn("3 weeks", staged["Response"].iloc[0])

            reread = pd.read_csv(path)
            self.assertEqual(reread["Contact"].iloc[0], "jane@example.com")


@requires_spacy
class TestLivePresidioRedaction(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        from tools.data_redaction import reset_redaction_engines

        reset_redaction_engines()

    def test_email_phone_and_card_are_replaced_ordinary_numbers_kept(self):
        from tools.data_redaction import redact_dataframe, reset_redaction_engines

        reset_redaction_engines()
        df = pd.DataFrame(
            {
                "Response": [
                    "Email jane@example.com or call 07700 900123. "
                    "Card 4111111111111111. I waited 3 weeks."
                ]
            }
        )
        redacted = redact_dataframe(df, columns=["Response"])
        text = str(redacted["Response"].iloc[0])
        self.assertNotIn("jane@example.com", text.lower())
        self.assertNotIn("07700 900123", text)
        self.assertNotIn("4111111111111111", text)
        self.assertIn("3 weeks", text)
        self.assertTrue(
            "<EMAIL_ADDRESS>" in text or "EMAIL_ADDRESS" in text,
            msg=f"Expected email entity placeholder in: {text}",
        )


if __name__ == "__main__":
    unittest.main()
