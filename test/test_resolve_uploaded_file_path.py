import os
import unittest

from gradio.utils import NamedString

from tools.helper_functions import resolve_uploaded_file_path

BUNDLED_NAME = "improved_dummy_consultation_responses.csv"
BUNDLED_PATH = os.path.join("example_data", BUNDLED_NAME)


@unittest.skipUnless(
    os.path.isfile(BUNDLED_PATH),
    "example_data files not present",
)
class ResolveUploadedFilePathTests(unittest.TestCase):
    def test_resolves_bundled_example_when_gradio_cache_missing(self):
        missing_cache = os.path.join(
            "tmp",
            "gradio_tmp",
            "deadbeef",
            BUNDLED_NAME,
        )
        resolved = resolve_uploaded_file_path(missing_cache)
        self.assertTrue(os.path.isfile(resolved))
        self.assertEqual(os.path.basename(resolved), BUNDLED_NAME)

    def test_resolves_gradio_six_filedata_dict(self):
        bundled = os.path.abspath(BUNDLED_PATH)
        resolved = resolve_uploaded_file_path(
            {"path": bundled, "orig_name": BUNDLED_NAME}
        )
        self.assertTrue(os.path.isfile(resolved))
        self.assertEqual(os.path.basename(resolved), BUNDLED_NAME)

    def test_resolves_named_string_filepath(self):
        bundled = os.path.abspath(BUNDLED_PATH)
        resolved = resolve_uploaded_file_path(NamedString(bundled))
        self.assertTrue(os.path.isfile(resolved))
        self.assertEqual(os.path.basename(resolved), BUNDLED_NAME)


if __name__ == "__main__":
    unittest.main()
