"""Headless batch app_defaults.env rendering."""

import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CDK_DIR))

from cdk_functions import build_headless_app_defaults_env_content


def test_build_headless_app_defaults_env_sets_outputs_bucket():
    seed_dir = str(CDK_DIR / "config" / "headless_s3_seed")
    content = build_headless_app_defaults_env_content(
        seed_dir,
        s3_outputs_bucket_name="lambeth-prod-summarisation-s3-output",
    )

    assert "S3_OUTPUTS_BUCKET=lambeth-prod-summarisation-s3-output" in content
    assert "RUN_DIRECT_MODE=1" in content
    assert "SAVE_OUTPUTS_TO_S3=True" in content
    assert content.count("S3_OUTPUTS_BUCKET=") == 1
