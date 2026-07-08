"""S3 batch Lambda handler behaviour (assignPublicIp, env merge)."""

import importlib.util
import sys
from pathlib import Path

CDK_DIR = Path(__file__).resolve().parents[1]
LAMBDA_PATH = CDK_DIR / "config" / "lambda" / "lambda_function.py"


def _load_lambda_module(module_name: str = "batch_lambda"):
    spec = importlib.util.spec_from_file_location(module_name, LAMBDA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_assign_public_ip_reads_env(monkeypatch):
    monkeypatch.setenv("ECS_ASSIGN_PUBLIC_IP", "ENABLED")
    module = _load_lambda_module("batch_lambda_ip")
    assert module.ECS_ASSIGN_PUBLIC_IP == "ENABLED"


def test_build_environment_merges_job_and_default_env(monkeypatch):
    get_object_buckets = []

    mod = _load_lambda_module("batch_lambda_merge")

    def fake_get_object(*, Bucket, Key):
        get_object_buckets.append((Bucket, Key))
        body = (
            b"RUN_DIRECT_MODE=1\n" b"DIRECT_MODE_INPUT_FILE=job.xlsx\n"
            if Key.endswith("job.env")
            else b"RUN_AWS_FUNCTIONS=1\n"
        )
        return {
            "Body": type(
                "Body",
                (),
                {"read": staticmethod(lambda: body)},
            )()
        }

    monkeypatch.setattr(
        mod,
        "s3",
        type("S3", (), {"get_object": staticmethod(fake_get_object)})(),
    )
    monkeypatch.setattr(
        mod,
        "ecs",
        type(
            "ECS",
            (),
            {"run_task": staticmethod(lambda **kwargs: {"tasks": [], "failures": []})},
        )(),
    )

    mod.BUCKET = "output-bucket"
    mod.LOG_BUCKET = "log-config-bucket"
    mod.INPUT_PREFIX = "input/"
    mod.ENV_PREFIX = "input/config/"
    mod.ENV_SUFFIX = ".env"
    mod.DEFAULT_PARAMS_KEY = "general-config/app_defaults.env"
    mod.CLUSTER = "cluster"
    mod.TASK_DEF = "arn:aws:ecs:eu-west-2:123:task-definition/app:1"
    mod.SUBNETS = ["subnet-1"]
    mod.SECURITY_GROUPS = ["sg-1"]
    mod.CONTAINER_NAME = "app"

    event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "output-bucket"},
                    "object": {"key": "input/config/job.env"},
                }
            }
        ]
    }
    result = mod.lambda_handler(event, None)
    assert result["runs"]
    assert result["runs"][0]["envCount"] >= 2
    assert (
        "log-config-bucket",
        "general-config/app_defaults.env",
    ) in get_object_buckets
    assert ("output-bucket", "input/config/job.env") in get_object_buckets


def test_log_bucket_falls_back_to_bucket_when_unset(monkeypatch):
    monkeypatch.delenv("LOG_BUCKET", raising=False)
    monkeypatch.setenv("BUCKET", "only-output-bucket")
    mod = _load_lambda_module("batch_lambda_log_fallback")
    assert mod.LOG_BUCKET == "only-output-bucket"
