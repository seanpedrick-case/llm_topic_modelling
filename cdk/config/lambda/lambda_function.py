import os
import urllib.parse

import boto3

ecs = boto3.client("ecs")
s3 = boto3.client("s3")

# Move static config to Lambda env vars for easier ops
BUCKET = os.environ.get("BUCKET", "lambeth-llm-topic-modelling")
INPUT_PREFIX = os.environ.get("INPUT_PREFIX", "input/")
ENV_PREFIX = os.environ.get("ENV_PREFIX", f"{INPUT_PREFIX}config/")
GENERAL_ENV_PREFIX = os.environ.get("GENERAL_ENV_PREFIX", "general-config/")
ENV_SUFFIX = os.environ.get("ENV_SUFFIX", ".env")
DEFAULT_PARAMS_KEY = os.environ.get(
    "DEFAULT_PARAMS_KEY", f"{GENERAL_ENV_PREFIX}app_defaults{ENV_SUFFIX}"
)
CLUSTER = os.environ.get("ECS_CLUSTER", "analytics_fargate_cluster")
TASK_DEF = os.environ.get("ECS_TASK_DEF", "llm_topic_modelling_fargate_def:5")
SUBNETS = os.environ.get(
    "SUBNETS", "subnet-083650a35f93a0ee5,subnet-0a41ba70470cabab8"
).split(",")
SECURITY_GROUPS = os.environ.get("SECURITY_GROUPS", "sg-051effb4c5e752df1").split(",")
ECS_ASSIGN_PUBLIC_IP = os.environ.get("ECS_ASSIGN_PUBLIC_IP", "DISABLED").upper()
DEFAULT_INPUT_S3_URI = os.environ.get(
    "DEFAULT_INPUT_S3_URI",
    f"s3://{BUCKET}/{INPUT_PREFIX}dummy_consultation_response.xlsx",
)
DEFAULT_TASK_TYPE = os.environ.get("DEFAULT_TASK_TYPE", "extract")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "llm_topic_modelling_container")


def _key_matches(key: str) -> bool:
    return key.startswith(ENV_PREFIX) and key.endswith(ENV_SUFFIX)


def _derive_runtime_params_from_key(key: str) -> dict:
    """Optional convention: decide task type from file name."""
    basename = key.split("/")[-1].lower()
    if "train" in basename:
        task_type = "train"
    elif "infer" in basename or "staging" in basename:
        task_type = "infer"
    else:
        task_type = DEFAULT_TASK_TYPE

    return {
        "DIRECT_MODE_TASK": task_type,
        "DIRECT_MODE_INPUT_FILE": DEFAULT_INPUT_S3_URI,
    }


def _parse_dotenv(dotenv_bytes: bytes, bucket: str, input_prefix: str) -> dict:
    """
    Parse a basic .env and prepend S3 paths to specific keys if they have values.
    """
    env = {}
    text = dotenv_bytes.decode("utf-8", errors="replace")

    # 1. Parse the file (Existing Logic)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value

    # 2. Apply S3 Prefix Logic (New Logic)
    # We define the base S3 path string using the provided arguments
    s3_base = f"s3://{bucket}/{input_prefix}"

    # List of keys that need modification
    keys_to_modify = ["DIRECT_MODE_INPUT_FILE", "DIRECT_MODE_CANDIDATE_TOPICS"]

    for key in keys_to_modify:
        # env.get(key) returns the value.
        # The 'if' check ensures the key exists AND is not an empty string.
        if env.get(key):
            env[key] = s3_base + env[key]

    return env


def _build_environment_array(*env_dicts):
    """Merge dictionaries left→right and produce ECS environment array format."""
    merged = {}
    for d in env_dicts:
        merged.update(d or {})
    # ECS override format: [{"name": "...","value":"..."}]
    return [{"name": k, "value": v} for k, v in merged.items()]


def lambda_handler(event, context):
    runs = []

    # Parse default env file with default config
    # Fetch the default .env file content from S3
    default_obj = s3.get_object(Bucket=BUCKET, Key=DEFAULT_PARAMS_KEY)
    default_dotenv_bytes = default_obj["Body"].read()

    # Parse .env → dict of env vars
    default_file_env = _parse_dotenv(default_dotenv_bytes, BUCKET, INPUT_PREFIX)

    for record in event.get("Records", []):

        # Parse env file from user
        s3rec = record.get("s3", {})
        bucket = s3rec.get("bucket", {}).get("name")
        raw_key = s3rec.get("object", {}).get("key")
        if not bucket or not raw_key:
            print(f"Object in bucket {bucket} and key {raw_key} not found, exiting.")
            continue

        key = urllib.parse.unquote_plus(raw_key)

        # combined_key = ENV_PREFIX + key
        combined_key = key

        print("combined_key:", combined_key)

        # Only process the intended prefix/suffix
        if not _key_matches(combined_key):
            print(f"Key does not match: {combined_key}, exiting")
            continue

        # Fetch the .env file content from S3
        obj = s3.get_object(Bucket=bucket, Key=combined_key)
        dotenv_bytes = obj["Body"].read()

        # Parse .env → dict of env vars
        file_env = _parse_dotenv(dotenv_bytes, BUCKET, INPUT_PREFIX)

        # Optionally derive run-time params from naming
        # run_params = _derive_runtime_params_from_key(key)

        # Merge: file env (dominant), overwrites default values.
        environment = _build_environment_array(file_env, default_file_env)

        print("Combined environment variables:", environment)

        # Execute ECS Fargate task with fully dynamic environment overrides
        response = ecs.run_task(
            cluster=CLUSTER,
            launchType="FARGATE",
            taskDefinition=TASK_DEF,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNETS,
                    "securityGroups": SECURITY_GROUPS,
                    "assignPublicIp": (
                        ECS_ASSIGN_PUBLIC_IP
                        if ECS_ASSIGN_PUBLIC_IP in ("ENABLED", "DISABLED")
                        else "DISABLED"
                    ),
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": CONTAINER_NAME,
                        "environment": environment,
                    }
                ]
            },
        )

        runs.append(
            {
                "bucket": bucket,
                "key": key,
                "taskArns": [t["taskArn"] for t in response.get("tasks", [])],
                "failures": response.get("failures", []),
                "envCount": len(environment),
            }
        )

    return {"runs": runs}
