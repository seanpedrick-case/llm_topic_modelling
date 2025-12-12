import json
import os

import boto3
from dotenv import load_dotenv

# Import the main function from your CLI script
from cli_topics import main as cli_main
from tools.config import (
    AWS_REGION,
    BATCH_SIZE_DEFAULT,
    DEDUPLICATION_THRESHOLD,
    DEFAULT_COST_CODE,
    DEFAULT_SAMPLED_SUMMARIES,
    LLM_MAX_NEW_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    OUTPUT_DEBUG_FILES,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SESSION_OUTPUT_FOLDER,
    USAGE_LOGS_FOLDER,
    convert_string_to_boolean,
)


def _get_env_list(env_var_name: str | list[str] | None) -> list[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    if isinstance(env_var_name, list):
        return env_var_name
    if env_var_name is None:
        return []

    # Handle string input
    value = str(env_var_name).strip()
    if not value or value == "[]":
        return []

    # Remove brackets if present (e.g., "[item1, item2]" -> "item1, item2")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    # Remove quotes and split by comma
    value = value.replace('"', "").replace("'", "")
    if not value:
        return []

    # Split by comma and filter out any empty strings
    return [s.strip() for s in value.split(",") if s.strip()]


print("Lambda entrypoint loading...")

# Initialize S3 client outside the handler for connection reuse
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", AWS_REGION))
print("S3 client initialised")

# Lambda's only writable directory is /tmp. Ensure that all temporary files are stored in this directory.
TMP_DIR = "/tmp"
INPUT_DIR = os.path.join(TMP_DIR, "input")
OUTPUT_DIR = os.path.join(TMP_DIR, "output")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(TMP_DIR, "gradio_tmp")
os.environ["MPLCONFIGDIR"] = os.path.join(TMP_DIR, "matplotlib_cache")
os.environ["FEEDBACK_LOGS_FOLDER"] = os.path.join(TMP_DIR, "feedback")
os.environ["ACCESS_LOGS_FOLDER"] = os.path.join(TMP_DIR, "logs")
os.environ["USAGE_LOGS_FOLDER"] = os.path.join(TMP_DIR, "usage")

# Define compatible file types for processing
COMPATIBLE_FILE_TYPES = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
}


def download_file_from_s3(bucket_name, key, download_path):
    """Download a file from S3 to the local filesystem."""
    try:
        s3_client.download_file(bucket_name, key, download_path)
        print(f"Successfully downloaded file from S3 to {download_path}")
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise


def upload_directory_to_s3(local_directory, bucket_name, s3_prefix):
    """Upload all files from a local directory to an S3 prefix."""
    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            # Create a relative path to maintain directory structure if needed
            relative_path = os.path.relpath(local_file_path, local_directory)
            output_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                s3_client.upload_file(local_file_path, bucket_name, output_key)
                print(
                    f"Successfully uploaded file to S3: {local_file_path}"
                )
            except Exception as e:
                print(f"Error uploading to S3: {e}")
                raise


def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    # 1. Setup temporary directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Extract information from the event
    # Assumes the event is triggered by S3 and may contain an 'arguments' payload
    try:
        record = event["Records"][0]
        bucket_name = record["s3"]["bucket"]["name"]
        input_key = record["s3"]["object"]["key"]

        # The user metadata can be used to pass arguments
        # This is more robust than embedding them in the main event body
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=input_key)
            metadata = response.get("Metadata", dict())
            print(f"S3 object metadata: {metadata}")

            # Arguments can be passed as a JSON string in metadata
            arguments_str = metadata.get("arguments", "{}")
            print(f"Arguments string from metadata: '{arguments_str}'")

            if arguments_str and arguments_str != "{}":
                arguments = json.loads(arguments_str)
                print(f"Successfully parsed arguments from metadata: {arguments}")
            else:
                arguments = dict()
                print("No arguments found in metadata, using empty dictionary")
        except Exception as e:
            print(f"Warning: Could not parse metadata arguments: {e}")
            print("Using empty arguments dictionary")
            arguments = dict()

    except (KeyError, IndexError) as e:
        print(
            f"Could not parse S3 event record: {e}. Checking for direct invocation payload."
        )
        # Fallback for direct invocation (e.g., from Step Functions or manual test)
        bucket_name = event.get("bucket_name")
        input_key = event.get("input_key")
        arguments = event.get("arguments", dict())
        if not all([bucket_name, input_key]):
            raise ValueError(
                "Missing 'bucket_name' or 'input_key' in direct invocation event."
            )

    # Log file type information
    file_extension = os.path.splitext(input_key)[1].lower()
    print(f"Detected file extension: '{file_extension}'")

    # 3. Download the main input file
    input_file_path = os.path.join(INPUT_DIR, os.path.basename(input_key))
    download_file_from_s3(bucket_name, input_key, input_file_path)

    # 3.1. Validate file type compatibility
    is_env_file = input_key.lower().endswith(".env")

    if not is_env_file and file_extension not in COMPATIBLE_FILE_TYPES:
        error_message = f"File type '{file_extension}' is not supported for processing. Compatible file types are: {', '.join(sorted(COMPATIBLE_FILE_TYPES))}"
        print(f"ERROR: {error_message}")
        print(f"File was not processed due to unsupported file type: {file_extension}")
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "Unsupported file type",
                    "message": error_message,
                    "supported_types": list(COMPATIBLE_FILE_TYPES),
                    "received_type": file_extension,
                    "file_processed": False,
                }
            ),
        }

    print(f"File type '{file_extension}' is compatible for processing")
    if is_env_file:
        print("Processing .env file for configuration")
    else:
        print(f"Processing {file_extension} file for topic modelling")

    # 3.5. Check if the downloaded file is a .env file and handle accordingly
    actual_input_file_path = input_file_path
    if input_key.lower().endswith(".env"):
        print("Detected .env file, loading environment variables...")

        # Load environment variables from the .env file
        print(f"Loading .env file from: {input_file_path}")

        # Check if file exists and is readable
        if os.path.exists(input_file_path):
            print(".env file exists and is readable")
            with open(input_file_path, "r") as f:
                content = f.read()
                print(f".env file content preview: {content[:200]}...")
        else:
            print(f"ERROR: .env file does not exist at {input_file_path}")

        load_dotenv(input_file_path, override=True)
        print("Environment variables loaded from .env file")

        # Extract the actual input file path from environment variables
        env_input_file = os.getenv("INPUT_FILE")

        if env_input_file:
            print(f"Found input file path in environment: {env_input_file}")

            # If the path is an S3 path, download it
            if env_input_file.startswith("s3://"):
                # Parse S3 path: s3://bucket/key
                s3_path_parts = env_input_file[5:].split("/", 1)
                if len(s3_path_parts) == 2:
                    env_bucket = s3_path_parts[0]
                    env_key = s3_path_parts[1]
                    actual_input_file_path = os.path.join(
                        INPUT_DIR, os.path.basename(env_key)
                    )
                    print(
                        f"Downloading actual input file from s3://{env_bucket}/{env_key}"
                    )
                    download_file_from_s3(env_bucket, env_key, actual_input_file_path)
                else:
                    print("Warning: Invalid S3 path format in environment variable")
                    actual_input_file_path = input_file_path
            else:
                # Assume it's a local path or relative path
                actual_input_file_path = env_input_file
                print(
                    f"Using input file path from environment: {actual_input_file_path}"
                )
        else:
            print("Warning: No input file path found in environment variables")
            # Fall back to using the .env file itself (though this might not be what we want)
            actual_input_file_path = input_file_path
    else:
        print("File is not a .env file, proceeding with normal processing")

    # 4. Prepare arguments for the CLI function
    # This dictionary should mirror the arguments that cli_topics.main() expects via direct_mode_args

    cli_args = {
        # Task Selection
        "task": arguments.get("task", os.getenv("DIRECT_MODE_TASK", "extract")),
        # General Arguments
        "input_file": [actual_input_file_path] if actual_input_file_path else None,
        "output_dir": arguments.get(
            "output_dir", os.getenv("DIRECT_MODE_OUTPUT_DIR", OUTPUT_DIR)
        ),
        "input_dir": arguments.get("input_dir", INPUT_DIR),
        "text_column": arguments.get(
            "text_column", os.getenv("DIRECT_MODE_TEXT_COLUMN", "")
        ),
        "previous_output_files": _get_env_list(
            arguments.get(
                "previous_output_files",
                os.getenv("DIRECT_MODE_PREVIOUS_OUTPUT_FILES", list()),
            )
        ),
        "username": arguments.get("username", os.getenv("DIRECT_MODE_USERNAME", "")),
        "save_to_user_folders": convert_string_to_boolean(
            arguments.get(
                "save_to_user_folders",
                os.getenv("SESSION_OUTPUT_FOLDER", str(SESSION_OUTPUT_FOLDER)),
            )
        ),
        "excel_sheets": _get_env_list(
            arguments.get("excel_sheets", os.getenv("DIRECT_MODE_EXCEL_SHEETS", list()))
        ),
        "group_by": arguments.get("group_by", os.getenv("DIRECT_MODE_GROUP_BY", "")),
        # Model Configuration
        "model_choice": arguments.get(
            "model_choice", os.getenv("DIRECT_MODE_MODEL_CHOICE", "")
        ),
        "temperature": float(
            arguments.get(
                "temperature",
                os.getenv("DIRECT_MODE_TEMPERATURE", str(LLM_TEMPERATURE)),
            )
        ),
        "batch_size": int(
            arguments.get(
                "batch_size",
                os.getenv("DIRECT_MODE_BATCH_SIZE", str(BATCH_SIZE_DEFAULT)),
            )
        ),
        "max_tokens": int(
            arguments.get(
                "max_tokens",
                os.getenv("DIRECT_MODE_MAX_TOKENS", str(LLM_MAX_NEW_TOKENS)),
            )
        ),
        "google_api_key": arguments.get(
            "google_api_key", os.getenv("GEMINI_API_KEY", "")
        ),
        "aws_access_key": None,  # Use IAM Role instead of keys
        "aws_secret_key": None,  # Use IAM Role instead of keys
        "aws_region": os.getenv("AWS_REGION", AWS_REGION),
        "hf_token": arguments.get("hf_token", os.getenv("HF_TOKEN", "")),
        "azure_api_key": arguments.get(
            "azure_api_key", os.getenv("AZURE_OPENAI_API_KEY", "")
        ),
        "azure_endpoint": arguments.get(
            "azure_endpoint", os.getenv("AZURE_OPENAI_INFERENCE_ENDPOINT", "")
        ),
        "api_url": arguments.get("api_url", os.getenv("API_URL", "")),
        "inference_server_model": arguments.get(
            "inference_server_model", os.getenv("CHOSEN_INFERENCE_SERVER_MODEL", "")
        ),
        # Topic Extraction Arguments
        "context": arguments.get("context", os.getenv("DIRECT_MODE_CONTEXT", "")),
        "candidate_topics": arguments.get(
            "candidate_topics", os.getenv("DIRECT_MODE_CANDIDATE_TOPICS", "")
        ),
        "force_zero_shot": arguments.get(
            "force_zero_shot", os.getenv("DIRECT_MODE_FORCE_ZERO_SHOT", "No")
        ),
        "force_single_topic": arguments.get(
            "force_single_topic", os.getenv("DIRECT_MODE_FORCE_SINGLE_TOPIC", "No")
        ),
        "produce_structured_summary": arguments.get(
            "produce_structured_summary",
            os.getenv("DIRECT_MODE_PRODUCE_STRUCTURED_SUMMARY", "No"),
        ),
        "sentiment": arguments.get(
            "sentiment", os.getenv("DIRECT_MODE_SENTIMENT", "Negative or Positive")
        ),
        "additional_summary_instructions": arguments.get(
            "additional_summary_instructions",
            os.getenv("DIRECT_MODE_ADDITIONAL_SUMMARY_INSTRUCTIONS", ""),
        ),
        # Validation Arguments
        "additional_validation_issues": arguments.get(
            "additional_validation_issues",
            os.getenv("DIRECT_MODE_ADDITIONAL_VALIDATION_ISSUES", ""),
        ),
        "show_previous_table": arguments.get(
            "show_previous_table", os.getenv("DIRECT_MODE_SHOW_PREVIOUS_TABLE", "Yes")
        ),
        "output_debug_files": arguments.get(
            "output_debug_files", str(OUTPUT_DEBUG_FILES)
        ),
        "max_time_for_loop": int(
            arguments.get("max_time_for_loop", os.getenv("MAX_TIME_FOR_LOOP", "99999"))
        ),
        # Deduplication Arguments
        "method": arguments.get(
            "method", os.getenv("DIRECT_MODE_DEDUPLICATION_METHOD", "fuzzy")
        ),
        "similarity_threshold": int(
            arguments.get(
                "similarity_threshold",
                os.getenv("DEDUPLICATION_THRESHOLD", DEDUPLICATION_THRESHOLD),
            )
        ),
        "merge_sentiment": arguments.get(
            "merge_sentiment", os.getenv("DIRECT_MODE_MERGE_SENTIMENT", "No")
        ),
        "merge_general_topics": arguments.get(
            "merge_general_topics", os.getenv("DIRECT_MODE_MERGE_GENERAL_TOPICS", "Yes")
        ),
        # Summarisation Arguments
        "summary_format": arguments.get(
            "summary_format", os.getenv("DIRECT_MODE_SUMMARY_FORMAT", "two_paragraph")
        ),
        "sample_reference_table": arguments.get(
            "sample_reference_table",
            os.getenv("DIRECT_MODE_SAMPLE_REFERENCE_TABLE", "True"),
        ),
        "no_of_sampled_summaries": int(
            arguments.get(
                "no_of_sampled_summaries",
                os.getenv("DEFAULT_SAMPLED_SUMMARIES", DEFAULT_SAMPLED_SUMMARIES),
            )
        ),
        "random_seed": int(
            arguments.get("random_seed", os.getenv("LLM_SEED", LLM_SEED))
        ),
        # Output Format Arguments
        "create_xlsx_output": convert_string_to_boolean(
            arguments.get(
                "create_xlsx_output",
                os.getenv("DIRECT_MODE_CREATE_XLSX_OUTPUT", "True"),
            )
        ),
        # Logging Arguments
        "save_logs_to_csv": convert_string_to_boolean(
            arguments.get(
                "save_logs_to_csv", os.getenv("SAVE_LOGS_TO_CSV", str(SAVE_LOGS_TO_CSV))
            )
        ),
        "save_logs_to_dynamodb": convert_string_to_boolean(
            arguments.get(
                "save_logs_to_dynamodb",
                os.getenv("SAVE_LOGS_TO_DYNAMODB", str(SAVE_LOGS_TO_DYNAMODB)),
            )
        ),
        "usage_logs_folder": arguments.get("usage_logs_folder", USAGE_LOGS_FOLDER),
        "cost_code": arguments.get(
            "cost_code", os.getenv("DEFAULT_COST_CODE", DEFAULT_COST_CODE)
        ),
    }

    # Download optional files if they are specified
    candidate_topics_key = arguments.get("candidate_topics_s3_key")
    if candidate_topics_key:
        candidate_topics_path = os.path.join(INPUT_DIR, "candidate_topics.csv")
        download_file_from_s3(bucket_name, candidate_topics_key, candidate_topics_path)
        cli_args["candidate_topics"] = candidate_topics_path

    # Download previous output files if they are S3 keys
    if cli_args["previous_output_files"]:
        downloaded_previous_files = []
        for prev_file in cli_args["previous_output_files"]:
            if prev_file.startswith("s3://"):
                # Parse S3 path
                s3_path_parts = prev_file[5:].split("/", 1)
                if len(s3_path_parts) == 2:
                    prev_bucket = s3_path_parts[0]
                    prev_key = s3_path_parts[1]
                    local_prev_path = os.path.join(
                        INPUT_DIR, os.path.basename(prev_key)
                    )
                    download_file_from_s3(prev_bucket, prev_key, local_prev_path)
                    downloaded_previous_files.append(local_prev_path)
                else:
                    downloaded_previous_files.append(prev_file)
            else:
                downloaded_previous_files.append(prev_file)
        cli_args["previous_output_files"] = downloaded_previous_files

    # 5. Execute the main application logic
    try:
        print("--- Starting CLI Topics Main Function ---")
        print(
            f"Arguments passed to cli_main: {json.dumps({k: v for k, v in cli_args.items() if k not in ['aws_access_key', 'aws_secret_key', 'google_api_key', 'azure_api_key', 'azure_endpoint', 'api_url', 'hf_token']}, default=str)}"
        )
        cli_main(direct_mode_args=cli_args)
        print("--- CLI Topics Main Function Finished ---")
    except Exception as e:
        print(f"An error occurred during CLI execution: {e}")
        import traceback

        traceback.print_exc()
        # Optionally, re-raise the exception to make the Lambda fail
        raise

    # 6. Upload results back to S3
    output_s3_prefix = f"output/{os.path.splitext(os.path.basename(input_key))[0]}"
    print(
        f"Uploading contents of {OUTPUT_DIR} to s3://{bucket_name}/{output_s3_prefix}/"
    )
    upload_directory_to_s3(OUTPUT_DIR, bucket_name, output_s3_prefix)

    return {
        "statusCode": 200,
        "body": json.dumps(
            f"Processing complete for {input_key}. Output saved to s3://{bucket_name}/{output_s3_prefix}/"
        ),
    }
