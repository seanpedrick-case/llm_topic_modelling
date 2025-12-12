import argparse
import csv
import os
import re
import time
import uuid
from datetime import datetime

import boto3
import botocore
import pandas as pd

from tools.aws_functions import download_file_from_s3, export_outputs_to_s3
from tools.combine_sheets_into_xlsx import collect_output_csvs_and_create_excel_output
from tools.config import (
    API_URL,
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    BATCH_SIZE_DEFAULT,
    CHOSEN_INFERENCE_SERVER_MODEL,
    CSV_USAGE_LOG_HEADERS,
    DEDUPLICATION_THRESHOLD,
    DEFAULT_COST_CODE,
    DEFAULT_SAMPLED_SUMMARIES,
    DYNAMODB_USAGE_LOG_HEADERS,
    GEMINI_API_KEY,
    GRADIO_TEMP_DIR,
    HF_TOKEN,
    INPUT_FOLDER,
    LLM_MAX_NEW_TOKENS,
    LLM_SEED,
    LLM_TEMPERATURE,
    MAX_TIME_FOR_LOOP,
    OUTPUT_DEBUG_FILES,
    OUTPUT_FOLDER,
    RUN_AWS_FUNCTIONS,
    S3_OUTPUTS_BUCKET,
    S3_OUTPUTS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SAVE_OUTPUTS_TO_S3,
    SESSION_OUTPUT_FOLDER,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_FILE_NAME,
    USAGE_LOGS_FOLDER,
    convert_string_to_boolean,
    default_model_choice,
    default_model_source,
    model_name_map,
)
from tools.dedup_summaries import (
    deduplicate_topics,
    deduplicate_topics_llm,
    overall_summary,
    wrapper_summarise_output_topics_per_group,
)
from tools.helper_functions import (
    load_in_data_file,
    load_in_previous_data_files,
)
from tools.llm_api_call import (
    all_in_one_pipeline,
    validate_topics_wrapper,
    wrapper_extract_topics_per_column_value,
)
from tools.prompts import (
    add_existing_topics_prompt,
    add_existing_topics_system_prompt,
    initial_table_prompt,
    initial_table_system_prompt,
    single_para_summary_format_prompt,
    two_para_summary_format_prompt,
)


def _generate_session_hash() -> str:
    """Generate a unique session hash for logging purposes."""
    return str(uuid.uuid4())[:8]


def _download_s3_file_if_needed(
    file_path: str,
    default_filename: str = "downloaded_file",
    aws_access_key: str = "",
    aws_secret_key: str = "",
    aws_region: str = "",
) -> str:
    """
    Download a file from S3 if the path starts with 's3://' or 'S3://', otherwise return the path as-is.

    Args:
        file_path: File path (either local or S3 URL)
        default_filename: Default filename to use if S3 key doesn't have a filename
        aws_access_key: AWS access key ID (optional, uses environment/config if not provided)
        aws_secret_key: AWS secret access key (optional, uses environment/config if not provided)
        aws_region: AWS region (optional, uses environment/config if not provided)

    Returns:
        Local file path (downloaded from S3 or original path)
    """
    if not file_path:
        return file_path

    # Check for S3 URL (case-insensitive)
    file_path_stripped = file_path.strip()
    file_path_upper = file_path_stripped.upper()
    if not file_path_upper.startswith("S3://"):
        return file_path

    # Ensure temp directory exists
    os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)

    # Parse S3 URL: s3://bucket/key (preserve original case for bucket/key)
    # Remove 's3://' prefix (case-insensitive)
    s3_path = (
        file_path_stripped.split("://", 1)[1]
        if "://" in file_path_stripped
        else file_path_stripped
    )
    # Split bucket and key (first '/' separates bucket from key)
    if "/" in s3_path:
        bucket_name_s3, s3_key = s3_path.split("/", 1)
    else:
        # If no key provided, use bucket name as key (unlikely but handle it)
        bucket_name_s3 = s3_path
        s3_key = ""

    # Get the filename from the S3 key
    filename = os.path.basename(s3_key) if s3_key else bucket_name_s3
    if not filename:
        filename = default_filename

    # Create local file path in temp directory
    local_file_path = os.path.join(GRADIO_TEMP_DIR, filename)

    # Download file from S3
    try:
        download_file_from_s3(
            bucket_name=bucket_name_s3,
            key=s3_key,
            local_file_path=local_file_path,
            aws_access_key_textbox=aws_access_key,
            aws_secret_key_textbox=aws_secret_key,
            aws_region_textbox=aws_region,
        )
        print(f"S3 file downloaded successfully: {file_path} -> {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading file from S3 ({file_path}): {e}")
        raise Exception(f"Failed to download file from S3: {e}")


def get_username_and_folders(
    username: str = "",
    output_folder_textbox: str = OUTPUT_FOLDER,
    input_folder_textbox: str = INPUT_FOLDER,
    session_output_folder: bool = SESSION_OUTPUT_FOLDER,
):
    """Generate session hash and set up output/input folders."""
    # Generate session hash for logging. Either from input user name or generated
    if username:
        out_session_hash = username
    else:
        out_session_hash = _generate_session_hash()

    if session_output_folder:
        output_folder = output_folder_textbox + out_session_hash + "/"
        input_folder = input_folder_textbox + out_session_hash + "/"
    else:
        output_folder = output_folder_textbox
        input_folder = input_folder_textbox

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)

    return (
        out_session_hash,
        output_folder,
        out_session_hash,
        input_folder,
    )


def _sanitize_folder_name(folder_name: str, max_length: int = 50) -> str:
    """
    Sanitize folder name for S3 compatibility.

    Replaces 'strange' characters (anything that's not alphanumeric, dash, underscore, or full stop)
    with underscores, and limits the length to max_length characters.

    Args:
        folder_name: Original folder name to sanitize
        max_length: Maximum length for the folder name (default: 50)

    Returns:
        Sanitized folder name
    """
    if not folder_name:
        return folder_name

    # Replace any character that's not alphanumeric, dash, underscore, or full stop with underscore
    # This handles @, commas, exclamation marks, spaces, etc.
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", folder_name)

    # Limit length to max_length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def upload_outputs_to_s3_if_enabled(
    output_files: list,
    base_file_name: str = None,
    session_hash: str = "",
    s3_output_folder: str = S3_OUTPUTS_FOLDER,
    s3_bucket: str = S3_OUTPUTS_BUCKET,
    save_outputs_to_s3: bool = None,
):
    """
    Upload output files to S3 if SAVE_OUTPUTS_TO_S3 is enabled.

    Args:
        output_files: List of output file paths to upload
        base_file_name: Base file name (input file) for organizing S3 folder structure
        session_hash: Session hash to include in S3 path
        s3_output_folder: S3 output folder path
        s3_bucket: S3 bucket name
        save_outputs_to_s3: Override for SAVE_OUTPUTS_TO_S3 config (if None, uses config value)
    """
    # Use provided value or fall back to config
    if save_outputs_to_s3 is None:
        save_outputs_to_s3 = convert_string_to_boolean(SAVE_OUTPUTS_TO_S3)

    if not save_outputs_to_s3:
        return

    if not s3_bucket:
        print("Warning: S3_OUTPUTS_BUCKET not configured. Skipping S3 upload.")
        return

    if not output_files:
        print("No output files to upload to S3.")
        return

    # Filter out empty/None values and ensure files exist
    valid_files = []
    for file_path in output_files:
        if file_path and os.path.exists(file_path):
            valid_files.append(file_path)
        elif file_path:
            print(f"Warning: Output file does not exist, skipping: {file_path}")

    if not valid_files:
        print("No valid output files to upload to S3.")
        return

    # Construct S3 output folder path
    # Include session hash if provided and SESSION_OUTPUT_FOLDER is enabled
    s3_folder_path = s3_output_folder or ""
    if session_hash and convert_string_to_boolean(SESSION_OUTPUT_FOLDER):
        if s3_folder_path and not s3_folder_path.endswith("/"):
            s3_folder_path += "/"
        # Sanitize session_hash to ensure S3 compatibility
        sanitized_session_hash = _sanitize_folder_name(session_hash)
        s3_folder_path += sanitized_session_hash + "/"

    print(f"\nUploading {len(valid_files)} output file(s) to S3...")
    try:
        export_outputs_to_s3(
            file_list_state=valid_files,
            s3_output_folder_state_value=s3_folder_path,
            save_outputs_to_s3_flag=True,
            base_file_state=base_file_name,
            s3_bucket=s3_bucket,
        )
    except Exception as e:
        print(f"Warning: Failed to upload outputs to S3: {e}")


def write_usage_log(
    session_hash: str,
    file_name: str,
    text_column: str,
    model_choice: str,
    conversation_metadata: str,
    input_tokens: int,
    output_tokens: int,
    number_of_calls: int,
    estimated_time_taken: float,
    cost_code: str = DEFAULT_COST_CODE,
    save_to_csv: bool = SAVE_LOGS_TO_CSV,
    save_to_dynamodb: bool = SAVE_LOGS_TO_DYNAMODB,
    include_conversation_metadata: bool = False,
):
    """
    Write usage log entry to CSV file and/or DynamoDB.

    Args:
        session_hash: Session identifier
        file_name: Name of the input file
        text_column: Column name used for analysis (as list for CSV)
        model_choice: LLM model used
        conversation_metadata: Metadata string
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        number_of_calls: Number of LLM calls
        estimated_time_taken: Time taken in seconds
        cost_code: Cost code for tracking
        save_to_csv: Whether to save to CSV
        save_to_dynamodb: Whether to save to DynamoDB
        include_conversation_metadata: Whether to include conversation metadata in the log
    """
    # Convert boolean parameters if they're strings
    if isinstance(save_to_csv, str):
        save_to_csv = convert_string_to_boolean(save_to_csv)
    if isinstance(save_to_dynamodb, str):
        save_to_dynamodb = convert_string_to_boolean(save_to_dynamodb)

    # Return early if neither logging method is enabled
    if not save_to_csv and not save_to_dynamodb:
        return

    if not conversation_metadata:
        conversation_metadata = ""

    # Ensure usage logs folder exists
    os.makedirs(USAGE_LOGS_FOLDER, exist_ok=True)

    # Construct full file path
    usage_log_file_path = os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)

    # Prepare data row - order matches app.py component order
    # session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice,
    # conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num,
    # number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop
    data = [
        session_hash,
        file_name,
        (
            text_column
            if isinstance(text_column, str)
            else (text_column[0] if text_column else "")
        ),
        model_choice,
        conversation_metadata if conversation_metadata else "",
        input_tokens,
        output_tokens,
        number_of_calls,
        estimated_time_taken,
        cost_code,
    ]

    # Add id and timestamp
    generated_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    data.extend([generated_id, timestamp])

    # Use custom headers if available, otherwise use default
    # Note: CSVLogger_custom uses component labels, but we need to match what collect_output_csvs_and_create_excel_output expects
    if CSV_USAGE_LOG_HEADERS and len(CSV_USAGE_LOG_HEADERS) == len(data):
        headers = CSV_USAGE_LOG_HEADERS
    else:
        # Default headers - these should match what CSVLogger_custom creates from Gradio component labels
        # The components are: session_hash_textbox, original_data_file_name_textbox, in_colnames,
        # model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num,
        # number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop
        # Since these are hidden components without labels, CSVLogger_custom uses component variable names
        # or default labels. We need to match what collect_output_csvs_and_create_excel_output expects:
        # "Total LLM calls", "Total input tokens", "Total output tokens"
        # But the actual CSV from Gradio likely has: "Number of calls", "Input tokens", "Output tokens"
        # Let's use the names that match what the Excel function expects
        headers = [
            "Session hash",
            "Reference data file name",
            "Select the open text column of interest. In an Excel file, this shows columns across all sheets.",
            "Large language model for topic extraction and summarisation",
            "Conversation metadata",
            "Total input tokens",  # Changed from "Input tokens" to match Excel function
            "Total output tokens",  # Changed from "Output tokens" to match Excel function
            "Total LLM calls",  # Changed from "Number of calls" to match Excel function
            "Estimated time taken (seconds)",
            "Cost code",
            "id",
            "timestamp",
        ]

    # Write to CSV if enabled
    if save_to_csv:
        # Ensure usage logs folder exists
        os.makedirs(USAGE_LOGS_FOLDER, exist_ok=True)

        # Construct full file path
        usage_log_file_path = os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)

        # Write to CSV
        file_exists = os.path.exists(usage_log_file_path)
        with open(
            usage_log_file_path, "a", newline="", encoding="utf-8-sig"
        ) as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write headers if file doesn't exist
                writer.writerow(headers)
            writer.writerow(data)

    # Write to DynamoDB if enabled
    if save_to_dynamodb:
        # DynamoDB logging implementation
        print("Saving to DynamoDB")

        try:
            # Connect to DynamoDB
            if RUN_AWS_FUNCTIONS == "1":
                try:
                    print("Connecting to DynamoDB via existing SSO connection")
                    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
                    dynamodb.meta.client.list_tables()
                except Exception as e:
                    print("No SSO credentials found:", e)
                    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                        print("Trying DynamoDB credentials from environment variables")
                        dynamodb = boto3.resource(
                            "dynamodb",
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION,
                        )
                    else:
                        raise Exception(
                            "AWS credentials for DynamoDB logging not found"
                        )
            else:
                raise Exception("AWS credentials for DynamoDB logging not found")

            # Get table name from config
            dynamodb_table_name = USAGE_LOG_DYNAMODB_TABLE_NAME
            if not dynamodb_table_name:
                raise ValueError(
                    "USAGE_LOG_DYNAMODB_TABLE_NAME not configured. Cannot save to DynamoDB."
                )

            # Determine headers for DynamoDB
            # Use DYNAMODB_USAGE_LOG_HEADERS if available and matches data length,
            # otherwise use CSV_USAGE_LOG_HEADERS if it matches, otherwise use default headers
            # Note: headers and data are guaranteed to have the same length and include id/timestamp
            if DYNAMODB_USAGE_LOG_HEADERS and len(DYNAMODB_USAGE_LOG_HEADERS) == len(
                data
            ):
                dynamodb_headers = list(DYNAMODB_USAGE_LOG_HEADERS)  # Make a copy
            elif CSV_USAGE_LOG_HEADERS and len(CSV_USAGE_LOG_HEADERS) == len(data):
                dynamodb_headers = list(CSV_USAGE_LOG_HEADERS)  # Make a copy
            else:
                # Use the headers we created which are guaranteed to match data
                dynamodb_headers = headers

            # Check if table exists, create if it doesn't
            try:
                table = dynamodb.Table(dynamodb_table_name)
                table.load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    print(
                        f"Table '{dynamodb_table_name}' does not exist. Creating it..."
                    )
                    attribute_definitions = [
                        {
                            "AttributeName": "id",
                            "AttributeType": "S",
                        }
                    ]

                    table = dynamodb.create_table(
                        TableName=dynamodb_table_name,
                        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                        AttributeDefinitions=attribute_definitions,
                        BillingMode="PAY_PER_REQUEST",
                    )
                    # Wait until the table exists
                    table.meta.client.get_waiter("table_exists").wait(
                        TableName=dynamodb_table_name
                    )
                    time.sleep(5)
                    print(f"Table '{dynamodb_table_name}' created successfully.")
                else:
                    raise

            # Prepare the DynamoDB item to upload
            # Map the headers to values (headers and data should match in length)
            if len(dynamodb_headers) == len(data):
                item = {
                    header: str(value) for header, value in zip(dynamodb_headers, data)
                }
            else:
                # Fallback: use the default headers which are guaranteed to match data
                print(
                    f"Warning: DynamoDB headers length ({len(dynamodb_headers)}) doesn't match data length ({len(data)}). Using default headers."
                )
                item = {header: str(value) for header, value in zip(headers, data)}

            # Upload to DynamoDB
            table.put_item(Item=item)
            print("Successfully uploaded log to DynamoDB")

        except Exception as e:
            print(f"Could not upload log to DynamoDB due to: {e}")
            import traceback

            traceback.print_exc()


# --- Main CLI Function ---
def main(direct_mode_args={}):
    """
    A unified command-line interface for topic extraction, validation, deduplication, and summarisation.

    Args:
        direct_mode_args (dict, optional): Dictionary of arguments for direct mode execution.
                                          If provided, uses these instead of parsing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A versatile CLI for topic extraction, validation, deduplication, and summarisation using LLMs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:

To run these, you need to do the following:

- Open a terminal window

- CD to the app folder that contains this file (cli_topics.py)

- Load the virtual environment using either conda or venv depending on your setup

- Run one of the example commands below

- The examples below use the free Gemini 2.5 Flash Lite model, that is free with an API key that you can get from here: https://aistudio.google.com/api-keys. You can either set this or API keys for other services as an environment variable (e.g. in config/app_config.py. See the file tools/config.py for more details about variables relevant to each service) or you can set them manually at the time of the function call via command line arguments, such as the following:

    Google/Gemini: --google_api_key
    AWS Bedrock: --aws_access_key, --aws_secret_key, --aws_region
    Hugging Face (for model download): --hf_token
    Azure/OpenAI: --azure_api_key, --azure_endpoint
    Inference Server endpoint for local models(e.g. llama server, vllm): --api_url

- Use --create_xlsx_output to create an Excel file combining all CSV outputs after task completion

- Look in the output/ folder to see output files:

# Topic Extraction

## Extract topics from a CSV file with default settings:
python cli_topics.py --task extract --input_file example_data/combined_case_notes.csv --text_column "Case Note"

## Extract topics with custom model and context:
python cli_topics.py --task extract --input_file example_data/combined_case_notes.csv --text_column "Case Note" --model_choice "gemini-2.5-flash-lite" --context "Social Care case notes for young people"

## Extract topics with grouping:
python cli_topics.py --task extract --input_file example_data/combined_case_notes.csv --text_column "Case Note" --group_by "Client"

## Extract topics with candidate topics (zero-shot):
python cli_topics.py --task extract --input_file example_data/dummy_consultation_response.csv --text_column "Response text" --candidate_topics example_data/dummy_consultation_response_themes.csv

# Topic Validation

## Validate previously extracted topics:
python cli_topics.py --task validate --input_file example_data/combined_case_notes.csv --text_column "Case Note" --previous_output_files output/combined_case_notes_col_Case_Note_reference_table.csv output/combined_case_notes_col_Case_Note_unique_topics.csv

# Deduplication

Note: you will need to change the reference to previous output files to match the exact file names created from the previous task. This includes the relative path to the app folder. Also, the function will create an xlsx output file by default. the --input_file and --text_column arguments are needed for this, unless you pass in --no_xlsx_output as seen below.

## Deduplicate topics using fuzzy matching:
python cli_topics.py --task deduplicate --previous_output_files output/combined_case_notes_col_Case_Note_reference_table.csv output/combined_case_notes_col_Case_Note_unique_topics.csv --similarity_threshold 90 --no_xlsx_output

## Deduplicate topics using LLM:
python cli_topics.py --task deduplicate --previous_output_files output/combined_case_notes_col_Case_Note_reference_table.csv output/combined_case_notes_col_Case_Note_unique_topics.csv --method llm --model_choice "gemini-2.5-flash-lite" --no_xlsx_output

# Summarisation

Note: you will need to change the reference to previous output files to match the exact file names created from the previous task. This includes the relative path to the app folder. Also, the function will create an xlsx output file by default. the --input_file and --text_column arguments are needed for this, unless you pass in --no_xlsx_output as seen below.

## Summarise topics:
python cli_topics.py --task summarise --previous_output_files output/combined_case_notes_col_Case_Note_reference_table.csv output/combined_case_notes_col_Case_Note_unique_topics.csv --model_choice "gemini-2.5-flash-lite" --no_xlsx_output

## Create overall summary:
python cli_topics.py --task overall_summary --previous_output_files output/combined_case_notes_col_Case_Note_unique_topics.csv --model_choice "gemini-2.5-flash-lite" --no_xlsx_output

# All-in-one pipeline

## Run complete pipeline (extract, deduplicate, summarise):
python cli_topics.py --task all_in_one --input_file example_data/combined_case_notes.csv --text_column "Case Note" --model_choice "gemini-2.5-flash-lite"

""",
    )

    # --- Task Selection ---
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task",
        choices=[
            "extract",
            "validate",
            "deduplicate",
            "summarise",
            "overall_summary",
            "all_in_one",
        ],
        default="extract",
        help="Task to perform: extract (topic extraction), validate (validate topics), deduplicate (deduplicate topics), summarise (summarise topics), overall_summary (create overall summary), or all_in_one (complete pipeline).",
    )

    # --- General Arguments ---
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--input_file",
        nargs="+",
        help="Path to the input file(s) to process. Separate multiple files with a space, and use quotes if there are spaces in the file name.",
    )
    general_group.add_argument(
        "--output_dir", default=OUTPUT_FOLDER, help="Directory for all output files."
    )
    general_group.add_argument(
        "--input_dir", default=INPUT_FOLDER, help="Directory for all input files."
    )
    general_group.add_argument(
        "--text_column",
        help="Name of the text column to process (required for extract, validate, and all_in_one tasks).",
    )
    general_group.add_argument(
        "--previous_output_files",
        nargs="+",
        help="Path(s) to previous output files (reference_table and/or unique_topics files) for validate, deduplicate, summarise, and overall_summary tasks.",
    )
    general_group.add_argument(
        "--username", default="", help="Username for the session."
    )
    general_group.add_argument(
        "--save_to_user_folders",
        default=SESSION_OUTPUT_FOLDER,
        help="Whether to save to user folders or not.",
    )
    general_group.add_argument(
        "--excel_sheets",
        nargs="+",
        default=list(),
        help="Specific Excel sheet names to process.",
    )
    general_group.add_argument(
        "--group_by",
        help="Column name to group results by.",
    )

    # --- Model Configuration ---
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_choice",
        default=default_model_choice,
        help=f"LLM model to use. Default: {default_model_choice}",
    )
    model_group.add_argument(
        "--model_source",
        default=default_model_source,
        help=f"Model source (e.g., 'Google', 'AWS', 'Local'). Default: {default_model_source}",
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help=f"Temperature for LLM generation. Default: {LLM_TEMPERATURE}",
    )
    model_group.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Number of responses to submit in a single LLM query. Default: {BATCH_SIZE_DEFAULT}",
    )
    model_group.add_argument(
        "--max_tokens",
        type=int,
        default=LLM_MAX_NEW_TOKENS,
        help=f"Maximum tokens for LLM generation. Default: {LLM_MAX_NEW_TOKENS}",
    )
    model_group.add_argument(
        "--google_api_key",
        default=GEMINI_API_KEY,
        help="Google API key for Gemini models.",
    )
    model_group.add_argument(
        "--aws_access_key",
        default=AWS_ACCESS_KEY,
        help="AWS Access Key ID for Bedrock models.",
    )
    model_group.add_argument(
        "--aws_secret_key",
        default=AWS_SECRET_KEY,
        help="AWS Secret Access Key for Bedrock models.",
    )
    model_group.add_argument(
        "--aws_region",
        default=AWS_REGION,
        help="AWS region for Bedrock models.",
    )
    model_group.add_argument(
        "--hf_token",
        default=HF_TOKEN,
        help="Hugging Face token for downloading gated models.",
    )
    model_group.add_argument(
        "--azure_api_key",
        default=AZURE_OPENAI_API_KEY,
        help="Azure/OpenAI API key for Azure/OpenAI models.",
    )
    model_group.add_argument(
        "--azure_endpoint",
        default=AZURE_OPENAI_INFERENCE_ENDPOINT,
        help="Azure Inference endpoint URL.",
    )
    model_group.add_argument(
        "--api_url",
        default=API_URL,
        help=f"Inference server API URL (for local models). Default: {API_URL}",
    )
    model_group.add_argument(
        "--inference_server_model",
        default=CHOSEN_INFERENCE_SERVER_MODEL,
        help=f"Inference server model name to use. Default: {CHOSEN_INFERENCE_SERVER_MODEL}",
    )

    # --- Topic Extraction Arguments ---
    extract_group = parser.add_argument_group("Topic Extraction Options")
    extract_group.add_argument(
        "--context",
        default="",
        help="Context sentence to provide to the LLM for topic extraction.",
    )
    extract_group.add_argument(
        "--candidate_topics",
        help="Path to CSV file with candidate topics for zero-shot extraction.",
    )
    extract_group.add_argument(
        "--force_zero_shot",
        choices=["Yes", "No"],
        default="No",
        help="Force responses into suggested topics. Default: No",
    )
    extract_group.add_argument(
        "--force_single_topic",
        choices=["Yes", "No"],
        default="No",
        help="Ask the model to assign responses to only a single topic. Default: No",
    )
    extract_group.add_argument(
        "--produce_structured_summary",
        choices=["Yes", "No"],
        default="No",
        help="Produce structured summaries using suggested topics as headers. Default: No",
    )
    extract_group.add_argument(
        "--sentiment",
        choices=[
            "Negative or Positive",
            "Negative, Neutral, or Positive",
            "Do not assess sentiment",
        ],
        default="Negative or Positive",
        help="Response sentiment analysis option. Default: Negative or Positive",
    )
    extract_group.add_argument(
        "--additional_summary_instructions",
        default="",
        help="Additional instructions for summary format.",
    )

    # --- Validation Arguments ---
    validate_group = parser.add_argument_group("Topic Validation Options")
    validate_group.add_argument(
        "--additional_validation_issues",
        default="",
        help="Additional validation issues for the model to consider (bullet-point list).",
    )
    validate_group.add_argument(
        "--show_previous_table",
        choices=["Yes", "No"],
        default="Yes",
        help="Provide response data to validation process. Default: Yes",
    )
    validate_group.add_argument(
        "--output_debug_files",
        choices=["True", "False"],
        default=OUTPUT_DEBUG_FILES,
        help=f"Output debug files. Default: {OUTPUT_DEBUG_FILES}",
    )
    validate_group.add_argument(
        "--max_time_for_loop",
        type=int,
        default=MAX_TIME_FOR_LOOP,
        help=f"Maximum time for validation loop in seconds. Default: {MAX_TIME_FOR_LOOP}",
    )

    # --- Deduplication Arguments ---
    dedup_group = parser.add_argument_group("Deduplication Options")
    dedup_group.add_argument(
        "--method",
        choices=["fuzzy", "llm"],
        default="fuzzy",
        help="Deduplication method: fuzzy (fuzzy matching) or llm (LLM semantic matching). Default: fuzzy",
    )
    dedup_group.add_argument(
        "--similarity_threshold",
        type=int,
        default=DEDUPLICATION_THRESHOLD,
        help=f"Similarity threshold (0-100) for fuzzy matching. Default: {DEDUPLICATION_THRESHOLD}",
    )
    dedup_group.add_argument(
        "--merge_sentiment",
        choices=["Yes", "No"],
        default="No",
        help="Merge sentiment values together for duplicate subtopics. Default: No",
    )
    dedup_group.add_argument(
        "--merge_general_topics",
        choices=["Yes", "No"],
        default="Yes",
        help="Merge general topic values together for duplicate subtopics. Default: Yes",
    )

    # --- Summarisation Arguments ---
    summarise_group = parser.add_argument_group("Summarisation Options")
    summarise_group.add_argument(
        "--summary_format",
        choices=["two_paragraph", "single_paragraph"],
        default="two_paragraph",
        help="Summary format type. Default: two_paragraph",
    )
    summarise_group.add_argument(
        "--sample_reference_table",
        choices=["True", "False"],
        default="True",
        help="Sample reference table (recommended for large datasets). Default: True",
    )
    summarise_group.add_argument(
        "--no_of_sampled_summaries",
        type=int,
        default=DEFAULT_SAMPLED_SUMMARIES,
        help=f"Number of summaries per group. Default: {DEFAULT_SAMPLED_SUMMARIES}",
    )
    summarise_group.add_argument(
        "--random_seed",
        type=int,
        default=LLM_SEED,
        help=f"Random seed for sampling. Default: {LLM_SEED}",
    )

    # --- Output Format Arguments ---
    output_group = parser.add_argument_group("Output Format Options")
    output_group.add_argument(
        "--no_xlsx_output",
        dest="create_xlsx_output",
        action="store_false",
        default=True,
        help="Disable creation of Excel (.xlsx) output file. By default, Excel output is created.",
    )

    # --- Logging Arguments ---
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "--save_logs_to_csv",
        default=SAVE_LOGS_TO_CSV,
        help="Save processing logs to CSV files.",
    )
    logging_group.add_argument(
        "--save_logs_to_dynamodb",
        default=SAVE_LOGS_TO_DYNAMODB,
        help="Save processing logs to DynamoDB.",
    )
    logging_group.add_argument(
        "--usage_logs_folder",
        default=USAGE_LOGS_FOLDER,
        help="Directory for usage log files.",
    )
    logging_group.add_argument(
        "--cost_code",
        default=DEFAULT_COST_CODE,
        help="Cost code for tracking usage.",
    )

    # Parse arguments - either from command line or direct mode
    if direct_mode_args:
        # Use direct mode arguments
        args = argparse.Namespace(**direct_mode_args)
    else:
        # Parse command line arguments
        args = parser.parse_args()

    # --- Handle S3 file downloads ---
    # Get AWS credentials from args or fall back to config values
    aws_access_key = getattr(args, "aws_access_key", None) or AWS_ACCESS_KEY or ""
    aws_secret_key = getattr(args, "aws_secret_key", None) or AWS_SECRET_KEY or ""
    aws_region = getattr(args, "aws_region", None) or AWS_REGION or ""

    # Download input files from S3 if needed
    # Note: args.input_file is typically a list (from CLI nargs="+" or from direct mode)
    # but we also handle pipe-separated strings for compatibility
    if args.input_file:
        if isinstance(args.input_file, list):
            # Handle list of files (may include S3 paths)
            downloaded_files = []
            for file_path in args.input_file:
                downloaded_path = _download_s3_file_if_needed(
                    file_path,
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_region=aws_region,
                )
                downloaded_files.append(downloaded_path)
            args.input_file = downloaded_files
        elif isinstance(args.input_file, str):
            # Handle pipe-separated string (for direct mode compatibility)
            if "|" in args.input_file:
                file_list = [f.strip() for f in args.input_file.split("|") if f.strip()]
                downloaded_files = []
                for file_path in file_list:
                    downloaded_path = _download_s3_file_if_needed(
                        file_path,
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key,
                        aws_region=aws_region,
                    )
                    downloaded_files.append(downloaded_path)
                args.input_file = downloaded_files
            else:
                # Single file path
                args.input_file = [
                    _download_s3_file_if_needed(
                        args.input_file,
                        aws_access_key=aws_access_key,
                        aws_secret_key=aws_secret_key,
                        aws_region=aws_region,
                    )
                ]

    # Download candidate topics file from S3 if needed
    if args.candidate_topics:
        args.candidate_topics = _download_s3_file_if_needed(
            args.candidate_topics,
            default_filename="downloaded_candidate_topics",
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )

    # --- Override model_choice with inference_server_model if provided ---
    # If inference_server_model is explicitly provided, use it to override model_choice
    # This allows users to specify which inference-server model to use
    if args.inference_server_model:
        # Check if the current model_choice is an inference-server model
        model_source = model_name_map.get(args.model_choice, {}).get(
            "source", default_model_source
        )
        # If model_source is "inference-server" OR if inference_server_model is explicitly provided
        # (different from default), use it
        if (
            model_source == "inference-server"
            or args.inference_server_model != CHOSEN_INFERENCE_SERVER_MODEL
        ):
            args.model_choice = args.inference_server_model
            # Ensure the model is registered in model_name_map with inference-server source
            if args.model_choice not in model_name_map:
                model_name_map[args.model_choice] = {
                    "short_name": args.model_choice,
                    "source": "inference-server",
                }
            # Also update the model_source to ensure it's set correctly
            model_name_map[args.model_choice]["source"] = "inference-server"

    # --- Initial Setup ---
    # Convert string boolean variables to boolean
    args.save_to_user_folders = convert_string_to_boolean(args.save_to_user_folders)
    args.save_logs_to_csv = convert_string_to_boolean(str(args.save_logs_to_csv))
    args.save_logs_to_dynamodb = convert_string_to_boolean(
        str(args.save_logs_to_dynamodb)
    )
    args.sample_reference_table = args.sample_reference_table == "True"
    args.output_debug_files = args.output_debug_files == "True"

    # Get username and folders
    (
        session_hash,
        args.output_dir,
        _,
        args.input_dir,
    ) = get_username_and_folders(
        username=args.username,
        output_folder_textbox=args.output_dir,
        input_folder_textbox=args.input_dir,
        session_output_folder=args.save_to_user_folders,
    )

    print(
        f"Conducting analyses with user {args.username or session_hash}. Outputs will be saved to {args.output_dir}."
    )

    # --- Route to the Correct Workflow Based on Task ---

    # Validate input_file requirement for tasks that need it
    if args.task in ["extract", "validate", "all_in_one"] and not args.input_file:
        print(f"Error: --input_file is required for '{args.task}' task.")
        return

    if (
        args.task in ["validate", "deduplicate", "summarise", "overall_summary"]
        and not args.previous_output_files
    ):
        print(f"Error: --previous_output_files is required for '{args.task}' task.")
        return

    if args.task in ["extract", "validate", "all_in_one"] and not args.text_column:
        print(f"Error: --text_column is required for '{args.task}' task.")
        return

    start_time = time.time()

    try:
        # Task 1: Extract Topics
        if args.task == "extract":
            print("--- Starting Topic Extraction Workflow... ---")

            # Load data file
            if isinstance(args.input_file, str):
                args.input_file = [args.input_file]

            file_data, file_name, total_number_of_batches = load_in_data_file(
                file_paths=args.input_file,
                in_colnames=[args.text_column],
                batch_size=args.batch_size,
                in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
            )

            # Prepare candidate topics if provided
            candidate_topics = None
            if args.candidate_topics:
                candidate_topics = args.candidate_topics

            # Determine summary format prompt
            summary_format_prompt = (
                two_para_summary_format_prompt
                if args.summary_format == "two_paragraph"
                else single_para_summary_format_prompt
            )

            # Run extraction
            (
                display_markdown,
                master_topic_df_state,
                master_unique_topics_df_state,
                master_reference_df_state,
                topic_extraction_output_files,
                text_output_file_list_state,
                latest_batch_completed,
                log_files_output,
                log_files_output_list_state,
                conversation_metadata_textbox,
                estimated_time_taken_number,
                deduplication_input_files,
                summarisation_input_files,
                modifiable_unique_topics_df_state,
                modification_input_files,
                in_join_files,
                missing_df_state,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                output_messages_textbox,
                logged_content_df,
            ) = wrapper_extract_topics_per_column_value(
                grouping_col=args.group_by,
                in_data_file=args.input_file,
                file_data=file_data,
                initial_existing_topics_table=pd.DataFrame(),
                initial_existing_reference_df=pd.DataFrame(),
                initial_existing_topic_summary_df=pd.DataFrame(),
                initial_unique_table_df_display_table_markdown="",
                original_file_name=file_name,
                total_number_of_batches=total_number_of_batches,
                in_api_key=args.google_api_key,
                temperature=args.temperature,
                chosen_cols=[args.text_column],
                model_choice=args.model_choice,
                candidate_topics=candidate_topics,
                initial_first_loop_state=True,
                initial_all_metadata_content_str="",
                initial_latest_batch_completed=0,
                initial_time_taken=0,
                batch_size=args.batch_size,
                context_textbox=args.context,
                sentiment_checkbox=args.sentiment,
                force_zero_shot_radio=args.force_zero_shot,
                in_excel_sheets=args.excel_sheets,
                force_single_topic_radio=args.force_single_topic,
                produce_structured_summary_radio=args.produce_structured_summary,
                aws_access_key_textbox=args.aws_access_key,
                aws_secret_key_textbox=args.aws_secret_key,
                aws_region_textbox=args.aws_region,
                hf_api_key_textbox=args.hf_token,
                azure_api_key_textbox=args.azure_api_key,
                azure_endpoint_textbox=args.azure_endpoint,
                output_folder=args.output_dir,
                existing_logged_content=list(),
                additional_instructions_summary_format=args.additional_summary_instructions,
                additional_validation_issues_provided="",
                show_previous_table="Yes",
                api_url=args.api_url if args.api_url else API_URL,
                max_tokens=args.max_tokens,
                model_name_map=model_name_map,
                max_time_for_loop=99999,
                reasoning_suffix="",
                CHOSEN_LOCAL_MODEL_TYPE="",
                output_debug_files=str(args.output_debug_files),
                model=None,
                tokenizer=None,
                assistant_model=None,
                max_rows=999999,
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- Topic Extraction Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if topic_extraction_output_files:
                print("Generated Files:", sorted(topic_extraction_output_files))

            # Write usage log (before Excel creation so it can be included in Excel)
            write_usage_log(
                session_hash=session_hash,
                file_name=file_name,
                text_column=args.text_column,
                model_choice=args.model_choice,
                conversation_metadata=conversation_metadata_textbox or "",
                input_tokens=input_tokens_num or 0,
                output_tokens=output_tokens_num or 0,
                number_of_calls=number_of_calls_num or 0,
                estimated_time_taken=estimated_time_taken_number or processing_time,
                cost_code=args.cost_code,
                save_to_csv=args.save_logs_to_csv,
                save_to_dynamodb=args.save_logs_to_dynamodb,
            )

            # Create Excel output if requested
            xlsx_files = []
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file,
                        chosen_cols=[args.text_column],
                        reference_data_file_name_textbox=file_name,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=master_reference_df_state,
                        master_unique_topics_df_state=master_unique_topics_df_state,
                        summarised_output_df=pd.DataFrame(),  # No summaries yet
                        missing_df_state=missing_df_state,
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

            # Upload outputs to S3 if enabled
            all_output_files = (
                list(topic_extraction_output_files)
                if topic_extraction_output_files
                else []
            )
            if xlsx_files:
                all_output_files.extend(xlsx_files)
            upload_outputs_to_s3_if_enabled(
                output_files=all_output_files,
                base_file_name=file_name,
                session_hash=session_hash,
            )

        # Task 2: Validate Topics
        elif args.task == "validate":
            print("--- Starting Topic Validation Workflow... ---")

            # Load data file
            if isinstance(args.input_file, str):
                args.input_file = [args.input_file]

            file_data, file_name, total_number_of_batches = load_in_data_file(
                file_paths=args.input_file,
                in_colnames=[args.text_column],
                batch_size=args.batch_size,
                in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
            )

            # Load previous output files
            (
                reference_df,
                topic_summary_df,
                latest_batch_completed_no_loop,
                deduplication_input_files_status,
                working_data_file_name_textbox,
                unique_topics_table_file_name_textbox,
            ) = load_in_previous_data_files(args.previous_output_files)

            # Run validation
            (
                display_markdown,
                master_topic_df_state,
                master_unique_topics_df_state,
                master_reference_df_state,
                validation_output_files,
                text_output_file_list_state,
                latest_batch_completed,
                log_files_output,
                log_files_output_list_state,
                conversation_metadata_textbox,
                estimated_time_taken_number,
                deduplication_input_files,
                summarisation_input_files,
                modifiable_unique_topics_df_state,
                modification_input_files,
                in_join_files,
                missing_df_state,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                output_messages_textbox,
                logged_content_df,
            ) = validate_topics_wrapper(
                file_data=file_data,
                reference_df=reference_df,
                topic_summary_df=topic_summary_df,
                file_name=working_data_file_name_textbox,
                chosen_cols=[args.text_column],
                batch_size=args.batch_size,
                model_choice=args.model_choice,
                in_api_key=args.google_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                azure_api_key_textbox=args.azure_api_key,
                azure_endpoint_textbox=args.azure_endpoint,
                reasoning_suffix="",
                group_name=args.group_by or "All",
                produce_structured_summary_radio=args.produce_structured_summary,
                force_zero_shot_radio=args.force_zero_shot,
                force_single_topic_radio=args.force_single_topic,
                context_textbox=args.context,
                additional_instructions_summary_format=args.additional_summary_instructions,
                output_folder=args.output_dir,
                output_debug_files=str(args.output_debug_files),
                original_full_file_name=file_name,
                additional_validation_issues_provided=args.additional_validation_issues,
                max_time_for_loop=args.max_time_for_loop,
                in_data_files=args.input_file,
                sentiment_checkbox=args.sentiment,
                logged_content=None,
                show_previous_table=args.show_previous_table,
                aws_access_key_textbox=args.aws_access_key,
                aws_secret_key_textbox=args.aws_secret_key,
                aws_region_textbox=args.aws_region,
                api_url=args.api_url if args.api_url else API_URL,
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- Topic Validation Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if validation_output_files:
                print("Generated Files:", sorted(validation_output_files))

            # Write usage log
            write_usage_log(
                session_hash=session_hash,
                file_name=file_name,
                text_column=args.text_column,
                model_choice=args.model_choice,
                conversation_metadata=conversation_metadata_textbox or "",
                input_tokens=input_tokens_num or 0,
                output_tokens=output_tokens_num or 0,
                number_of_calls=number_of_calls_num or 0,
                estimated_time_taken=estimated_time_taken_number or processing_time,
                cost_code=args.cost_code,
                save_to_csv=args.save_logs_to_csv,
                save_to_dynamodb=args.save_logs_to_dynamodb,
            )

            # Create Excel output if requested
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file,
                        chosen_cols=[args.text_column],
                        reference_data_file_name_textbox=file_name,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=master_reference_df_state,
                        master_unique_topics_df_state=master_unique_topics_df_state,
                        summarised_output_df=pd.DataFrame(),  # No summaries yet
                        missing_df_state=missing_df_state,
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

        # Task 3: Deduplicate Topics
        elif args.task == "deduplicate":
            print("--- Starting Topic Deduplication Workflow... ---")

            # Load previous output files
            (
                reference_df,
                topic_summary_df,
                latest_batch_completed_no_loop,
                deduplication_input_files_status,
                working_data_file_name_textbox,
                unique_topics_table_file_name_textbox,
            ) = load_in_previous_data_files(args.previous_output_files)

            if args.method == "fuzzy":
                # Fuzzy matching deduplication
                (
                    ref_df_after_dedup,
                    unique_df_after_dedup,
                    summarisation_input_files,
                    log_files_output,
                    summarised_output_markdown,
                ) = deduplicate_topics(
                    reference_df=reference_df,
                    topic_summary_df=topic_summary_df,
                    reference_table_file_name=working_data_file_name_textbox,
                    unique_topics_table_file_name=unique_topics_table_file_name_textbox,
                    in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                    merge_sentiment=args.merge_sentiment,
                    merge_general_topics=args.merge_general_topics,
                    score_threshold=args.similarity_threshold,
                    in_data_files=args.input_file if args.input_file else list(),
                    chosen_cols=[args.text_column] if args.text_column else list(),
                    output_folder=args.output_dir,
                )
            else:
                # LLM deduplication
                model_source = model_name_map.get(args.model_choice, {}).get(
                    "source", default_model_source
                )
                (
                    ref_df_after_dedup,
                    unique_df_after_dedup,
                    summarisation_input_files,
                    log_files_output,
                    summarised_output_markdown,
                    input_tokens_num,
                    output_tokens_num,
                    number_of_calls_num,
                    estimated_time_taken_number,
                ) = deduplicate_topics_llm(
                    reference_df=reference_df,
                    topic_summary_df=topic_summary_df,
                    reference_table_file_name=working_data_file_name_textbox,
                    unique_topics_table_file_name=unique_topics_table_file_name_textbox,
                    model_choice=args.model_choice,
                    in_api_key=args.google_api_key,
                    temperature=args.temperature,
                    model_source=model_source,
                    bedrock_runtime=None,
                    local_model=None,
                    tokenizer=None,
                    assistant_model=None,
                    in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                    merge_sentiment=args.merge_sentiment,
                    merge_general_topics=args.merge_general_topics,
                    in_data_files=args.input_file if args.input_file else list(),
                    chosen_cols=[args.text_column] if args.text_column else list(),
                    output_folder=args.output_dir,
                    candidate_topics=(
                        args.candidate_topics if args.candidate_topics else None
                    ),
                    azure_endpoint=args.azure_endpoint,
                    output_debug_files=str(args.output_debug_files),
                    api_url=args.api_url if args.api_url else API_URL,
                )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- Topic Deduplication Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if summarisation_input_files:
                print("Generated Files:", sorted(summarisation_input_files))

            # Write usage log (only for LLM deduplication which has token counts)
            if args.method == "llm":
                # Extract token counts from LLM deduplication result
                llm_input_tokens = (
                    input_tokens_num if "input_tokens_num" in locals() else 0
                )
                llm_output_tokens = (
                    output_tokens_num if "output_tokens_num" in locals() else 0
                )
                llm_calls = (
                    number_of_calls_num if "number_of_calls_num" in locals() else 0
                )
                llm_time = (
                    estimated_time_taken_number
                    if "estimated_time_taken_number" in locals()
                    else processing_time
                )

                write_usage_log(
                    session_hash=session_hash,
                    file_name=working_data_file_name_textbox,
                    text_column=args.text_column if args.text_column else "",
                    model_choice=args.model_choice,
                    conversation_metadata="",
                    input_tokens=llm_input_tokens,
                    output_tokens=llm_output_tokens,
                    number_of_calls=llm_calls,
                    estimated_time_taken=llm_time,
                    cost_code=args.cost_code,
                    save_to_csv=args.save_logs_to_csv,
                    save_to_dynamodb=args.save_logs_to_dynamodb,
                )

            # Create Excel output if requested
            xlsx_files = []
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    # Use the deduplicated dataframes
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file if args.input_file else [],
                        chosen_cols=[args.text_column] if args.text_column else [],
                        reference_data_file_name_textbox=working_data_file_name_textbox,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=ref_df_after_dedup,
                        master_unique_topics_df_state=unique_df_after_dedup,
                        summarised_output_df=pd.DataFrame(),  # No summaries yet
                        missing_df_state=pd.DataFrame(),
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

            # Upload outputs to S3 if enabled
            all_output_files = (
                list(summarisation_input_files) if summarisation_input_files else []
            )
            if xlsx_files:
                all_output_files.extend(xlsx_files)
            upload_outputs_to_s3_if_enabled(
                output_files=all_output_files,
                base_file_name=working_data_file_name_textbox,
                session_hash=session_hash,
            )

        # Task 4: Summarise Topics
        elif args.task == "summarise":
            print("--- Starting Topic Summarisation Workflow... ---")

            # Load previous output files
            (
                reference_df,
                topic_summary_df,
                latest_batch_completed_no_loop,
                deduplication_input_files_status,
                working_data_file_name_textbox,
                unique_topics_table_file_name_textbox,
            ) = load_in_previous_data_files(args.previous_output_files)

            # Determine summary format prompt
            summary_format_prompt = (
                two_para_summary_format_prompt
                if args.summary_format == "two_paragraph"
                else single_para_summary_format_prompt
            )

            # Run summarisation
            (
                summary_reference_table_sample_state,
                master_unique_topics_df_revised_summaries_state,
                master_reference_df_revised_summaries_state,
                summary_output_files,
                summarised_outputs_list,
                latest_summary_completed_num,
                conversation_metadata_textbox,
                summarised_output_markdown,
                log_files_output,
                overall_summarisation_input_files,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                estimated_time_taken_number,
                output_messages_textbox,
                logged_content_df,
            ) = wrapper_summarise_output_topics_per_group(
                grouping_col=args.group_by,
                sampled_reference_table_df=reference_df.copy(),  # Will be sampled if sample_reference_table=True
                topic_summary_df=topic_summary_df,
                reference_table_df=reference_df,
                model_choice=args.model_choice,
                in_api_key=args.google_api_key,
                temperature=args.temperature,
                reference_data_file_name=working_data_file_name_textbox,
                summarised_outputs=list(),
                latest_summary_completed=0,
                out_metadata_str="",
                in_data_files=args.input_file if args.input_file else list(),
                in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                chosen_cols=[args.text_column] if args.text_column else list(),
                log_output_files=list(),
                summarise_format_radio=summary_format_prompt,
                output_folder=args.output_dir,
                context_textbox=args.context,
                aws_access_key_textbox=args.aws_access_key,
                aws_secret_key_textbox=args.aws_secret_key,
                aws_region_textbox=args.aws_region,
                model_name_map=model_name_map,
                hf_api_key_textbox=args.hf_token,
                azure_endpoint_textbox=args.azure_endpoint,
                existing_logged_content=list(),
                sample_reference_table=args.sample_reference_table,
                no_of_sampled_summaries=args.no_of_sampled_summaries,
                random_seed=args.random_seed,
                api_url=args.api_url if args.api_url else API_URL,
                additional_summary_instructions_provided=args.additional_summary_instructions,
                output_debug_files=str(args.output_debug_files),
                reasoning_suffix="",
                local_model=None,
                tokenizer=None,
                assistant_model=None,
                do_summaries="Yes",
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- Topic Summarisation Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if summary_output_files:
                print("Generated Files:", sorted(summary_output_files))

            # Write usage log
            write_usage_log(
                session_hash=session_hash,
                file_name=working_data_file_name_textbox,
                text_column=args.text_column if args.text_column else "",
                model_choice=args.model_choice,
                conversation_metadata=conversation_metadata_textbox or "",
                input_tokens=input_tokens_num or 0,
                output_tokens=output_tokens_num or 0,
                number_of_calls=number_of_calls_num or 0,
                estimated_time_taken=estimated_time_taken_number or processing_time,
                cost_code=args.cost_code,
                save_to_csv=args.save_logs_to_csv,
                save_to_dynamodb=args.save_logs_to_dynamodb,
            )

            # Create Excel output if requested
            xlsx_files = []
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file if args.input_file else [],
                        chosen_cols=[args.text_column] if args.text_column else [],
                        reference_data_file_name_textbox=working_data_file_name_textbox,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=master_reference_df_revised_summaries_state,
                        master_unique_topics_df_state=master_unique_topics_df_revised_summaries_state,
                        summarised_output_df=pd.DataFrame(),  # Summaries are in the revised dataframes
                        missing_df_state=pd.DataFrame(),
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

            # Upload outputs to S3 if enabled
            all_output_files = (
                list(summary_output_files) if summary_output_files else []
            )
            if xlsx_files:
                all_output_files.extend(xlsx_files)
            upload_outputs_to_s3_if_enabled(
                output_files=all_output_files,
                base_file_name=working_data_file_name_textbox,
                session_hash=session_hash,
            )

        # Task 5: Overall Summary
        elif args.task == "overall_summary":
            print("--- Starting Overall Summary Workflow... ---")

            # Load previous output files
            (
                reference_df,
                topic_summary_df,
                latest_batch_completed_no_loop,
                deduplication_input_files_status,
                working_data_file_name_textbox,
                unique_topics_table_file_name_textbox,
            ) = load_in_previous_data_files(args.previous_output_files)

            # Run overall summary
            (
                overall_summary_output_files,
                overall_summarised_output_markdown,
                summarised_output_df,
                conversation_metadata_textbox,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                estimated_time_taken_number,
                output_messages_textbox,
                logged_content_df,
            ) = overall_summary(
                topic_summary_df=topic_summary_df,
                model_choice=args.model_choice,
                in_api_key=args.google_api_key,
                temperature=args.temperature,
                reference_data_file_name=working_data_file_name_textbox,
                output_folder=args.output_dir,
                chosen_cols=[args.text_column] if args.text_column else list(),
                context_textbox=args.context,
                aws_access_key_textbox=args.aws_access_key,
                aws_secret_key_textbox=args.aws_secret_key,
                aws_region_textbox=args.aws_region,
                model_name_map=model_name_map,
                hf_api_key_textbox=args.hf_token,
                azure_endpoint_textbox=args.azure_endpoint,
                existing_logged_content=list(),
                api_url=args.api_url if args.api_url else API_URL,
                output_debug_files=str(args.output_debug_files),
                log_output_files=list(),
                reasoning_suffix="",
                local_model=None,
                tokenizer=None,
                assistant_model=None,
                do_summaries="Yes",
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- Overall Summary Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if overall_summary_output_files:
                print("Generated Files:", sorted(overall_summary_output_files))

            # Write usage log
            write_usage_log(
                session_hash=session_hash,
                file_name=working_data_file_name_textbox,
                text_column=args.text_column if args.text_column else "",
                model_choice=args.model_choice,
                conversation_metadata=conversation_metadata_textbox or "",
                input_tokens=input_tokens_num or 0,
                output_tokens=output_tokens_num or 0,
                number_of_calls=number_of_calls_num or 0,
                estimated_time_taken=estimated_time_taken_number or processing_time,
                cost_code=args.cost_code,
                save_to_csv=args.save_logs_to_csv,
                save_to_dynamodb=args.save_logs_to_dynamodb,
            )

            # Create Excel output if requested
            xlsx_files = []
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file if args.input_file else [],
                        chosen_cols=[args.text_column] if args.text_column else [],
                        reference_data_file_name_textbox=working_data_file_name_textbox,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=reference_df,  # Use original reference_df
                        master_unique_topics_df_state=topic_summary_df,  # Use original topic_summary_df
                        summarised_output_df=summarised_output_df,
                        missing_df_state=pd.DataFrame(),
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

            # Upload outputs to S3 if enabled
            all_output_files = (
                list(overall_summary_output_files)
                if overall_summary_output_files
                else []
            )
            if xlsx_files:
                all_output_files.extend(xlsx_files)
            upload_outputs_to_s3_if_enabled(
                output_files=all_output_files,
                base_file_name=working_data_file_name_textbox,
                session_hash=session_hash,
            )

        # Task 6: All-in-One Pipeline
        elif args.task == "all_in_one":
            print("--- Starting All-in-One Pipeline Workflow... ---")

            # Load data file
            if isinstance(args.input_file, str):
                args.input_file = [args.input_file]

            file_data, file_name, total_number_of_batches = load_in_data_file(
                file_paths=args.input_file,
                in_colnames=[args.text_column],
                batch_size=args.batch_size,
                in_excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
            )

            # Prepare candidate topics if provided
            candidate_topics = None
            if args.candidate_topics:
                candidate_topics = args.candidate_topics

            # Determine summary format prompt
            summary_format_prompt = (
                two_para_summary_format_prompt
                if args.summary_format == "two_paragraph"
                else single_para_summary_format_prompt
            )

            # Run all-in-one pipeline
            (
                display_markdown,
                master_topic_df_state,
                master_unique_topics_df_state,
                master_reference_df_state,
                topic_extraction_output_files,
                text_output_file_list_state,
                latest_batch_completed,
                log_files_output,
                log_files_output_list_state,
                conversation_metadata_textbox,
                estimated_time_taken_number,
                deduplication_input_files,
                summarisation_input_files,
                modifiable_unique_topics_df_state,
                modification_input_files,
                in_join_files,
                missing_df_state,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                output_messages_textbox,
                summary_reference_table_sample_state,
                summarised_references_markdown,
                master_unique_topics_df_revised_summaries_state,
                master_reference_df_revised_summaries_state,
                summary_output_files,
                summarised_outputs_list,
                latest_summary_completed_num,
                overall_summarisation_input_files,
                overall_summary_output_files,
                overall_summarised_output_markdown,
                summarised_output_df,
                logged_content_df,
            ) = all_in_one_pipeline(
                grouping_col=args.group_by,
                in_data_files=args.input_file,
                file_data=file_data,
                existing_topics_table=pd.DataFrame(),
                existing_reference_df=pd.DataFrame(),
                existing_topic_summary_df=pd.DataFrame(),
                unique_table_df_display_table_markdown="",
                original_file_name=file_name,
                total_number_of_batches=total_number_of_batches,
                in_api_key=args.google_api_key,
                temperature=args.temperature,
                chosen_cols=[args.text_column],
                model_choice=args.model_choice,
                candidate_topics=candidate_topics,
                first_loop_state=True,
                conversation_metadata_text="",
                latest_batch_completed=0,
                time_taken_so_far=0,
                initial_table_prompt_text=initial_table_prompt,
                initial_table_system_prompt_text=initial_table_system_prompt,
                add_existing_topics_system_prompt_text=add_existing_topics_system_prompt,
                add_existing_topics_prompt_text=add_existing_topics_prompt,
                number_of_prompts_used=1,
                batch_size=args.batch_size,
                context_text=args.context,
                sentiment_choice=args.sentiment,
                force_zero_shot_choice=args.force_zero_shot,
                in_excel_sheets=args.excel_sheets,
                force_single_topic_choice=args.force_single_topic,
                produce_structures_summary_choice=args.produce_structured_summary,
                aws_access_key_text=args.aws_access_key,
                aws_secret_key_text=args.aws_secret_key,
                aws_region_text=args.aws_region,
                hf_api_key_text=args.hf_token,
                azure_api_key_text=args.azure_api_key,
                azure_endpoint_text=args.azure_endpoint,
                output_folder=args.output_dir,
                merge_sentiment=args.merge_sentiment,
                merge_general_topics=args.merge_general_topics,
                score_threshold=args.similarity_threshold,
                summarise_format=summary_format_prompt,
                random_seed=args.random_seed,
                log_files_output_list_state=list(),
                model_name_map_state=model_name_map,
                usage_logs_location=(
                    os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                    if args.save_logs_to_csv
                    else ""
                ),
                existing_logged_content=list(),
                additional_instructions_summary_format=args.additional_summary_instructions,
                additional_validation_issues_provided="",
                show_previous_table="Yes",
                sample_reference_table_checkbox=args.sample_reference_table,
                api_url=args.api_url if args.api_url else API_URL,
                output_debug_files=str(args.output_debug_files),
                model=None,
                tokenizer=None,
                assistant_model=None,
                max_rows=999999,
            )

            end_time = time.time()
            processing_time = end_time - start_time

            print("\n--- All-in-One Pipeline Complete ---")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"\nOutput files saved to: {args.output_dir}")
            if overall_summary_output_files:
                print("Generated Files:", sorted(overall_summary_output_files))

            # Write usage log
            write_usage_log(
                session_hash=session_hash,
                file_name=file_name,
                text_column=args.text_column,
                model_choice=args.model_choice,
                conversation_metadata=conversation_metadata_textbox or "",
                input_tokens=input_tokens_num or 0,
                output_tokens=output_tokens_num or 0,
                number_of_calls=number_of_calls_num or 0,
                estimated_time_taken=estimated_time_taken_number or processing_time,
                cost_code=args.cost_code,
                save_to_csv=args.save_logs_to_csv,
                save_to_dynamodb=args.save_logs_to_dynamodb,
            )

            # Create Excel output if requested
            xlsx_files = []
            if args.create_xlsx_output:
                print("\nCreating Excel output file...")
                try:
                    xlsx_files, _ = collect_output_csvs_and_create_excel_output(
                        in_data_files=args.input_file,
                        chosen_cols=[args.text_column],
                        reference_data_file_name_textbox=file_name,
                        in_group_col=args.group_by,
                        model_choice=args.model_choice,
                        master_reference_df_state=master_reference_df_revised_summaries_state,
                        master_unique_topics_df_state=master_unique_topics_df_revised_summaries_state,
                        summarised_output_df=summarised_output_df,
                        missing_df_state=missing_df_state,
                        excel_sheets=args.excel_sheets[0] if args.excel_sheets else "",
                        usage_logs_location=(
                            os.path.join(USAGE_LOGS_FOLDER, USAGE_LOG_FILE_NAME)
                            if args.save_logs_to_csv
                            else ""
                        ),
                        model_name_map=model_name_map,
                        output_folder=args.output_dir,
                        structured_summaries=args.produce_structured_summary,
                    )
                    if xlsx_files:
                        print(f"Excel output created: {sorted(xlsx_files)}")
                except Exception as e:
                    print(f"Warning: Could not create Excel output: {e}")

            # Upload outputs to S3 if enabled
            # Collect all output files from the pipeline
            all_output_files = []
            if topic_extraction_output_files:
                all_output_files.extend(topic_extraction_output_files)
            if overall_summary_output_files:
                all_output_files.extend(overall_summary_output_files)
            if xlsx_files:
                all_output_files.extend(xlsx_files)
            upload_outputs_to_s3_if_enabled(
                output_files=all_output_files,
                base_file_name=file_name,
                session_hash=session_hash,
            )

        else:
            print(f"Error: Invalid task '{args.task}'.")
            print(
                "Valid options: 'extract', 'validate', 'deduplicate', 'summarise', 'overall_summary', or 'all_in_one'"
            )

    except Exception as e:
        print(f"\nAn error occurred during the workflow: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
