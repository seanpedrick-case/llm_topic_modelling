import os
from typing import List

import boto3

from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS,
    RUN_AWS_FUNCTIONS,
    S3_LOG_BUCKET,
    S3_OUTPUTS_BUCKET,
)

# Empty bucket name in case authentication fails
bucket_name = S3_LOG_BUCKET


def connect_to_bedrock_runtime(
    model_name_map: dict,
    model_choice: str,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
):
    # If running an anthropic model, assume that running an AWS Bedrock model, load in Bedrock
    model_source = model_name_map[model_choice]["source"]

    if "AWS" in model_source:
        if RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        elif aws_access_key_textbox and aws_secret_key_textbox:
            print(
                "Connecting to Bedrock using AWS access key and secret keys from user input."
            )
            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_textbox,
                aws_secret_access_key=aws_secret_key_textbox,
                region_name=AWS_REGION,
            )
        elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
            print("Getting Bedrock credentials from environment variables")
            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
            )
        elif RUN_AWS_FUNCTIONS == "1":
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        else:
            bedrock_runtime = ""
            out_message = "Cannot connect to AWS Bedrock service. Please provide access keys under LLM settings, or choose another model type."
            print(out_message)
            raise Exception(out_message)
    else:
        bedrock_runtime = None

    return bedrock_runtime


def connect_to_s3_client(
    aws_access_key_textbox: str = "", aws_secret_key_textbox: str = ""
):
    # If running an anthropic model, assume that running an AWS s3 model, load in s3
    s3_client = None

    if aws_access_key_textbox and aws_secret_key_textbox:
        print("Connecting to s3 using AWS access key and secret keys from user input.")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_textbox,
            aws_secret_access_key=aws_secret_key_textbox,
            region_name=AWS_REGION,
        )
    elif RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
        print("Connecting to s3 via existing SSO connection")
        s3_client = boto3.client("s3", region_name=AWS_REGION)
    elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
        print("Getting s3 credentials from environment variables")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
    else:
        s3_client = ""
        out_message = "Cannot connect to S3 service. Please provide access keys under LLM settings, or choose another model type."
        print(out_message)
        raise Exception(out_message)

    return s3_client


# Download direct from S3 - requires login credentials
def download_file_from_s3(
    bucket_name: str,
    key: str,
    local_file_path: str,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
):

    if RUN_AWS_FUNCTIONS == "1":

        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        # boto3.client('s3')
        s3.download_file(bucket_name, key, local_file_path)
        print(f"File downloaded from S3: s3://{bucket_name}/{key} to {local_file_path}")


def download_folder_from_s3(
    bucket_name: str,
    s3_folder: str,
    local_folder: str,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
):
    """
    Download all files from an S3 folder to a local folder.
    """
    if RUN_AWS_FUNCTIONS == "1":
        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        # boto3.client('s3')

        # List objects in the specified S3 folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

        # Download each object
        for obj in response.get("Contents", []):
            # Extract object key and construct local file path
            object_key = obj["Key"]
            local_file_path = os.path.join(
                local_folder, os.path.relpath(object_key, s3_folder)
            )

            # Create directories if necessary
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the object
            try:
                s3.download_file(bucket_name, object_key, local_file_path)
                print(
                    f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'"
                )
            except Exception as e:
                print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)


def download_files_from_s3(
    bucket_name: str,
    s3_folder: str,
    local_folder: str,
    filenames: list[str],
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
):
    """
    Download specific files from an S3 folder to a local folder.
    """
    if RUN_AWS_FUNCTIONS == "1":
        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        # boto3.client('s3')

        print("Trying to download file: ", filenames)

        if filenames == "*":
            # List all objects in the S3 folder
            print("Trying to download all files in AWS folder: ", s3_folder)
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

            print("Found files in AWS folder: ", response.get("Contents", []))

            filenames = [
                obj["Key"].split("/")[-1] for obj in response.get("Contents", [])
            ]

            print("Found filenames in AWS folder: ", filenames)

        for filename in filenames:
            object_key = os.path.join(s3_folder, filename)
            local_file_path = os.path.join(local_folder, filename)

            # Create directories if necessary
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the object
            try:
                s3.download_file(bucket_name, object_key, local_file_path)
                print(
                    f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'"
                )
            except Exception as e:
                print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)


def upload_file_to_s3(
    local_file_paths: List[str],
    s3_key: str,
    s3_bucket: str = bucket_name,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
):
    """
    Uploads a file from local machine to Amazon S3.

    Args:
    - local_file_path: Local file path(s) of the file(s) to upload.
    - s3_key: Key (path) to the file in the S3 bucket.
    - s3_bucket: Name of the S3 bucket.

    Returns:
    - Message as variable/printed to console
    """
    if RUN_AWS_FUNCTIONS == "1":

        final_out_message = list()

        s3_client = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        # boto3.client('s3')

        if isinstance(local_file_paths, str):
            local_file_paths = [local_file_paths]

        for file in local_file_paths:
            try:
                # Get file name off file path
                file_name = os.path.basename(file)

                s3_key_full = s3_key + file_name
                print("S3 key: ", s3_key_full)

                s3_client.upload_file(file, s3_bucket, s3_key_full)
                out_message = "File " + file_name + " uploaded successfully!"
                print(out_message)

            except Exception as e:
                out_message = f"Error uploading file(s): {e}"
                print(out_message)

            final_out_message.append(out_message)
            final_out_message_str = "\n".join(final_out_message)

    else:
        final_out_message_str = "Not connected to AWS, no files uploaded."

    return final_out_message_str


# Helper to upload outputs to S3 when enabled in config.
def export_outputs_to_s3(
    file_list_state,
    s3_output_folder_state_value: str,
    save_outputs_to_s3_flag: bool,
    base_file_state=None,
    s3_bucket: str = S3_OUTPUTS_BUCKET,
):
    """
    Upload a list of local output files to the configured S3 outputs folder.

    - file_list_state: Gradio dropdown state that holds a list of file paths or a
        single path/string. If blank/empty, no action is taken.
    - s3_output_folder_state_value: Final S3 key prefix (including any session hash)
        to use as the destination folder for uploads.
    - s3_bucket: Name of the S3 bucket.
    """
    try:

        # Respect the runtime toggle as well as environment configuration
        if not save_outputs_to_s3_flag:
            return

        if not s3_output_folder_state_value:
            # No configured S3 outputs folder – nothing to do
            return

        # Normalise input to a Python list of strings
        file_paths = file_list_state
        if not file_paths:
            return

        # Gradio dropdown may return a single string or a list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Filter out any non-truthy values
        file_paths = [p for p in file_paths if p]
        if not file_paths:
            return

        # Derive a base file stem (name without extension) from the original
        # file(s) being analysed, if provided. This is used to create an
        # additional subfolder layer so that outputs are grouped under the
        # analysed file name rather than under each output file name.
        base_stem = None
        if base_file_state:
            base_path = None

            # Gradio File components typically provide a list of objects with a `.name` attribute
            if isinstance(base_file_state, str):
                base_path = base_file_state
            elif isinstance(base_file_state, list) and base_file_state:
                first_item = base_file_state[0]
                base_path = getattr(first_item, "name", None) or str(first_item)
            else:
                base_path = getattr(base_file_state, "name", None) or str(
                    base_file_state
                )

            if base_path:
                base_name = os.path.basename(base_path)
                base_stem, _ = os.path.splitext(base_name)

        # Ensure base S3 prefix (session/date) ends with a trailing slash
        base_prefix = s3_output_folder_state_value
        if not base_prefix.endswith("/"):
            base_prefix = base_prefix + "/"

        # For each file, append a subfolder. If we have a derived base_stem
        # from the input being analysed, use that; otherwise, fall back to
        # the individual output file name stem. Final pattern:
        #   <session_output_folder>/<date>/<base_file_stem>/<file_name>
        # or, if base_file_stem is not available:
        #   <session_output_folder>/<date>/<output_file_stem>/<file_name>
        for file in file_paths:
            file_name = os.path.basename(file)

            if base_stem:
                folder_stem = base_stem
            else:
                folder_stem, _ = os.path.splitext(file_name)

            per_file_prefix = base_prefix + folder_stem + "/"

            out_message = upload_file_to_s3(
                local_file_paths=[file],
                s3_key=per_file_prefix,
                s3_bucket=s3_bucket,
            )

            # Log any issues to console so failures are visible in logs/stdout
            if (
                "Error uploading file" in out_message
                or "could not upload" in out_message.lower()
            ):
                print("export_outputs_to_s3 encountered issues:", out_message)

        print("Successfully uploaded outputs to S3")

    except Exception as e:
        # Do not break the app flow if S3 upload fails – just report to console
        print(f"export_outputs_to_s3 failed with error: {e}")

    # No GUI outputs to update
    return
