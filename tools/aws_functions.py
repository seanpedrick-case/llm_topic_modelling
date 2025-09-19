from typing import Type, List
import pandas as pd
import boto3
import os
from tools.config import RUN_AWS_FUNCTIONS, AWS_REGION, S3_LOG_BUCKET, AWS_ACCESS_KEY, AWS_SECRET_KEY, PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS

# Empty bucket name in case authentication fails
bucket_name=S3_LOG_BUCKET

def connect_to_bedrock_runtime(model_name_map:dict, model_choice:str, aws_access_key_textbox:str="", aws_secret_key_textbox:str=""):
    # If running an anthropic model, assume that running an AWS Bedrock model, load in Bedrock
    model_source = model_name_map[model_choice]["source"]

    if "AWS" in model_source:        
        if RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        elif aws_access_key_textbox and aws_secret_key_textbox:
            print("Connecting to Bedrock using AWS access key and secret keys from user input.")
            bedrock_runtime = boto3.client('bedrock-runtime', 
                aws_access_key_id=aws_access_key_textbox, 
                aws_secret_access_key=aws_secret_key_textbox, region_name=AWS_REGION)
        elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
            print("Getting Bedrock credentials from environment variables")
            bedrock_runtime = boto3.client('bedrock-runtime', 
                aws_access_key_id=AWS_ACCESS_KEY, 
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION)
        elif RUN_AWS_FUNCTIONS == "1":
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)             
        else:
            bedrock_runtime = ""
            out_message = "Cannot connect to AWS Bedrock service. Please provide access keys under LLM settings, or choose another model type."
            print(out_message)
            raise Exception(out_message)
    else: 
        bedrock_runtime = list()

    return bedrock_runtime

def connect_to_s3_client(aws_access_key_textbox:str="", aws_secret_key_textbox:str=""):
    # If running an anthropic model, assume that running an AWS s3 model, load in s3
    s3_client = list()

    if aws_access_key_textbox and aws_secret_key_textbox:
        print("Connecting to s3 using AWS access key and secret keys from user input.")
        s3_client = boto3.client('s3', 
            aws_access_key_id=aws_access_key_textbox, 
            aws_secret_access_key=aws_secret_key_textbox, region_name=AWS_REGION)
    elif RUN_AWS_FUNCTIONS == "1" and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS == "1":
        print("Connecting to s3 via existing SSO connection")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
        print("Getting s3 credentials from environment variables")
        s3_client = boto3.client('s3', 
            aws_access_key_id=AWS_ACCESS_KEY, 
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION)               
    else:
        s3_client = ""
        out_message = "Cannot connect to S3 service. Please provide access keys under LLM settings, or choose another model type."
        print(out_message)
        raise Exception(out_message)

    return s3_client

# Download direct from S3 - requires login credentials
def download_file_from_s3(bucket_name:str, key:str, local_file_path:str, aws_access_key_textbox:str="", aws_secret_key_textbox:str="", RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS):

    if RUN_AWS_FUNCTIONS=="1":

        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        #boto3.client('s3')
        s3.download_file(bucket_name, key, local_file_path)
        print(f"File downloaded from S3: s3://{bucket_name}/{key} to {local_file_path}")
                         
def download_folder_from_s3(bucket_name:str, s3_folder:str, local_folder:str, aws_access_key_textbox:str="", aws_secret_key_textbox:str="", RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS):
    """
    Download all files from an S3 folder to a local folder.
    """
    if RUN_AWS_FUNCTIONS == "1":
        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        #boto3.client('s3')

        # List objects in the specified S3 folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

        # Download each object
        for obj in response.get('Contents', []):
            # Extract object key and construct local file path
            object_key = obj['Key']
            local_file_path = os.path.join(local_folder, os.path.relpath(object_key, s3_folder))

            # Create directories if necessary
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the object
            try:
                s3.download_file(bucket_name, object_key, local_file_path)
                print(f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'")
            except Exception as e:
                print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)

def download_files_from_s3(bucket_name:str, s3_folder:str, local_folder:str, filenames:list[str], aws_access_key_textbox:str="", aws_secret_key_textbox:str="", RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS):
    """
    Download specific files from an S3 folder to a local folder.
    """
    if RUN_AWS_FUNCTIONS == "1":
        s3 = connect_to_s3_client(aws_access_key_textbox, aws_secret_key_textbox)
        #boto3.client('s3')

        print("Trying to download file: ", filenames)

        if filenames == '*':
            # List all objects in the S3 folder
            print("Trying to download all files in AWS folder: ", s3_folder)
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

            print("Found files in AWS folder: ", response.get('Contents', []))

            filenames = [obj['Key'].split('/')[-1] for obj in response.get('Contents', [])]

            print("Found filenames in AWS folder: ", filenames)

        for filename in filenames:
            object_key = os.path.join(s3_folder, filename)
            local_file_path = os.path.join(local_folder, filename)

            # Create directories if necessary
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the object
            try:
                s3.download_file(bucket_name, object_key, local_file_path)
                print(f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'")
            except Exception as e:
                print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)

def upload_file_to_s3(local_file_paths:List[str], s3_key:str, s3_bucket:str=bucket_name, aws_access_key_textbox:str="", aws_secret_key_textbox:str="", RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS):
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
        #boto3.client('s3')

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
            final_out_message_str = '\n'.join(final_out_message)

    else:
        final_out_message_str = "Not connected to AWS, no files uploaded."

    return final_out_message_str
        
    
