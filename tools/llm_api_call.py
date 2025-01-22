import os
import google.generativeai as ai
import pandas as pd
import numpy as np
import gradio as gr
import markdown
import time
import boto3
import json
import math
import string
import re
import spaces
from rapidfuzz import process, fuzz
from tqdm import tqdm
from gradio import Progress
from typing import List, Tuple
from io import StringIO

GradioFileData = gr.FileData

from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt, summarise_topic_descriptions_prompt, summarise_topic_descriptions_system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt, create_general_topics_system_prompt, create_general_topics_prompt
from tools.helper_functions import output_folder, detect_file_type, get_file_path_end, read_file, get_or_create_env_var, model_name_map, put_columns_in_df, wrap_text
from tools.chatfuncs import LlamaCPPGenerationConfig, call_llama_cpp_model, load_model, RUN_LOCAL_MODEL

# ResponseObject class for AWS Bedrock calls
class ResponseObject:
        def __init__(self, text, usage_metadata):
            self.text = text
            self.usage_metadata = usage_metadata

max_tokens = 4096
timeout_wait = 30 # AWS now seems to have a 60 second minimum wait between API calls
number_of_api_retry_attempts = 5
max_time_for_loop = 99999
batch_size_default = 5
deduplication_threshold = 90

MAX_COMMENT_CHARS = get_or_create_env_var('MAX_COMMENT_CHARS', '14000')
print(f'The value of MAX_COMMENT_CHARS is {MAX_COMMENT_CHARS}')

max_comment_character_length = int(MAX_COMMENT_CHARS)

AWS_DEFAULT_REGION = get_or_create_env_var('AWS_DEFAULT_REGION', 'eu-west-2')
print(f'The value of AWS_DEFAULT_REGION is {AWS_DEFAULT_REGION}')

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)

### HELPER FUNCTIONS

def normalise_string(text):
    # Replace two or more dashes with a single dash
    text = re.sub(r'-{2,}', '-', text)
    
    # Replace two or more spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text

def load_in_file(file_path: str, colname:str=""):
    """
    Loads in a tabular data file and returns data and file name.

    Parameters:
    - file_path (str): The path to the file to be processed.
    """
    file_type = detect_file_type(file_path)
    print("File type is:", file_type)

    file_name = get_file_path_end(file_path)
    file_data = read_file(file_path)

    if colname:
        file_data[colname] = file_data[colname].fillna("")

        file_data[colname] = file_data[colname].astype(str).str.replace("\bnan\b", "", regex=True)  
        
        #print(file_data[colname])

    return file_data, file_name

def load_in_data_file(file_paths:List[str], in_colnames:List[str], batch_size:int=50):
    '''Load in data table, work out how many batches needed.'''

    try:
        file_data, file_name = load_in_file(file_paths[0], colname=in_colnames)
        num_batches = math.ceil(len(file_data) / batch_size)
        print("Total number of batches:", num_batches)

    except Exception as e:
        print(e)
        file_data = pd.DataFrame()
        file_name = ""
        num_batches = 1  
    
    return file_data, file_name, num_batches

def load_in_previous_data_files(file_paths_partial_output:List[str]):
    '''Load in data table from a partially completed consultation summary to continue it.'''

    reference_file_data = pd.DataFrame()
    reference_file_name = ""
    unique_file_data = pd.DataFrame()
    unique_file_name = ""
    out_message = ""
    latest_batch = 0

    for file in file_paths_partial_output:
        # If reference table
        if 'reference_table' in file.name:
            try:
                reference_file_data, reference_file_name = load_in_file(file)
                #print("reference_file_data:", reference_file_data.head(2))
                out_message = out_message + " Reference file load successful"
            except Exception as e:
                out_message = "Could not load reference file data:" + str(e)
                print("Could not load reference file data:", e)
        # If unique table
        if 'unique_topics' in file.name:
            try:
                unique_file_data, unique_file_name = load_in_file(file)
                #print("unique_topics_file:", unique_file_data.head(2))
                out_message = out_message + " Unique table file load successful"
            except Exception as e:
                out_message = "Could not load unique table file data:" + str(e)
                print("Could not load unique table file data:", e)
        if 'batch_' in file.name:
            latest_batch = re.search(r'batch_(\d+)', file.name).group(1)
            print("latest batch:", latest_batch)
            latest_batch = int(latest_batch)

    if latest_batch == 0:
        out_message = out_message + " Latest batch number not found."
    if reference_file_data.empty:
        out_message = out_message + " No reference data table provided."
    if unique_file_data.empty:
        out_message = out_message + " No unique data table provided."   

    print(out_message)
            
    return reference_file_data, unique_file_data, latest_batch, out_message, reference_file_name, unique_file_name

def data_file_to_markdown_table(file_data:pd.DataFrame, file_name:str, chosen_cols: List[str], output_folder: str, batch_number: int, batch_size: int) -> Tuple[str, str, str]:
    """
    Processes a file by simplifying its content based on chosen columns and saves the result to a specified output folder.

    Parameters:
    - file_data (pd.DataFrame): Tabular data file with responses.
    - file_name (str): File name with extension.
    - chosen_cols (List[str]): A list of column names to include in the simplified file.
    - output_folder (str): The directory where the simplified file will be saved.
    - batch_number (int): The current batch number for processing.
    - batch_size (int): The number of rows to process in each batch.

    Returns:
    - Tuple[str, str, str]: A tuple containing the path to the simplified CSV file, the simplified markdown table as a string, and the file path end (used for naming the output file).
    """

    #print("\nfile_data_in_markdown func:", file_data)
    #print("\nBatch size in markdown func:", str(batch_size))
    
    normalised_simple_markdown_table = ""
    simplified_csv_table_path = ""

    # Simplify table to just responses column and the Response reference number
    simple_file = file_data[[chosen_cols]].reset_index(names="Reference")
    simple_file["Reference"] = simple_file["Reference"].astype(int) + 1
    simple_file = simple_file.rename(columns={chosen_cols: "Response"})
    simple_file["Response"] = simple_file["Response"].str.strip()
    file_len = len(simple_file["Reference"])
   

     # Subset the data for the current batch
    start_row = batch_number * batch_size
    if start_row > file_len + 1:
        print("Start row greater than file row length")
        return simplified_csv_table_path, normalised_simple_markdown_table, file_name

    if (start_row + batch_size) <= file_len + 1:
        end_row = start_row + batch_size
    else:
        end_row = file_len + 1

    simple_file = simple_file[start_row:end_row]  # Select the current batch

    # Now replace the reference numbers with numbers starting from 1
    simple_file["Reference"] = simple_file["Reference"] - start_row

    #print("simple_file:", simple_file)

    # Remove problematic characters including ASCII and various quote marks
        # Remove problematic characters including control characters, special characters, and excessive leading/trailing whitespace
    simple_file["Response"] = simple_file["Response"].str.replace(r'[\x00-\x1F\x7F]|[""<>]|\\', '', regex=True)  # Remove control and special characters
    simple_file["Response"] = simple_file["Response"].str.strip()  # Remove leading and trailing whitespace
    simple_file["Response"] = simple_file["Response"].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space
    simple_file["Response"] = simple_file["Response"].str.replace(r'\n{2,}', '\n', regex=True)  # Replace multiple line breaks with a single line break
    simple_file["Response"] = simple_file["Response"].str.slice(0, max_comment_character_length) # Maximum 1,500 character responses

    # Remove blank and extremely short responses
    simple_file = simple_file.loc[~(simple_file["Response"].isnull()) &\
                                  ~(simple_file["Response"] == "None") &\
                                  ~(simple_file["Response"] == " ") &\
                                  ~(simple_file["Response"] == ""),:]#~(simple_file["Response"].str.len() < 5), :]

    #simplified_csv_table_path = output_folder + 'simple_markdown_table_' + file_name + '_row_' + str(start_row) + '_to_' + str(end_row) + '.csv'
    #simple_file.to_csv(simplified_csv_table_path, index=None)

    simple_markdown_table = simple_file.to_markdown(index=None)

    normalised_simple_markdown_table = normalise_string(simple_markdown_table)

    return simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, simple_file

def replace_punctuation_with_underscore(input_string):
    # Create a translation table where each punctuation character maps to '_'
    translation_table = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    
    # Translate the input string using the translation table
    return input_string.translate(translation_table)

### LLM FUNCTIONS

def construct_gemini_generative_model(in_api_key: str, temperature: float, model_choice: str, system_prompt: str, max_tokens: int) -> Tuple[object, dict]:
    """
    Constructs a GenerativeModel for Gemini API calls.

    Parameters:
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model, controlling the randomness of the output.
    - model_choice (str): The choice of model to use for generation.
    - system_prompt (str): The system prompt to guide the generation.
    - max_tokens (int): The maximum number of tokens to generate.

    Returns:
    - Tuple[object, dict]: A tuple containing the constructed GenerativeModel and its configuration.
    """
    # Construct a GenerativeModel
    try:
        if in_api_key:
            #print("Getting API key from textbox")
            api_key = in_api_key
            ai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            #print("Searching for API key in environmental variables")
            api_key = os.environ["GOOGLE_API_KEY"]
            ai.configure(api_key=api_key)
        else:
            print("No API key foound")
            raise gr.Error("No API key found.")
    except Exception as e:
        print(e)
    
    config = ai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)

    #model = ai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=config)
    model = ai.GenerativeModel(model_name='models/' + model_choice, system_instruction=system_prompt, generation_config=config)
    
    # Upload CSV file (replace with your actual file path)
    #file_id = ai.upload_file(upload_file_path)

    
    # if file_type == 'xlsx':
    #     print("Running through all xlsx sheets")
    #     #anon_xlsx = pd.ExcelFile(upload_file_path)
    #     if not in_excel_sheets:
    #         out_message.append("No Excel sheets selected. Please select at least one to anonymise.")
    #         continue

    #     anon_xlsx = pd.ExcelFile(upload_file_path)                

    #     # Create xlsx file:
    #     anon_xlsx_export_file_name = output_folder + file_name + "_redacted.xlsx"


    ### QUERYING LARGE LANGUAGE MODEL ###
    # Prompt caching the table and system prompt. See here: https://ai.google.dev/gemini-api/docs/caching?lang=python
    # Create a cache with a 5 minute TTL. ONLY FOR CACHES OF AT LEAST 32k TOKENS!
    # cache = ai.caching.CachedContent.create(
    # model='models/' + model_choice,
    # display_name=file_name, # used to identify the cache
    # system_instruction=system_prompt,
    # ttl=datetime.timedelta(minutes=5),
    # )

    return model, config

def call_aws_claude(prompt: str, system_prompt: str, temperature: float, max_tokens: int, model_choice: str) -> ResponseObject:
    """
    This function sends a request to AWS Claude with the following parameters:
    - prompt: The user's input prompt to be processed by the model.
    - system_prompt: A system-defined prompt that provides context or instructions for the model.
    - temperature: A value that controls the randomness of the model's output, with higher values resulting in more diverse responses.
    - max_tokens: The maximum number of tokens (words or characters) in the model's response.
    - model_choice: The specific model to use for processing the request.
    
    The function constructs the request configuration, invokes the model, extracts the response text, and returns a ResponseObject containing the text and metadata.
    """

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "top_p": 0.999,
        "temperature":temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = model_choice
    accept = "application/json"
    contentType = "application/json"

    request = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    # Extract text from request
    response_body = json.loads(request.get("body").read())
    text = response_body.get("content")[0].get("text")

    response = ResponseObject(
    text=text,
    usage_metadata=request['ResponseMetadata']
    )

    # Now you can access both the text and metadata
    #print("Text:", response.text)
    print("Metadata:", response.usage_metadata)
    #print("Text:", response.text)
    
    return response

# Function to send a request and update history
def send_request(prompt: str, conversation_history: List[dict], model: object, config: dict, model_choice: str, system_prompt: str, temperature: float, local_model=[], progress=Progress(track_tqdm=True)) -> Tuple[str, List[dict]]:
    """
    This function sends a request to a language model with the given prompt, conversation history, model configuration, model choice, system prompt, and temperature.
    It constructs the full prompt by appending the new user prompt to the conversation history, generates a response from the model, and updates the conversation history with the new prompt and response.
    If the model choice is specific to AWS Claude, it calls the `call_aws_claude` function; otherwise, it uses the `model.generate_content` method.
    The function returns the response text and the updated conversation history.
    """
    # Constructing the full prompt from the conversation history
    full_prompt = "Conversation history:\n"
    
    for entry in conversation_history:
        role = entry['role'].capitalize()  # Assuming the history is stored with 'role' and 'parts'
        message = ' '.join(entry['parts'])  # Combining all parts of the message
        full_prompt += f"{role}: {message}\n"
    
    # Adding the new user prompt
    full_prompt += f"\nUser: {prompt}"

    # Clear any existing progress bars
    tqdm._instances.clear()

    # Print the full prompt for debugging purposes
    #print("full_prompt:", full_prompt)

    #progress_bar = tqdm(range(0,number_of_api_retry_attempts), desc="Calling API with " + str(timeout_wait) + " seconds per retry.", unit="attempts")

    progress_bar = range(0,number_of_api_retry_attempts)

    # Generate the model's response
    if model_choice in ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]:

        for i in progress_bar:
            try:
                print("Calling Gemini model, attempt", i + 1)
                #print("full_prompt:", full_prompt)
                #print("generation_config:", config)

                response = model.generate_content(contents=full_prompt, generation_config=config)

                #progress_bar.close()
                #tqdm._instances.clear()

                print("Successful call to Gemini model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to Gemini model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")               

                time.sleep(timeout_wait)
            
            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history
    elif model_choice in ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]:
        for i in progress_bar:
            try:
                print("Calling AWS Claude model, attempt", i + 1)
                response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)

                #progress_bar.close()
                #tqdm._instances.clear()

                print("Successful call to Claude model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to Claude model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")         

                time.sleep(timeout_wait)
                #response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)

            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history
    else:
        # This is the Gemma model
        for i in progress_bar:
            try:
                print("Calling Gemma 2B Instruct model, attempt", i + 1)

                gen_config = LlamaCPPGenerationConfig()
                gen_config.update_temp(temperature)

                response = call_llama_cpp_model(prompt, gen_config, model=local_model)

                #progress_bar.close()
                #tqdm._instances.clear()

                print("Successful call to Gemma model.")
                print("Response:", response)
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to Gemma model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")         

                time.sleep(timeout_wait)
                #response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)

            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history       

    # Update the conversation history with the new prompt and response
    conversation_history.append({'role': 'user', 'parts': [prompt]})

# output_str = output['choices'][0]['text']

    # Check if is a LLama.cpp model response
        # Check if the response is a ResponseObject
    if isinstance(response, ResponseObject):
        conversation_history.append({'role': 'assistant', 'parts': [response.text]})
    elif 'choices' in response:
        conversation_history.append({'role': 'assistant', 'parts': [response['choices'][0]['text']]})
    else:
        conversation_history.append({'role': 'assistant', 'parts': [response.text]})
    
    # Print the updated conversation history
    #print("conversation_history:", conversation_history)
    
    return response, conversation_history

def process_requests(prompts: List[str], system_prompt: str, conversation_history: List[dict], whole_conversation: List[str], whole_conversation_metadata: List[str], model: object, config: dict, model_choice: str, temperature: float, batch_no:int = 1, local_model = [], master:bool = False) -> Tuple[List[ResponseObject], List[dict], List[str], List[str]]:
    """
    Processes a list of prompts by sending them to the model, appending the responses to the conversation history, and updating the whole conversation and metadata.

    Args:
        prompts (List[str]): A list of prompts to be processed.
        system_prompt (str): The system prompt.
        conversation_history (List[dict]): The history of the conversation.
        whole_conversation (List[str]): The complete conversation including prompts and responses.
        whole_conversation_metadata (List[str]): Metadata about the whole conversation.
        model (object): The model to use for processing the prompts.
        config (dict): Configuration for the model.
        model_choice (str): The choice of model to use.        
        temperature (float): The temperature parameter for the model.
        batch_no (int): Batch number of the large language model request.
        local_model: Local gguf model (if loaded)
        master (bool): Is this request for the master table.

    Returns:
        Tuple[List[ResponseObject], List[dict], List[str], List[str]]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, and the updated whole conversation metadata.
    """
    responses = []

    # Clear any existing progress bars
    tqdm._instances.clear()

    for prompt in prompts:

        #print("prompt to LLM:", prompt)

        response, conversation_history = send_request(prompt, conversation_history, model=model, config=config, model_choice=model_choice, system_prompt=system_prompt, temperature=temperature, local_model=local_model)

        if isinstance(response, ResponseObject):
            response_text = response.text
        elif 'choices' in response:
            response_text = response['choices'][0]['text']
        else:
            response_text = response.text

        responses.append(response)
        whole_conversation.append(prompt)
        whole_conversation.append(response_text)

        # Create conversation metadata
        if master == False:
            whole_conversation_metadata.append(f"Query batch {batch_no} prompt {len(responses)} metadata:")
        else:
            whole_conversation_metadata.append(f"Query summary metadata:")

        if not isinstance(response, str):
            try:
                print("model_choice:", model_choice)
                if "claude" in model_choice:
                    print("Appending selected metadata items to metadata")
                    whole_conversation_metadata.append('x-amzn-bedrock-output-token-count:')
                    whole_conversation_metadata.append(str(response.usage_metadata['HTTPHeaders']['x-amzn-bedrock-output-token-count']))
                    whole_conversation_metadata.append('x-amzn-bedrock-input-token-count:')
                    whole_conversation_metadata.append(str(response.usage_metadata['HTTPHeaders']['x-amzn-bedrock-input-token-count']))
                elif "gemini" in model_choice:
                    whole_conversation_metadata.append(str(response.usage_metadata))
                else:
                    whole_conversation_metadata.append(str(response['usage']))
            except KeyError as e:
                print(f"Key error: {e} - Check the structure of response.usage_metadata")
        else:
            print("Response is a string object.")
            whole_conversation_metadata.append("Length prompt: " + str(len(prompt)) + ". Length response: " + str(len(response)))


    return responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text

### INITIAL TOPIC MODEL DEVELOPMENT FUNCTIONS

def clean_markdown_table(text: str):
    lines = text.splitlines()

    # Remove any empty rows or rows with only pipes
    cleaned_lines = [line for line in lines if not re.match(r'^\s*\|?\s*\|?\s*$', line)]

    # Merge lines that belong to the same row (i.e., don't start with |)
    merged_lines = []
    buffer = ""
    
    for line in cleaned_lines:
        if line.lstrip().startswith('|'):  # If line starts with |, it's a new row
            if buffer:
                merged_lines.append(buffer)  # Append the buffered content
            buffer = line  # Start a new buffer with this row
        else:
            # Continuation of the previous row
            buffer += ' ' + line.strip()  # Add content to the current buffer

    # Don't forget to append the last buffer
    if buffer:
        merged_lines.append(buffer)

    # Fix the header separator row if necessary
    if len(merged_lines) > 1:
        header_pipes = merged_lines[0].count('|')  # Count pipes in the header row
        header_separator = '|---|' * (header_pipes - 1) + '|---|'  # Generate proper separator
        
        # Replace or insert the separator row
        if not re.match(r'^\|[-:|]+$', merged_lines[1]):  # Check if the second row is a valid separator
            merged_lines.insert(1, header_separator)
        else:
            # Adjust the separator to match the header pipes
            merged_lines[1] = '|---|' * (header_pipes - 1) + '|'

    # Ensure consistent number of pipes in each row
    result = []
    header_pipes = merged_lines[0].count('|')  # Use the header row to count the number of pipes

    for line in merged_lines:
        # Strip excessive whitespace around pipes
        line = re.sub(r'\s*\|\s*', '|', line.strip())

        # Fix inconsistent number of pipes by adjusting them to match the header
        pipe_count = line.count('|')
        if pipe_count < header_pipes:
            line += '|' * (header_pipes - pipe_count)  # Add missing pipes
        elif pipe_count > header_pipes:
            # If too many pipes, split line and keep the first `header_pipes` columns
            columns = line.split('|')[:header_pipes + 1]  # +1 to keep last pipe at the end
            line = '|'.join(columns)
        
        line = re.sub(r'(\d),(?=\d)', r'\1, ', line)

        result.append(line)

    # Join lines back into the cleaned markdown text
    cleaned_text = '\n'.join(result)

    # Replace numbers next to commas and other numbers with a space
    

    return cleaned_text

def clean_column_name(column_name, max_length=20):
    # Convert to string
    column_name = str(column_name)  
    # Replace non-alphanumeric characters (except underscores) with underscores
    column_name = re.sub(r'\W+', '_', column_name)  
    # Remove leading/trailing underscores
    column_name = column_name.strip('_')  
    # Ensure the result is not empty; fall back to "column" if necessary
    column_name = column_name if column_name else "column"
    # Truncate to max_length
    return column_name[:max_length]

def create_unique_table_df_from_reference_table(reference_df:pd.DataFrame):
    new_unique_topics_df = reference_df[["General Topic", "Subtopic", "Sentiment"]]

    new_unique_topics_df = new_unique_topics_df.rename(columns={new_unique_topics_df.columns[0]: "General Topic", new_unique_topics_df.columns[1]: "Subtopic", new_unique_topics_df.columns[2]: "Sentiment"})
    
    # Join existing and new unique topics
    out_unique_topics_df = new_unique_topics_df

    out_unique_topics_df = out_unique_topics_df.rename(columns={out_unique_topics_df.columns[0]: "General Topic", out_unique_topics_df.columns[1]: "Subtopic", out_unique_topics_df.columns[2]: "Sentiment"})

    #print("out_unique_topics_df:", out_unique_topics_df)

    out_unique_topics_df = out_unique_topics_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment"]).\
            drop(["Response References", "Summary"], axis = 1, errors="ignore") 

    # Get count of rows that refer to particular topics
    reference_counts = reference_df.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': lambda x: '<br>'.join(
        sorted(set(x), key=lambda summary: reference_df.loc[reference_df['Summary'] == summary, 'Start row of group'].min())
    )
    }).reset_index()

    # Join the counts to existing_unique_topics_df
    out_unique_topics_df = out_unique_topics_df.merge(reference_counts, how='left', on=["General Topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    return out_unique_topics_df

# Convert output table to markdown and then to a pandas dataframe to csv
def remove_before_last_term(input_string: str) -> str:
    # Use regex to find the last occurrence of the term
    match = re.search(r'(\| ?General Topic)', input_string)
    if match:
        # Find the last occurrence by using rfind
        last_index = input_string.rfind(match.group(0))
        return input_string[last_index:]  # Return everything from the last match onward
    return input_string  # Return the original string if the term is not found

def convert_response_text_to_markdown_table(response_text:str, table_type:str = "Main table"):
    is_error = False
    start_of_table_response = remove_before_last_term(response_text)
    cleaned_response = clean_markdown_table(start_of_table_response)
    
    markdown_table = markdown.markdown(cleaned_response, extensions=['tables'])

    # Remove <p> tags and make sure it has a valid HTML structure
    html_table = re.sub(r'<p>(.*?)</p>', r'\1', markdown_table)
    html_table = html_table.replace('<p>', '').replace('</p>', '').strip()

    # Now ensure that the HTML structure is correct
    if table_type == "Main table":
        if "<table>" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>General Topic</th>
                    <th>Subtopic</th>
                    <th>Sentiment</th>                
                    <th>Response References</th>
                    <th>Summary</th>
                </tr>
                {html_table}
            </table>
            """
    elif table_type == "Revised topics table":
        if "<table>" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>General Topic</th>
                    <th>Subtopic</th>
                </tr>
                {html_table}
            </table>
            """

    html_buffer = StringIO(html_table)    

    try:
        out_df = pd.read_html(html_buffer)[0]  # Assuming the first table in the HTML is the one you want
    except Exception as e:
        print("Error when trying to parse table:", e)
        is_error = True
        raise ValueError()
        return pd.DataFrame(), is_error
    
    return out_df, is_error


def write_llm_output_and_logs(responses: List[ResponseObject],
                              whole_conversation: List[str],
                              whole_conversation_metadata: List[str],
                              file_name: str,
                              latest_batch_completed: int,
                              start_row:int,
                              end_row:int,
                              model_choice_clean: str,
                              temperature: float,
                              log_files_output_paths: List[str],
                              existing_reference_df:pd.DataFrame,
                              existing_topics_df:pd.DataFrame,
                              batch_size_number:int,
                              in_column:str,
                              first_run: bool = False) -> None:
    """
    Writes the output of the large language model requests and logs to files.

    Parameters:
    - responses (List[ResponseObject]): A list of ResponseObject instances containing the text and usage metadata of the responses.
    - whole_conversation (List[str]): A list of strings representing the complete conversation including prompts and responses.
    - whole_conversation_metadata (List[str]): A list of strings representing metadata about the whole conversation.
    - file_name (str): The base part of the output file name.
    - latest_batch_completed (int): The index of the current batch.
    - start_row (int): Start row of the current batch.
    - end_row (int): End row of the current batch.
    - model_choice_clean (str): The cleaned model choice string.
    - temperature (float): The temperature parameter used in the model.
    - log_files_output_paths (List[str]): A list of paths to the log files.
    - existing_reference_df (pd.DataFrame): The existing reference dataframe mapping response numbers to topics.
    - existing_topics_df (pd.DataFrame): The existing unique topics dataframe 
    - first_run (bool): A boolean indicating if this is the first run through this function in this process. Defaults to False.
    """
    unique_topics_df_out_path = []
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    unique_topics_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame()
    markdown_table = ""
    out_reference_df = pd.DataFrame()
    out_unique_topics_df = pd.DataFrame()
    batch_file_path_details = "error"

    # If there was an error in parsing, return boolean saying error
    is_error = False

    # Convert conversation to string and add to log outputs
    whole_conversation_str = '\n'.join(whole_conversation)
    whole_conversation_metadata_str = '\n'.join(whole_conversation_metadata)

    start_row_reported = start_row + 1

    # Example usage
    in_column_cleaned = clean_column_name(in_column, max_length=20)

    # Need to reduce output file names as full length files may be too long
    file_name = clean_column_name(file_name, max_length=30)    

    # Save outputs for each batch. If master file created, label file as master
    batch_file_path_details = f"{file_name}_batch_{latest_batch_completed + 1}_size_{batch_size_number}_col_{in_column_cleaned}"
    row_number_string_start = f"Rows {start_row_reported} to {end_row}: "

    print("batch_file_path_details:", batch_file_path_details)

    whole_conversation_path = output_folder + batch_file_path_details + "_full_conversation_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"
    whole_conversation_path_meta = output_folder + batch_file_path_details + "_metadata_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

    #with open(whole_conversation_path, "w", encoding='utf-8', errors='replace') as f:
    #    f.write(whole_conversation_str)

    with open(whole_conversation_path_meta, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_metadata_str)

    #log_files_output_paths.append(whole_conversation_path)
    log_files_output_paths.append(whole_conversation_path_meta)
    
    if isinstance(responses[-1], ResponseObject): response_text =  responses[-1].text
    elif "choices" in responses[-1]: response_text =  responses[-1]["choices"][0]['text']
    else: response_text =  responses[-1].text

    # Convert response text to a markdown table
    try:
        topic_with_response_df, is_error = convert_response_text_to_markdown_table(response_text)
    except Exception as e:
        print("Error in parsing markdown table from response text:", e)
        return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_file_path_details, is_error

    # Rename columns to ensure consistent use of data frames later in code
    topic_with_response_df.columns = ["General Topic", "Subtopic", "Sentiment", "Response References", "Summary"]

    # Fill in NA rows with values from above (topics seem to be included only on one row):
    topic_with_response_df = topic_with_response_df.ffill()

    #print("topic_with_response_df:", topic_with_response_df)

    # For instances where you end up with float values in Response references
    topic_with_response_df["Response References"] = topic_with_response_df["Response References"].astype(str).str.replace(".0", "", regex=False)

    # Strip and lower case topic names to remove issues where model is randomly capitalising topics/sentiment
    topic_with_response_df["General Topic"] = topic_with_response_df["General Topic"].str.strip().str.lower().str.capitalize()
    topic_with_response_df["Subtopic"] = topic_with_response_df["Subtopic"].str.strip().str.lower().str.capitalize()
    topic_with_response_df["Sentiment"] = topic_with_response_df["Sentiment"].str.strip().str.lower().str.capitalize()
    
    topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

    # Table to map references to topics
    reference_data = []

    # Iterate through each row in the original DataFrame
    for index, row in topic_with_response_df.iterrows():
        #references = re.split(r',\s*|\s+', str(row.iloc[4])) if pd.notna(row.iloc[4]) else ""
        references = re.findall(r'\d+', str(row.iloc[3])) if pd.notna(row.iloc[3]) else []
        # If no numbers found in the Response References column, check the Summary column in case reference numbers were put there by mistake
        if not references:
            references = re.findall(r'\d+', str(row.iloc[4])) if pd.notna(row.iloc[4]) else []
        topic = row.iloc[0] if pd.notna(row.iloc[0]) else ""
        subtopic = row.iloc[1] if pd.notna(row.iloc[1]) else ""
        sentiment = row.iloc[2] if pd.notna(row.iloc[2]) else ""
        summary = row.iloc[4] if pd.notna(row.iloc[4]) else ""
        # If the reference response column is very long, and there's nothing in the summary column, assume that the summary was put in the reference column
        if not summary and len(row.iloc[3] > 30):
            summary = row.iloc[3]

        summary = row_number_string_start + summary

        # Create a new entry for each reference number
        for ref in references:
            # Add start_row back onto reference_number
            try:
                response_ref_no =  str(int(ref) + int(start_row))
            except ValueError:
                print("Reference is not a number")
                continue

            reference_data.append({
                'Response References': response_ref_no,
                'General Topic': topic,
                'Subtopic': subtopic,
                'Sentiment': sentiment,
                'Summary': summary,
                "Start row of group": start_row_reported
            })

    # Create a new DataFrame from the reference data
    new_reference_df = pd.DataFrame(reference_data)

    print("new_reference_df:", new_reference_df)
    
    # Append on old reference data
    out_reference_df = pd.concat([new_reference_df, existing_reference_df]).dropna(how='all')

    # Remove duplicate Response references for the same topic
    out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    out_reference_df.sort_values(["Start row of group", "Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True) 

    # Save the new DataFrame to CSV
    reference_table_out_path = output_folder + batch_file_path_details + "_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"    

    # Table of all unique topics with descriptions
    #print("topic_with_response_df:", topic_with_response_df)
    new_unique_topics_df = topic_with_response_df[["General Topic", "Subtopic", "Sentiment"]]

    new_unique_topics_df = new_unique_topics_df.rename(columns={new_unique_topics_df.columns[0]: "General Topic", new_unique_topics_df.columns[1]: "Subtopic", new_unique_topics_df.columns[2]: "Sentiment"})
    
    # Join existing and new unique topics
    out_unique_topics_df = pd.concat([new_unique_topics_df, existing_topics_df]).dropna(how='all')

    out_unique_topics_df = out_unique_topics_df.rename(columns={out_unique_topics_df.columns[0]: "General Topic", out_unique_topics_df.columns[1]: "Subtopic", out_unique_topics_df.columns[2]: "Sentiment"})

    #print("out_unique_topics_df:", out_unique_topics_df)

    out_unique_topics_df = out_unique_topics_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment"]).\
            drop(["Response References", "Summary"], axis = 1, errors="ignore") 

    # Get count of rows that refer to particular topics
    reference_counts = out_reference_df.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': lambda x: '<br>'.join(
        sorted(set(x), key=lambda summary: out_reference_df.loc[out_reference_df['Summary'] == summary, 'Start row of group'].min())
    )
    }).reset_index()

    # Join the counts to existing_unique_topics_df
    out_unique_topics_df = out_unique_topics_df.merge(reference_counts, how='left', on=["General Topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    unique_topics_df_out_path = output_folder + batch_file_path_details + "_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

    return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_file_path_details, is_error

@spaces.GPU
def extract_topics(in_data_file,
              file_data:pd.DataFrame,
              existing_topics_table:pd.DataFrame,
              existing_reference_df:pd.DataFrame,
              existing_unique_topics_df:pd.DataFrame,
              display_table:str,
              file_name:str,
              num_batches:int,
              in_api_key:str,
              temperature:float,
              chosen_cols:List[str],
              model_choice:str,
              candidate_topics: GradioFileData = None,
              latest_batch_completed:int=0,
              out_message:List=[],
              out_file_paths:List = [],
              log_files_output_paths:List = [],
              first_loop_state:bool=False,
              whole_conversation_metadata_str:str="",
              initial_table_prompt:str=initial_table_prompt,
              prompt2:str=prompt2,
              prompt3:str=prompt3,
              system_prompt:str=system_prompt,
              add_existing_topics_system_prompt:str=add_existing_topics_system_prompt,
              add_existing_topics_prompt:str=add_existing_topics_prompt,
              number_of_prompts_used:int=1,
              batch_size:int=50,
              context_textbox:str="",
              time_taken:float = 0,
              max_tokens:int=max_tokens,
              model_name_map:dict=model_name_map,              
              max_time_for_loop:int=max_time_for_loop,              
              progress=Progress(track_tqdm=True)):

    '''
    Query an LLM (local, (Gemma 2B Instruct, Gemini or Anthropic-based on AWS) with up to three prompts about a table of open text data. Up to 'batch_size' rows will be queried at a time.

    Parameters:
    - in_data_file (gr.File): Gradio file object containing input data
    - file_data (pd.DataFrame): Pandas dataframe containing the consultation response data.
    - existing_topics_table (pd.DataFrame): Pandas dataframe containing the latest master topic table that has been iterated through batches.
    - existing_reference_df (pd.DataFrame): Pandas dataframe containing the list of Response reference numbers alongside the derived topics and subtopics.
    - existing_unique_topics_df (pd.DataFrame): Pandas dataframe containing the unique list of topics, subtopics, sentiment and summaries until this point.
    - display_table (str): Table for display in markdown format.
    - file_name (str): File name of the data file.
    - num_batches (int): Number of batches required to go through all the response rows.
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model.
    - chosen_cols (List[str]): A list of chosen columns to process.
    - candidate_topics (gr.FileData): A Gradio FileData object of existing candidate topics submitted by the user.
    - model_choice (str): The choice of model to use.
    - latest_batch_completed (int): The index of the latest file completed.
    - out_message (list): A list to store output messages.
    - out_file_paths (list): A list to store output file paths.
    - log_files_output_paths (list): A list to store log file output paths.
    - first_loop_state (bool): A flag indicating the first loop state.
    - whole_conversation_metadata_str (str): A string to store whole conversation metadata.
    - initial_table_prompt (str): The first prompt for the model.
    - prompt2 (str): The second prompt for the model.
    - prompt3 (str): The third prompt for the model.
    - system_prompt (str): The system prompt for the model.
    - add_existing_topics_system_prompt (str): The system prompt for the summary part of the model.
    - add_existing_topics_prompt (str): The prompt for the model summary.
    - number of requests (int): The number of prompts to send to the model.
    - batch_size (int): The number of data rows to consider in each request.
    - context_textbox (str, optional): A string giving some context to the consultation/task.
    - time_taken (float, optional): The amount of time taken to process the responses up until this point.
    - max_tokens (int): The maximum number of tokens for the model.
    - model_name_map (dict, optional): A dictionary mapping full model name to shortened.
    - max_time_for_loop (int, optional): The number of seconds maximum that the function should run for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there).
    - progress (Progress): A progress tracker.
    '''

    tic = time.perf_counter()
    model = ""
    config = ""
    final_time = 0.0
    whole_conversation_metadata = []
    is_error = False
    create_revised_general_topics = False
    local_model = []
    tokenizer = []
    zero_shot_topics_df = pd.DataFrame()
    #llama_system_prefix = "<|start_header_id|>system<|end_header_id|>\n" #"<start_of_turn>user\n"
    #llama_system_suffix = "<|eot_id|>" #"<end_of_turn>\n<start_of_turn>model\n"
    #llama_cpp_prefix = "<|start_header_id|>system<|end_header_id|>\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n" #"<start_of_turn>user\n"
    #llama_cpp_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n" #"<end_of_turn>\n<start_of_turn>model\n"
    #llama_cpp_prefix = "<|user|>\n" # This is for phi 3.5
    #llama_cpp_suffix = "<|end|>\n<|assistant|>" # This is for phi 3.5
    llama_cpp_prefix = "<start_of_turn>user\n"
    llama_cpp_suffix = "<end_of_turn>\n<start_of_turn>model\n"

    # Reset output files on each run:
    # out_file_paths = []

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            print("in_data_file:", in_data_file)
            in_colnames_drop, in_excel_sheets, file_name = put_columns_in_df(in_data_file)
            print("in_colnames:", in_colnames_drop)
            file_data, file_name, num_batches = load_in_data_file(in_data_file, chosen_cols, batch_size_default)
            print("file_data loaded in:", file_data)
        except:
            # Check if files and text exist
            out_message = "Please enter a data file to summarise."
            print(out_message)
            return out_message, existing_topics_table, existing_unique_topics_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths#, out_message


    #model_choice_clean = replace_punctuation_with_underscore(model_choice)
    model_choice_clean = model_name_map[model_choice]    

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("This is the first time through the loop")
        if (latest_batch_completed == 999) | (latest_batch_completed == 0):
            latest_batch_completed = 0
            out_message = []
            out_file_paths = []
            print("model_choice_clean:", model_choice_clean)

            if (model_choice == "gemma_2b_it_local") & (RUN_LOCAL_MODEL == "1"):
                progress(0.1, "Loading in Gemma 2b model")
                local_model, tokenizer = load_model()
                print("Local model loaded:", local_model)

    print("latest_batch_completed at start of function:", str(latest_batch_completed))
    print("total number of batches:", str(num_batches))

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_batch_completed >= num_batches:
        print("Last batch reached, returning batch:", str(latest_batch_completed))
        # Set to a very high number so as not to mess with subsequent file processing by the user
        #latest_batch_completed = 999

        toc = time.perf_counter()
        final_time = (toc - tic) + time_taken
        out_time = f"Everything finished in {final_time} seconds."
        print(out_time)

        print("All summaries completed. Creating outputs.")

        model_choice_clean = model_name_map[model_choice]   
        # Example usage
        in_column_cleaned = clean_column_name(chosen_cols, max_length=20)

        # Need to reduce output file names as full length files may be too long
        file_name = clean_column_name(file_name, max_length=30)    

        # Save outputs for each batch. If master file created, label file as master
        file_path_details = f"{file_name}_col_{in_column_cleaned}"

        # Save the new DataFrame to CSV
        #topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        reference_table_out_path = output_folder + file_path_details + "_final_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv" 
        unique_topics_df_out_path = output_folder +file_path_details + "_final_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

        # Write outputs to csv
        ## Topics with references
        #new_topic_df.to_csv(topic_table_out_path, index=None)
        #log_files_output_paths.append(topic_table_out_path)

        ## Reference table mapping response numbers to topics
        existing_reference_df.to_csv(reference_table_out_path, index=None)
        out_file_paths.append(reference_table_out_path)

        ## Unique topic list
        existing_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
        out_file_paths.append(unique_topics_df_out_path)

        ## Create a dataframe for missing response references:
        # Assuming existing_reference_df and file_data are already defined

        # Simplify table to just responses column and the Response reference number
        simple_file = file_data[[chosen_cols]].reset_index(names="Reference")
        simple_file["Reference"] = simple_file["Reference"].astype(int) + 1
        simple_file = simple_file.rename(columns={chosen_cols: "Response"})
        simple_file["Response"] = simple_file["Response"].str.strip()

        # Step 1: Identify missing references
        #print("simple_file:", simple_file)

        missing_references = simple_file[~simple_file['Reference'].astype(str).isin(existing_reference_df['Response References'].astype(str).unique())]

        # Step 2: Create a new DataFrame with the same columns as existing_reference_df
        missing_df = pd.DataFrame(columns=existing_reference_df.columns)

        # Step 3: Populate the new DataFrame
        missing_df['Response References'] = missing_references['Reference']
        missing_df = missing_df.fillna(np.nan)  # Fill other columns with NA

        # Display the new DataFrame
        #print("missing_df:", missing_df)

        missing_df_out_path = output_folder + file_path_details + "_missing_references_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        missing_df.to_csv(missing_df_out_path, index=None)
        log_files_output_paths.append(missing_df_out_path)

        out_file_paths = list(set(out_file_paths))
        log_files_output_paths = list(set(log_files_output_paths))        

        summary_out_file_paths = [file_path for file_path in out_file_paths if "final_" in file_path]
        print("summary_out_file_paths:", summary_out_file_paths)

        #final_out_message = '\n'.join(out_message)
        return display_table, existing_topics_table, existing_unique_topics_df, existing_reference_df, summary_out_file_paths, summary_out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths
       
    
    if num_batches > 0:
        progress_measure = round(latest_batch_completed / num_batches, 1)
        progress(progress_measure, desc="Querying large language model")
    else:
        progress(0.1, desc="Querying large language model")

    # Load file
    # If out message or out_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not out_file_paths:
        out_file_paths = []
   
    
    if model_choice == "anthropic.claude-3-sonnet-20240229-v1:0" and file_data.shape[1] > 300:
        out_message = "Your data has more than 300 rows, using the Sonnet model will be too expensive. Please choose the Haiku model instead."
        print(out_message)
        return out_message, existing_topics_table, existing_unique_topics_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths#, out_message
        
    topics_loop_description = "Extracting topics from response batches (each batch of " + str(batch_size) + " responses)."
    topics_loop = tqdm(range(latest_batch_completed, num_batches), desc = topics_loop_description, unit="batches remaining")

    
    for i in topics_loop:       
        #for latest_batch_completed in range(num_batches):
        reported_batch_no = latest_batch_completed + 1  
        print("Running query batch", str(reported_batch_no))

        # Call the function to prepare the input table
        simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, simple_table_df = data_file_to_markdown_table(file_data, file_name, chosen_cols, output_folder, latest_batch_completed, batch_size)
        log_files_output_paths.append(simplified_csv_table_path)


        # Conversation history
        conversation_history = []

        print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

        # If the latest batch of responses contains at least one instance of text
        if not simple_table_df.empty:

            print("latest_batch_completed:", latest_batch_completed)

            print("candidate_topics:", candidate_topics)

            # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
            if latest_batch_completed >= 1 or candidate_topics is not None:

                #print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

                # Prepare Gemini models before query       
                if model_choice in ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
                    print("Using Gemini model:", model_choice)
                    model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=add_existing_topics_system_prompt, max_tokens=max_tokens)
                elif model_choice in ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]:
                    print("Using AWS Bedrock model:", model_choice)
                else:
                    print("Using local model:", model_choice)

                # Preparing candidate topics
                if candidate_topics and existing_unique_topics_df.empty:
                    progress(0.1, "Creating revised zero shot topics table")
                    # 'Zero shot topics' are those supplied by the user
                    max_topic_no = 120

                    zero_shot_topics = read_file(candidate_topics.name)

                    if zero_shot_topics.shape[1] == 1:  # Check if there is only one column
                        zero_shot_topics_series = zero_shot_topics.iloc[:, 0].str.strip().str.lower().str.capitalize()
                        # Max 120 topics allowed
                        if len(zero_shot_topics_series) > max_topic_no:
                            print("Maximum", max_topic_no, "topics allowed to fit within large language model context limits.")
                            zero_shot_topics_series = zero_shot_topics_series.iloc[:max_topic_no]

                        zero_shot_topics_list = list(zero_shot_topics_series)

                        print("Zero shot topics are:", zero_shot_topics_list)

                        if create_revised_general_topics == True:
                            # Create the most up to date list of topics and subtopics.
                            # If there are candidate topics, but the existing_unique_topics_df hasn't yet been constructed, then create.
                            unique_topics_df = pd.DataFrame(data={"General Topic":[""] * len(zero_shot_topics_list), "Subtopic":zero_shot_topics_list})
                            unique_topics_markdown = unique_topics_df.to_markdown()

                            print("unique_topics_markdown:", unique_topics_markdown)
                            
                            formatted_general_topics_system_prompt = create_general_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                            # Format the general_topics prompt with the topics
                            formatted_general_topics_prompt = create_general_topics_prompt.format(topics=unique_topics_markdown)

                            if model_choice == "gemma_2b_it_local":
                                formatted_general_topics_prompt = llama_cpp_prefix + formatted_general_topics_system_prompt + "\n" + formatted_general_topics_prompt + llama_cpp_suffix

                            formatted_general_topics_prompt_list = [formatted_general_topics_prompt]

                            whole_conversation = []

                            general_topic_response, general_topic_conversation_history, general_topic_conversation, general_topic_conversation_metadata, response_text = process_requests(formatted_general_topics_prompt_list, formatted_general_topics_system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, master = True)

                            # Convert response text to a markdown table
                            try:
                                zero_shot_topics_df, is_error = convert_response_text_to_markdown_table(response_text, table_type = "Revised topics table")
                                print("Output revised zero shot topics table is:", zero_shot_topics_df)

                                zero_shot_revised_path = output_folder + "zero_shot_topics_with_general_topics.csv"
                                zero_shot_topics_df.to_csv(zero_shot_revised_path, index = None)
                                out_file_paths.append(zero_shot_revised_path)
                            except Exception as e:
                                print("Error in parsing markdown table from response text:", e)
                                print("Not adding revised General Topics to table")
                                zero_shot_topics_df = pd.DataFrame(data={"General Topic":[""] * len(zero_shot_topics_list), "Subtopic":zero_shot_topics_list})

                            if zero_shot_topics_df.empty:
                                print("Creation of revised general topics df failed, reverting to original list")
                                zero_shot_topics_df = pd.DataFrame(data={"General Topic":[""] * len(zero_shot_topics_list), "Subtopic":zero_shot_topics_list})
                        else:
                            zero_shot_topics_df = pd.DataFrame(data={"General Topic":[""] * len(zero_shot_topics_list), "Subtopic":zero_shot_topics_list})

                        # This part concatenates all zero shot and new topics together, so that for the next prompt the LLM will have the full list available
                        if not existing_unique_topics_df.empty:
                            existing_unique_topics_df = pd.concat([existing_unique_topics_df, zero_shot_topics_df]).drop_duplicates("Subtopic")
                        else:
                            existing_unique_topics_df = zero_shot_topics_df

                    # If your zero shot column file already contains General Topic and Subtopic columns
                    if set(["General Topic", "Subtopic"]).issubset(zero_shot_topics.columns):
                        # Max 120 topics allowed
                        if zero_shot_topics.shape[0] > max_topic_no:
                            print("Maximum", max_topic_no, "topics allowed to fit within large language model context limits.")
                            zero_shot_topics = zero_shot_topics.iloc[:max_topic_no,:]

                        if existing_unique_topics_df.empty:
                            existing_unique_topics_df = pd.DataFrame(data={'General Topic':zero_shot_topics.iloc[:,0], 'Subtopic':zero_shot_topics.iloc[:,1]})
                        
                        zero_shot_topics_df = zero_shot_topics

                if candidate_topics and not zero_shot_topics_df.empty:
                    # If you have already created revised zero shot topics, concat to the current
                    existing_unique_topics_df = pd.concat([existing_unique_topics_df, zero_shot_topics_df])

                    #existing_unique_topics_df.to_csv(output_folder + "Existing topics with zero shot dropped.csv", index = None)

                #all_topic_tables_df_merged = existing_unique_topics_df
                existing_unique_topics_df["Response References"] = ""

                unique_topics_markdown = existing_unique_topics_df[["General Topic", "Subtopic"]].drop_duplicates(["Subtopic"]).to_markdown(index=False)
            
                #existing_unique_topics_df.to_csv(output_folder + f"{file_name}_existing_unique_topics_df_" + #model_choice_clean + "_temp_" + str(temperature) + "_batch_" + str(latest_batch_completed) + ".csv", index=None)

                # Format the summary prompt with the response table and topics
                formatted_system_prompt = add_existing_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)
                formatted_summary_prompt = add_existing_topics_prompt.format(response_table=normalised_simple_markdown_table, topics=unique_topics_markdown)
                

                if model_choice == "gemma_2b_it_local":
                    formatted_summary_prompt = llama_cpp_prefix + formatted_system_prompt + "\n" + formatted_summary_prompt + llama_cpp_suffix
                    full_prompt = formatted_summary_prompt
                else:
                    full_prompt = formatted_system_prompt + formatted_summary_prompt
                    
                #latest_batch_number_string = "batch_" + str(latest_batch_completed - 1)

                # Define the output file path for the formatted prompt
                formatted_prompt_output_path = output_folder + file_name + "_" + str(reported_batch_no) +  "_full_prompt_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                # Write the formatted prompt to the specified file
                try:
                    with open(formatted_prompt_output_path, "w", encoding='utf-8', errors='replace') as f:
                        f.write(full_prompt)
                except Exception as e:
                    print(f"Error writing prompt to file {formatted_prompt_output_path}: {e}")

                summary_prompt_list = [formatted_summary_prompt]

                # print("master_summary_prompt_list:", summary_prompt_list[0])

                summary_conversation_history = []
                summary_whole_conversation = []

                # Process requests to large language model
                master_summary_response, summary_conversation_history, whole_summary_conversation, whole_conversation_metadata, response_text = process_requests(summary_prompt_list, add_existing_topics_system_prompt, summary_conversation_history, summary_whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, master = True)

                # print("master_summary_response:", master_summary_response[-1].text)
                # print("Whole conversation metadata:", whole_conversation_metadata)

                topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, new_topic_df, new_markdown_table, new_reference_df, new_unique_topics_df, master_batch_out_file_part, is_error =  write_llm_output_and_logs(master_summary_response, whole_summary_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, batch_size, chosen_cols, first_run=False)

                # If error in table parsing, leave function
                if is_error == True:
                    final_message_out = "Could not complete summary, error in LLM output."
                    display_table, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths#, final_message_out

                # Write outputs to csv
                ## Topics with references
                new_topic_df.to_csv(topic_table_out_path, index=None)
                log_files_output_paths.append(topic_table_out_path)

                ## Reference table mapping response numbers to topics
                new_reference_df.to_csv(reference_table_out_path, index=None)
                out_file_paths.append(reference_table_out_path)

                ## Unique topic list
                new_unique_topics_df = pd.concat([new_unique_topics_df, existing_unique_topics_df]).drop_duplicates('Subtopic')

                new_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
                out_file_paths.append(unique_topics_df_out_path)
                
                #all_topic_tables_df.append(new_topic_df)
                #all_markdown_topic_tables.append(new_markdown_table)

                #display_table = master_summary_response[-1].text

                # Show unique topics alongside document counts as output
                display_table = new_unique_topics_df.to_markdown(index=False)

                #whole_conversation_metadata.append(whole_conversation_metadata_str)
                whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)
                

                #out_file_paths = [col for col in out_file_paths if latest_batch_number_string in col]
                #log_files_output_paths = [col for col in log_files_output_paths if latest_batch_number_string in col]

                out_file_paths = [col for col in out_file_paths if str(reported_batch_no) in col]
                log_files_output_paths = [col for col in out_file_paths if str(reported_batch_no) in col]

                print("out_file_paths at end of loop:", out_file_paths)

            # If this is the first batch, run this
            else:
                #system_prompt = system_prompt + normalised_simple_markdown_table

                # Prepare Gemini models before query       
                if model_choice in ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
                    print("Using Gemini model:", model_choice)
                    model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
                else:
                    print("Using AWS Bedrock model:", model_choice)

                formatted_initial_table_prompt = initial_table_prompt.format(response_table=normalised_simple_markdown_table)

                formatted_initial_table_system_prompt = system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                if prompt2: formatted_prompt2 = prompt2.format(response_table=normalised_simple_markdown_table)
                else: formatted_prompt2 = prompt2
                
                if prompt3: formatted_prompt3 = prompt3.format(response_table=normalised_simple_markdown_table)
                else: formatted_prompt3 = prompt3

                if model_choice == "gemma_2b_it_local":
                    formatted_initial_table_prompt = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_initial_table_prompt + llama_cpp_suffix
                    formatted_prompt2 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt2 + llama_cpp_suffix
                    formatted_prompt3 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt3 + llama_cpp_suffix

                batch_prompts = [formatted_initial_table_prompt, formatted_prompt2, formatted_prompt3][:number_of_prompts_used]  # Adjust this list to send fewer requests 
                
                whole_conversation = [formatted_initial_table_system_prompt] 

                # Process requests to large language model
                responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(batch_prompts, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model)
                
                # print("Whole conversation metadata before:", whole_conversation_metadata)

                # print("responses:", responses[-1].text)
                # print("Whole conversation metadata:", whole_conversation_metadata)

                topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_table_df, markdown_table, reference_df, new_unique_topics_df, batch_file_path_details, is_error =  write_llm_output_and_logs(responses, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, batch_size, chosen_cols, first_run=True)

                # If error in table parsing, leave function
                if is_error == True:
                    display_table, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths#, final_message_out
                
                
                #all_topic_tables_df.append(topic_table_df)

                topic_table_df.to_csv(topic_table_out_path, index=None)
                out_file_paths.append(topic_table_out_path)

                reference_df.to_csv(reference_table_out_path, index=None)
                out_file_paths.append(reference_table_out_path)

                ## Unique topic list

                new_unique_topics_df = pd.concat([new_unique_topics_df, existing_unique_topics_df]).drop_duplicates('Subtopic')

                new_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
                out_file_paths.append(unique_topics_df_out_path)
                
                #all_markdown_topic_tables.append(markdown_table)

                whole_conversation_metadata.append(whole_conversation_metadata_str)
                whole_conversation_metadata_str = '. '.join(whole_conversation_metadata)
                
                # Write final output to text file also
                try:
                    final_table_output_path = output_folder + batch_file_path_details + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                    if isinstance(responses[-1], ResponseObject):
                        with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                            f.write(responses[-1].text)
                        display_table = responses[-1].text
                    elif "choices" in responses[-1]:
                        with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                            f.write(responses[-1]["choices"][0]['text'])
                        display_table =responses[-1]["choices"][0]['text']
                    else:
                        with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                            f.write(responses[-1].text)
                        display_table = responses[-1].text

                    log_files_output_paths.append(final_table_output_path)

                except Exception as e:
                    print(e)
                
                new_topic_df = topic_table_df
                new_reference_df = reference_df

        else:
            print("Current batch of responses contains no text, moving onto next. Batch number:", str(latest_batch_completed + 1), ". Start row:", start_row, ". End row:", end_row)

        # Increase latest file completed count unless we are at the last file
        if latest_batch_completed != num_batches:
            print("Completed batch number:", str(reported_batch_no))
            latest_batch_completed += 1 

        toc = time.perf_counter()
        final_time = toc - tic

        if final_time > max_time_for_loop:
            print("Max time reached, breaking loop.")
            topics_loop.close()
            tqdm._instances.clear()
            break

        # Overwrite 'existing' elements to add new tables
        existing_reference_df = new_reference_df.dropna(how='all')
        existing_unique_topics_df = new_unique_topics_df.dropna(how='all')
        existing_topics_table = new_topic_df.dropna(how='all')

    out_time = f"{final_time:0.1f} seconds."
    
    out_message.append('All queries successfully completed in')

    final_message_out = '\n'.join(out_message)
    final_message_out = final_message_out + " " + out_time  

    print(final_message_out) 

    return display_table, existing_topics_table, existing_unique_topics_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths

# SUMMARISATION FUNCTIONS

def deduplicate_categories(category_series: pd.Series, join_series: pd.Series, reference_df: pd.DataFrame, merge_sentiment:str="Yes", threshold: float = deduplication_threshold) -> pd.DataFrame:
    """
    Deduplicates similar category names in a pandas Series based on a fuzzy matching threshold,
    merging smaller topics into larger topics.

    Parameters:
        category_series (pd.Series): Series containing category names to deduplicate.
        join_series (pd.Series): Additional series used for joining back to original results.
        reference_df (pd.DataFrame): DataFrame containing the reference data to count occurrences.
        threshold (float): Similarity threshold for considering two strings as duplicates.

    Returns:
        pd.DataFrame: DataFrame with columns ['old_category', 'deduplicated_category'].
    """
    # Count occurrences of each category in the reference_df
    category_counts = reference_df['Subtopic'].value_counts().to_dict()

    # Initialize the result dictionary
    deduplication_map = {}

    # First pass: Handle exact matches
    for category in category_series.unique():
        if category in deduplication_map:
            continue
            
        # Find all exact matches
        exact_matches = category_series[category_series.str.lower() == category.lower()].index.tolist()
        if len(exact_matches) > 1:
            # Find the variant with the highest count
            match_counts = {match: category_counts.get(category_series[match], 0) for match in exact_matches}
            most_common = max(match_counts.items(), key=lambda x: x[1])[0]
            most_common_category = category_series[most_common]
            
            # Map all exact matches to the most common variant
            for match in exact_matches:
                deduplication_map[category_series[match]] = most_common_category

    # Second pass: Handle fuzzy matches for remaining categories
    for category in category_series.unique():
        # Skip if the category is already processed
        if category in deduplication_map:
            continue

        # Find close matches to the current category, excluding the current category itself
        matches = process.extract(category, 
                                [cat for cat in category_series.unique() if cat != category], 
                                scorer=fuzz.token_set_ratio, 
                                score_cutoff=threshold)

        if matches:  # Check if there are any matches
            best_match = max(matches, key=lambda x: x[1])  # Get the match with the highest score
            match, score, _ = best_match  # Unpack the best match

            # Compare counts to ensure smaller topics merge into larger ones
            if category_counts.get(category, 0) < category_counts.get(match, 0):
                deduplication_map[category] = match  # Map the smaller category to the larger one
            else:
                deduplication_map[match] = category  # Map the larger category to the smaller one
        else:
            deduplication_map[category] = category  # No match found, keep the category as is

    # Create the result DataFrame
    if merge_sentiment == "Yes":
        result_df = pd.DataFrame({
            'old_category': category_series + " | " + join_series,
            'deduplicated_category': category_series.map(lambda x: deduplication_map.get(x, x))
        })
    else:
        result_df = pd.DataFrame({
            'old_category': category_series + " | " + join_series,
            'deduplicated_category': category_series.map(lambda x: deduplication_map.get(x, x))
        })

    return result_df

def deduplicate_topics(reference_df,
                       unique_topics_df,
                       reference_table_file_name:str,
                       unique_topics_table_file_name:str,
                       merge_sentiment:str= "No",
                       merge_general_topics:str="No",
                       score_threshold:int=deduplication_threshold,
                       deduplicate_topics:str="Yes"):
    '''
    Deduplicate topics based on a reference and unique topics table
    '''
    output_files = []

    reference_table_file_name_no_ext = get_file_path_end(reference_table_file_name)
    unique_topics_table_file_name_no_ext = get_file_path_end(unique_topics_table_file_name)

    # Run through this x times to try to get all duplicate topics
    if deduplicate_topics == "Yes":
        for i in range(0, 5):
            #print("Deduplication run:", i)
            

            #reference_df_unique[["old_category"]].to_csv(output_folder + "reference_df_unique_old_categories_" + str(i) + ".csv", index=None)

            if merge_sentiment == "No":
                # First, combine duplicate topics in reference_df
                reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                reference_df_unique = reference_df.drop_duplicates("old_category")

                # Deduplicate categories within each sentiment group
                deduplicated_topic_map_df = reference_df_unique.groupby("Sentiment").apply(
                    lambda group: deduplicate_categories(group["Subtopic"], group["Sentiment"], reference_df, threshold=score_threshold)
                ).reset_index(drop=True)  # Reset index after groupby
            else:
                # Deduplicate categories by subtopic name only
                # First, combine duplicate topics in reference_df
                reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                reference_df_unique = reference_df.drop_duplicates("old_category")

                deduplicated_topic_map_df = deduplicate_categories(reference_df_unique["Subtopic"], reference_df_unique["Sentiment"], reference_df, merge_sentiment=merge_sentiment, threshold=score_threshold).reset_index(drop=True)
           
            if deduplicated_topic_map_df['deduplicated_category'].isnull().all():
            # Check if 'deduplicated_category' contains any values
                print("No deduplicated categories found, skipping the following code.")

            else:
                # Join deduplicated columns back to original df
                deduplicated_topic_map_df.to_csv(output_folder + "deduplicated_topic_map_df_" + str(i) + ".csv", index=None)
                # Remove rows where 'deduplicated_category' is blank or NaN
                deduplicated_topic_map_df = deduplicated_topic_map_df.loc[(deduplicated_topic_map_df['deduplicated_category'].str.strip() != '') & ~(deduplicated_topic_map_df['deduplicated_category'].isnull()), ['old_category','deduplicated_category']]

                deduplicated_topic_map_df.to_csv(output_folder + "deduplicated_topic_map_df_" + str(i) + ".csv", index=None)

                reference_df = reference_df.merge(deduplicated_topic_map_df, on="old_category", how="left")

                reference_df.rename(columns={"Subtopic": "Subtopic_old", "Sentiment": "Sentiment_old"}, inplace=True)
                # Extract subtopic and sentiment from deduplicated_category
                reference_df["Subtopic"] = reference_df["deduplicated_category"].str.extract(r'^(.*?) \|')[0]  # Extract subtopic
                reference_df["Sentiment"] = reference_df["deduplicated_category"].str.extract(r'\| (.*)$')[0]  # Extract sentiment

                # Combine with old values to ensure no data is lost
                reference_df["Subtopic"] = reference_df["deduplicated_category"].combine_first(reference_df["Subtopic_old"])
                reference_df["Sentiment"] = reference_df["Sentiment"].combine_first(reference_df["Sentiment_old"])

            #reference_df.to_csv(output_folder + "reference_table_after_dedup.csv", index=None)

            reference_df.drop(['old_category', 'deduplicated_category', "Subtopic_old", "Sentiment_old"], axis=1, inplace=True, errors="ignore")

            reference_df = reference_df[["Response References", "General Topic", "Subtopic", "Sentiment", "Summary", "Start row of group"]]

            reference_df["General Topic"] = reference_df["General Topic"].str.lower().str.capitalize() 
            reference_df["Subtopic"] = reference_df["Subtopic"].str.lower().str.capitalize() 
            reference_df["Sentiment"] = reference_df["Sentiment"].str.lower().str.capitalize() 

            if merge_general_topics == "Yes":
                # Replace General topic names for each Subtopic with that for the Subtopic with the most responses
                # Step 1: Count the number of occurrences for each General Topic and Subtopic combination
                count_df = reference_df.groupby(['Subtopic', 'General Topic']).size().reset_index(name='Count')

                # Step 2: Find the General Topic with the maximum count for each Subtopic
                max_general_topic = count_df.loc[count_df.groupby('Subtopic')['Count'].idxmax()]

                # Step 3: Map the General Topic back to the original DataFrame
                reference_df = reference_df.merge(max_general_topic[['Subtopic', 'General Topic']], on='Subtopic', suffixes=('', '_max'), how='left')

                reference_df['General Topic'] = reference_df["General Topic_max"].combine_first(reference_df["General Topic"])        

            if merge_sentiment == "Yes":
                # Step 1: Count the number of occurrences for each General Topic and Subtopic combination
                count_df = reference_df.groupby(['Subtopic', 'Sentiment']).size().reset_index(name='Count')

                # Step 2: Determine the number of unique Sentiment values for each Subtopic
                unique_sentiments = count_df.groupby('Subtopic')['Sentiment'].nunique().reset_index(name='UniqueCount')

                # Step 3: Update Sentiment to 'Mixed' where there is more than one unique sentiment
                reference_df = reference_df.merge(unique_sentiments, on='Subtopic', how='left')
                reference_df['Sentiment'] = reference_df.apply(
                    lambda row: 'Mixed' if row['UniqueCount'] > 1 else row['Sentiment'],
                    axis=1
                )

                # Clean up the DataFrame by dropping the UniqueCount column
                reference_df.drop(columns=['UniqueCount'], inplace=True)

            reference_df = reference_df[["Response References", "General Topic", "Subtopic", "Sentiment", "Summary", "Start row of group"]]

        # Remake unique_topics_df based on new reference_df
        unique_topics_df = create_unique_table_df_from_reference_table(reference_df)

        reference_table_file_name_no_ext = get_file_path_end(reference_table_file_name)
        unique_topics_table_file_name_no_ext = get_file_path_end(unique_topics_table_file_name)

        reference_file_path = output_folder + reference_table_file_name_no_ext + "_dedup.csv"
        unique_topics_file_path = output_folder + unique_topics_table_file_name_no_ext + "_dedup.csv"
        reference_df.to_csv(reference_file_path, index = None)
        unique_topics_df.to_csv(unique_topics_file_path, index=None)

        output_files.append(reference_file_path)
        output_files.append(unique_topics_file_path)

    return reference_df, unique_topics_df, output_files

def sample_reference_table_summaries(reference_df:pd.DataFrame,
                                     unique_topics_df:pd.DataFrame,
                                     random_seed:int,
                                     no_of_sampled_summaries:int=150):
    
    '''
    Sample x number of summaries from which to produce summaries, so that the input token length is not too long.
    '''
    
    all_summaries = pd.DataFrame()
    output_files = []

    reference_df_grouped = reference_df.groupby(["General Topic", "Subtopic", "Sentiment"])

    for group_keys, reference_df_group in reference_df_grouped:
        #print(f"Group: {group_keys}")
        #print(f"Data: {reference_df_group}")

        if len(reference_df_group["General Topic"]) > 1:

            filtered_reference_df = reference_df_group.reset_index()

            filtered_reference_df_unique = filtered_reference_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment", "Summary"])

            # Sample n of the unique topic summaries. To limit the length of the text going into the summarisation tool
            filtered_reference_df_unique_sampled = filtered_reference_df_unique.sample(min(no_of_sampled_summaries, len(filtered_reference_df_unique)), random_state=random_seed)

            #topic_summary_table_markdown = filtered_reference_df_unique_sampled.to_markdown(index=False)

            #print(filtered_reference_df_unique_sampled)

            all_summaries = pd.concat([all_summaries, filtered_reference_df_unique_sampled])

    #all_summaries.to_csv(output_folder + "all_summaries.csv", index=None)
    
    summarised_references = all_summaries.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': lambda x: '\n'.join([s.split(': ', 1)[1] for s in x if ': ' in s])  # Join substrings after ': '
    }).reset_index()

    summarised_references = summarised_references.loc[(summarised_references["Sentiment"] != "Not Mentioned") & (summarised_references["Response References"] > 1)]

    #summarised_references.to_csv(output_folder + "summarised_references.csv", index=None)

    summarised_references_markdown = summarised_references.to_markdown(index=False)

    return summarised_references, summarised_references_markdown, reference_df, unique_topics_df

def summarise_output_topics_query(model_choice:str, in_api_key:str, temperature:float, formatted_summary_prompt:str, summarise_topic_descriptions_system_prompt:str, local_model=[]):
    conversation_history = []
    whole_conversation_metadata = []

    # Prepare Gemini models before query       
    if model_choice in ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
        print("Using Gemini model:", model_choice)
        model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
    else:
        print("Using AWS Bedrock model:", model_choice)
        model = model_choice
        config = {}

    whole_conversation = [summarise_topic_descriptions_system_prompt] 

    # Process requests to large language model
    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(formatted_summary_prompt, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, local_model=local_model)

    print("Finished summary query")

    if isinstance(responses[-1], ResponseObject):
        response_texts = [resp.text for resp in responses]
    elif "choices" in responses[-1]:
        response_texts = [resp["choices"][0]['text'] for resp in responses]
    else:
        response_texts = [resp.text for resp in responses]

    latest_response_text = response_texts[-1]

    #print("latest_response_text:", latest_response_text)
    #print("Whole conversation metadata:", whole_conversation_metadata)

    return latest_response_text, conversation_history, whole_conversation_metadata

@spaces.GPU
def summarise_output_topics(summarised_references:pd.DataFrame,
                            unique_table_df:pd.DataFrame,
                            reference_table_df:pd.DataFrame,
                            model_choice:str,
                            in_api_key:str,
                            topic_summary_table_markdown:str,
                            temperature:float,
                            table_file_name:str,
                            summarised_outputs:list = [],  
                            latest_summary_completed:int = 0,
                            out_metadata_str:str = "",
                            output_files:list = [],
                            summarise_topic_descriptions_prompt:str=summarise_topic_descriptions_prompt, summarise_topic_descriptions_system_prompt:str=summarise_topic_descriptions_system_prompt,
                            do_summaries="Yes",
                            progress=gr.Progress(track_tqdm=True)):
    '''
    Create better summaries of the raw batch-level summaries created in the first run of the model.
    '''
    out_metadata = []
    local_model = []
    summarised_output_markdown = ""
    
    print("In summarise_output_topics function.")

    all_summaries = summarised_references["Summary"].tolist()

    length_all_summaries = len(all_summaries)

    #print("latest_summary_completed:", latest_summary_completed)
    #print("length_all_summaries:", length_all_summaries)

    # If all summaries completed, make final outputs
    if latest_summary_completed >= length_all_summaries:
        print("All summaries completed. Creating outputs.")

        model_choice_clean = model_name_map[model_choice]   
        file_name = re.search(r'(.*?)(?:_batch_|_col_)', table_file_name).group(1) if re.search(r'(.*?)(?:_batch_|_col_)', table_file_name) else table_file_name
        latest_batch_completed = int(re.search(r'batch_(\d+)_', table_file_name).group(1)) if 'batch_' in table_file_name else ""
        batch_size_number = int(re.search(r'size_(\d+)_', table_file_name).group(1)) if 'size_' in table_file_name else ""
        in_column_cleaned = re.search(r'col_(.*?)_reference', table_file_name).group(1) if 'col_' in table_file_name else ""

        # Save outputs for each batch. If master file created, label file as master
        if latest_batch_completed:
            batch_file_path_details = f"{file_name}_batch_{latest_batch_completed}_size_{batch_size_number}_col_{in_column_cleaned}"
        else:
            batch_file_path_details = f"{file_name}_col_{in_column_cleaned}"

        summarised_references["Revised summary"] = summarised_outputs        

        join_cols = ["General Topic", "Subtopic", "Sentiment"]
        join_plus_summary_cols = ["General Topic", "Subtopic", "Sentiment", "Revised summary"]

        summarised_references_j = summarised_references[join_plus_summary_cols].drop_duplicates(join_plus_summary_cols)

        unique_table_df_revised = unique_table_df.merge(summarised_references_j, on = join_cols, how = "left")
        # If no new summary is available, keep the original
        unique_table_df_revised["Revised summary"] = unique_table_df_revised["Revised summary"].combine_first(unique_table_df_revised["Summary"])

        unique_table_df_revised = unique_table_df_revised[["General Topic",	"Subtopic",	"Sentiment", "Response References",	"Revised summary"]]        

        reference_table_df_revised = reference_table_df.merge(summarised_references_j, on = join_cols, how = "left")
        # If no new summary is available, keep the original
        reference_table_df_revised["Revised summary"] = reference_table_df_revised["Revised summary"].combine_first(reference_table_df_revised["Summary"])
        reference_table_df_revised = reference_table_df_revised.drop("Summary", axis=1)

        # Remove topics that are tagged as 'Not Mentioned'
        unique_table_df_revised = unique_table_df_revised.loc[unique_table_df_revised["Sentiment"] != "Not Mentioned", :]
        reference_table_df_revised = reference_table_df_revised.loc[reference_table_df_revised["Sentiment"] != "Not Mentioned", :]

        unique_table_df_revised_path = output_folder + batch_file_path_details + "_summarised_unique_topic_table_" + model_choice_clean + ".csv"
        unique_table_df_revised.to_csv(unique_table_df_revised_path, index = None)

        reference_table_df_revised_path = output_folder + batch_file_path_details + "_summarised_reference_table_" + model_choice_clean + ".csv"
        reference_table_df_revised.to_csv(reference_table_df_revised_path, index = None)

        output_files.extend([reference_table_df_revised_path, unique_table_df_revised_path])

        unique_table_df_revised_display = unique_table_df_revised.apply(lambda col: col.map(wrap_text))

        summarised_output_markdown = unique_table_df_revised_display.to_markdown(index=False)

        return summarised_references, unique_table_df_revised, reference_table_df_revised, output_files, summarised_outputs, latest_summary_completed, out_metadata_str, summarised_output_markdown

    tic = time.perf_counter()
    
    #print("Starting with:", latest_summary_completed)
    #print("Last summary number:", length_all_summaries)

    if (model_choice == "gemma_2b_it_local") & (RUN_LOCAL_MODEL == "1"):
                progress(0.1, "Loading in Gemma 2b model")
                local_model, tokenizer = load_model()
                print("Local model loaded:", local_model)

    summary_loop_description = "Creating summaries. " + str(latest_summary_completed) + " summaries completed so far."
    summary_loop = tqdm(range(latest_summary_completed, length_all_summaries), desc="Creating summaries", unit="summaries")   

    if do_summaries == "Yes":
        for summary_no in summary_loop:

            print("Current summary number is:", summary_no)

            summary_text = all_summaries[summary_no]
            #print("summary_text:", summary_text)
            formatted_summary_prompt = [summarise_topic_descriptions_prompt.format(summaries=summary_text)]

            try:
                response, conversation_history, metadata = summarise_output_topics_query(model_choice, in_api_key, temperature, formatted_summary_prompt, summarise_topic_descriptions_system_prompt, local_model)
                summarised_output = response
                summarised_output = re.sub(r'\n{2,}', '\n', summarised_output)  # Replace multiple line breaks with a single line break
                summarised_output = re.sub(r'^\n{1,}', '', summarised_output)  # Remove one or more line breaks at the start
                summarised_output = summarised_output.strip()
            except Exception as e:
                print(e)
                summarised_output = ""

            summarised_outputs.append(summarised_output)
            out_metadata.extend(metadata)
            out_metadata_str = '. '.join(out_metadata)

            latest_summary_completed += 1

            # Check if beyond max time allowed for processing and break if necessary
            toc = time.perf_counter()
            time_taken = tic - toc

            if time_taken > max_time_for_loop:
                print("Time taken for loop is greater than maximum time allowed. Exiting and restarting loop")
                summary_loop.close()
                tqdm._instances.clear()
                break

    # If all summaries completeed
    if latest_summary_completed >= length_all_summaries:
        print("At last summary.")

    return summarised_references, unique_table_df, reference_table_df, output_files, summarised_outputs, latest_summary_completed, out_metadata_str, summarised_output_markdown
