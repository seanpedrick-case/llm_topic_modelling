import os
import google.generativeai as ai
import pandas as pd
import numpy as np
import gradio as gr
import markdown
import time
import boto3
import string
import re
import spaces
from tqdm import tqdm

from gradio import Progress
from typing import List, Tuple
from io import StringIO

GradioFileData = gr.FileData

from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt,  add_existing_topics_system_prompt, add_existing_topics_prompt,  force_existing_topics_prompt, allow_new_topics_prompt, force_single_topic_prompt
from tools.helper_functions import read_file, put_columns_in_df, wrap_text, initial_clean, load_in_data_file, load_in_file, create_topic_summary_df_from_reference_table, convert_reference_table_to_pivot_table, get_basic_response_data
from tools.llm_funcs import ResponseObject, process_requests, construct_gemini_generative_model
from tools.config import RUN_LOCAL_MODEL, AWS_REGION, MAX_COMMENT_CHARS, MAX_OUTPUT_VALIDATION_ATTEMPTS, MAX_TOKENS, TIMEOUT_WAIT, NUMBER_OF_RETRY_ATTEMPTS, MAX_TIME_FOR_LOOP, BATCH_SIZE_DEFAULT, DEDUPLICATION_THRESHOLD, RUN_AWS_FUNCTIONS, model_name_map, OUTPUT_FOLDER, CHOSEN_LOCAL_MODEL_TYPE, LOCAL_REPO_ID, LOCAL_MODEL_FILE, LOCAL_MODEL_FOLDER

if RUN_LOCAL_MODEL == "1":
    from tools.llm_funcs import load_model

max_tokens = MAX_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS

if RUN_AWS_FUNCTIONS == '1':
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
else:
    bedrock_runtime = []

### HELPER FUNCTIONS

def normalise_string(text:str):
    # Replace two or more dashes with a single dash
    text = re.sub(r'-{2,}', '-', text)
    
    # Replace two or more spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text

def load_in_previous_data_files(file_paths_partial_output:List[str], for_modified_table:bool=False):
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
                out_message = out_message + " Reference file load successful."
            except Exception as e:
                out_message = "Could not load reference file data:" + str(e)
                raise Exception("Could not load reference file data:", e)
        # If unique table
        if 'unique_topic' in file.name:
            try:
                unique_file_data, unique_file_name = load_in_file(file)
                #print("unique_topics_file:", unique_file_data.head(2))
                out_message = out_message + " Unique table file load successful."
            except Exception as e:
                out_message = "Could not load unique table file data:" + str(e)
                raise Exception("Could not load unique table file data:", e)
        if 'batch_' in file.name:
            latest_batch = re.search(r'batch_(\d+)', file.name).group(1)
            print("latest batch:", latest_batch)
            latest_batch = int(latest_batch)

    if latest_batch == 0:
        out_message = out_message + " Latest batch number not found."
    if reference_file_data.empty:
        out_message = out_message + " No reference data table provided."
        #raise Exception(out_message)
    if unique_file_data.empty:
        out_message = out_message + " No unique data table provided."   

    print(out_message)

    # Return all data if using for deduplication task. Return just modified unique table if using just for table modification
    if for_modified_table == False:            
        return reference_file_data, unique_file_data, latest_batch, out_message, reference_file_name, unique_file_name
    else:
        
        reference_file_data.drop("Topic_number", axis=1, inplace=True, errors="ignore")

        unique_file_data = create_topic_summary_df_from_reference_table(reference_file_data)

        unique_file_data.drop("Summary",axis=1, inplace=True)

        # Then merge the topic numbers back to the original dataframe
        reference_file_data = reference_file_data.merge(
            unique_file_data[['General Topic', 'Subtopic', 'Sentiment', 'Topic_number']],
            on=['General Topic', 'Subtopic', 'Sentiment'],
            how='left'
        )        

        out_file_names = [reference_file_name + ".csv"]
        out_file_names.append(unique_file_name + ".csv")

        print("reference_file_name:", reference_file_name)
        print("unique_file_name:", unique_file_name)

        return gr.Dataframe(value=unique_file_data, headers=None, col_count=(unique_file_data.shape[1], "fixed"), row_count = (unique_file_data.shape[0], "fixed"), visible=True, type="pandas"), reference_file_data, unique_file_data, reference_file_name, unique_file_name, out_file_names

def data_file_to_markdown_table(file_data:pd.DataFrame, file_name:str, chosen_cols: List[str], batch_number: int, batch_size: int, verify_titles:bool=False) -> Tuple[str, str, str]:
    """
    Processes a file by simplifying its content based on chosen columns and saves the result to a specified output folder.

    Parameters:
    - file_data (pd.DataFrame): Tabular data file with responses.
    - file_name (str): File name with extension.
    - chosen_cols (List[str]): A list of column names to include in the simplified file.
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
    basic_response_data = get_basic_response_data(file_data, chosen_cols, verify_titles=verify_titles)
    
    file_len = len(basic_response_data["Reference"])
   

     # Subset the data for the current batch
    start_row = batch_number * batch_size
    if start_row > file_len + 1:
        print("Start row greater than file row length")
        return simplified_csv_table_path, normalised_simple_markdown_table, file_name

    if (start_row + batch_size) <= file_len + 1:
        end_row = start_row + batch_size
    else:
        end_row = file_len + 1

    #print("start_row:", start_row)
    #print("end_row:", end_row)

    batch_basic_response_data = basic_response_data[start_row:end_row]  # Select the current batch

    #print("batch_basic_response_data:", batch_basic_response_data)

    # Now replace the reference numbers with numbers starting from 1
    batch_basic_response_data.loc[:, "Reference"] = batch_basic_response_data["Reference"] - start_row

    #print("batch_basic_response_data:", batch_basic_response_data)

    # Remove problematic characters including control characters, special characters, and excessive leading/trailing whitespace
    batch_basic_response_data.loc[:, "Response"]= batch_basic_response_data["Response"].str.replace(r'[\x00-\x1F\x7F]|[""<>]|\\', '', regex=True)  # Remove control and special characters
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data["Response"].str.strip()  # Remove leading and trailing whitespace
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data["Response"].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data["Response"].str.replace(r'\n{2,}', '\n', regex=True)  # Replace multiple line breaks with a single line break
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data["Response"].str.slice(0, max_comment_character_length) # Maximum 1,500 character responses

    # Remove blank and extremely short responses
    batch_basic_response_data = batch_basic_response_data.loc[~(batch_basic_response_data["Response"].isnull()) &\
                                  ~(batch_basic_response_data["Response"] == "None") &\
                                  ~(batch_basic_response_data["Response"] == " ") &\
                                  ~(batch_basic_response_data["Response"] == ""),:]#~(batch_basic_response_data["Response"].str.len() < 5), :]

    simple_markdown_table = batch_basic_response_data.to_markdown(index=None)

    normalised_simple_markdown_table = normalise_string(simple_markdown_table)

    return simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, batch_basic_response_data

def replace_punctuation_with_underscore(input_string):
    # Create a translation table where each punctuation character maps to '_'
    translation_table = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    
    # Translate the input string using the translation table
    return input_string.translate(translation_table)

### INITIAL TOPIC MODEL DEVELOPMENT FUNCTIONS

def clean_markdown_table(text: str):
    # Split text into lines
    lines = text.splitlines()
    
    # Step 1: Identify table structure and process line continuations
    table_rows = []
    current_row = None
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Check if this is a table row or alignment row
        is_table_row = '|' in stripped or stripped.startswith(':-') or ':-:' in stripped
        
        if is_table_row:
            # If we have a current row being built, add it to our list
            if current_row is not None:
                table_rows.append(current_row)
                
            # Start a new row
            current_row = stripped
        elif current_row is not None:
            # This must be a continuation of the previous row
            current_row += " " + stripped
        else:
            # Not part of the table
            current_row = stripped
    
    # Don't forget the last row
    if current_row is not None:
        table_rows.append(current_row)
    
    # Step 2: Properly format the table
    # First, determine the maximum number of columns
    max_columns = 0
    for row in table_rows:
        cells = row.split('|')
        # Account for rows that may not start/end with a pipe
        if row.startswith('|'):
            cells = cells[1:]
        if row.endswith('|'):
            cells = cells[:-1]
        max_columns = max(max_columns, len(cells))
    
    # Now format each row
    formatted_rows = []
    for row in table_rows:
        # Ensure the row starts and ends with pipes
        if not row.startswith('|'):
            row = '|' + row
        if not row.endswith('|'):
            row = row + '|'
            
        # Split into cells
        cells = row.split('|')[1:-1]  # Remove empty entries from split
        
        # Ensure we have the right number of cells
        while len(cells) < max_columns:
            cells.append('')
            
        # Rebuild the row
        formatted_row = '|' + '|'.join(cells) + '|'
        formatted_rows.append(formatted_row)
    
    # Join everything back together
    result = '\n'.join(formatted_rows)
    
    return result

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

# Convert output table to markdown and then to a pandas dataframe to csv
def remove_before_last_term(input_string: str) -> str:
    # Use regex to find the last occurrence of the term
    match = re.search(r'(\| ?General Topic)', input_string)
    if match:
        # Find the last occurrence by using rfind
        last_index = input_string.rfind(match.group(0))
        return input_string[last_index:]  # Return everything from the last match onward
    return input_string  # Return the original string if the term is not found

def convert_to_html_table(input_string: str, table_type: str = "Main table"):
    # Remove HTML tags from input string
    input_string = input_string.replace("<p>", "").replace("</p>", "")
    
    if "<table" in input_string:
        # Input is already in HTML format
        html_table = input_string
    else:
        # Input is in Markdown format
        print("input_string:", input_string)
        lines = input_string.strip().split("\n")
        clean_md_text = "\n".join([lines[0]] + lines[2:])  # Keep header, skip separator, keep data
        
        # Read Markdown table into a DataFrame
        df = pd.read_csv(pd.io.common.StringIO(clean_md_text), sep="|", skipinitialspace=True)
        
        # Ensure unique column names
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col for i, col in enumerate(df.columns)]
        
        # Convert DataFrame to HTML
        html_table = df.to_html(index=False, border=1)
    
    # Ensure that the HTML structure is correct
    if table_type == "Main table":
        if "<table" not in html_table:
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
        if "<table" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>General Topic</th>
                    <th>Subtopic</th>
                </tr>
                {html_table}
            </table>
            """
    elif table_type == "Verify titles table":        
        if "<table" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>Response References</th>
                    <th>Is this a suitable title</th>
                    <th>Explanation</th>
                    <th>Alternative title</th>
                </tr>
                {html_table}
            </table>
            """            
    
    return html_table

def convert_response_text_to_markdown_table(response_text:str, table_type:str = "Main table"):
    is_error = False
    start_of_table_response = remove_before_last_term(response_text)

    cleaned_response = clean_markdown_table(start_of_table_response)

    try:
        string_html_table = markdown.markdown(cleaned_response, extensions=['markdown.extensions.tables'])
    except Exception as e:
        print("Unable to convert response to string_html_table due to", e)
        string_html_table = ""

    html_table = convert_to_html_table(string_html_table)
          
    html_buffer = StringIO(html_table)

    try:
        tables = pd.read_html(html_buffer)
        if tables:
            out_df = tables[0]  # Use the first table if available
        else:
            raise ValueError("No tables found in the provided HTML.")
            is_error = True
            out_df = pd.DataFrame()
    except Exception as e:
        print("Error when trying to parse table:", e)
        is_error = True
        out_df = pd.DataFrame()

    return out_df, is_error
    
    #print("out_df in convert function:", out_df)

def call_llm_with_markdown_table_checks(batch_prompts: List[str],
                                        system_prompt: str,
                                        conversation_history: List[dict],
                                        whole_conversation: List[str], 
                                        whole_conversation_metadata: List[str],
                                        model: object,
                                        config: dict,
                                        model_choice: str, 
                                        temperature: float,
                                        reported_batch_no: int,
                                        local_model: object,
                                        MAX_OUTPUT_VALIDATION_ATTEMPTS: int,                                        
                                        master:bool=False,
                                        CHOSEN_LOCAL_MODEL_TYPE:str=CHOSEN_LOCAL_MODEL_TYPE) -> Tuple[List[ResponseObject], List[dict], List[str], List[str], str]:
    """
    Call the large language model with checks for a valid markdown table.

    Parameters:
    - batch_prompts (List[str]): A list of prompts to be processed.
    - system_prompt (str): The system prompt.
    - conversation_history (List[dict]): The history of the conversation.
    - whole_conversation (List[str]): The complete conversation including prompts and responses.
    - whole_conversation_metadata (List[str]): Metadata about the whole conversation.
    - model (object): The model to use for processing the prompts.
    - config (dict): Configuration for the model.
    - model_choice (str): The choice of model to use.        
    - temperature (float): The temperature parameter for the model.
    - reported_batch_no (int): The reported batch number.
    - local_model (object): The local model to use.
    - MAX_OUTPUT_VALIDATION_ATTEMPTS (int): The maximum number of attempts to validate the output.
    - master (bool, optional): Boolean to determine whether this call is for the master output table.

    Returns:
    - Tuple[List[ResponseObject], List[dict], List[str], List[str], str]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, the updated whole conversation metadata, and the response text.
    """

    call_temperature = temperature  # This is correct now with the fixed parameter name

    # Update Gemini config with the temperature settings
    config = ai.GenerationConfig(temperature=call_temperature, max_output_tokens=max_tokens)

    for attempt in range(MAX_OUTPUT_VALIDATION_ATTEMPTS):
        # Process requests to large language model
        responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(
            batch_prompts, system_prompt, conversation_history, whole_conversation, 
            whole_conversation_metadata, model, config, model_choice, 
            call_temperature, reported_batch_no, local_model, master=master
        )

        if model_choice != CHOSEN_LOCAL_MODEL_TYPE:
            stripped_response = responses[-1].text.strip()
        else:
            stripped_response = responses[-1]['choices'][0]['text'].strip()

        # Check if response meets our criteria (length and contains table)
        if len(stripped_response) > 120 and '|' in stripped_response:
            print(f"Attempt {attempt + 1} produced response with markdown table.")
            break  # Success - exit loop

        # Increase temperature for next attempt
        call_temperature = temperature + (0.1 * (attempt + 1))
        print(f"Attempt {attempt + 1} resulted in invalid table: {stripped_response}. "
            f"Trying again with temperature: {call_temperature}")

    else:  # This runs if no break occurred (all attempts failed)
        print(f"Failed to get valid response after {MAX_OUTPUT_VALIDATION_ATTEMPTS} attempts")

    return responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text

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
                              first_run: bool = False,
                              output_folder:str=OUTPUT_FOLDER) -> None:
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
    topic_summary_df_out_path = []
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    topic_summary_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame()
    markdown_table = ""
    out_reference_df = pd.DataFrame()
    out_topic_summary_df = pd.DataFrame()
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

    whole_conversation_path = output_folder + batch_file_path_details + "_full_conversation_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"
    whole_conversation_path_meta = output_folder + batch_file_path_details + "_metadata_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

    with open(whole_conversation_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_str)

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
        return topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_topic_summary_df, batch_file_path_details, is_error

    # Rename columns to ensure consistent use of data frames later in code
    new_column_names = {
    topic_with_response_df.columns[0]: "General Topic",
    topic_with_response_df.columns[1]: "Subtopic",
    topic_with_response_df.columns[2]: "Sentiment",
    topic_with_response_df.columns[3]: "Response References",
    topic_with_response_df.columns[4]: "Summary"
    }

    topic_with_response_df = topic_with_response_df.rename(columns=new_column_names)


    # Fill in NA rows with values from above (topics seem to be included only on one row):
    topic_with_response_df = topic_with_response_df.ffill()

    #print("topic_with_response_df:", topic_with_response_df)

    # For instances where you end up with float values in Response References
    topic_with_response_df["Response References"] = topic_with_response_df["Response References"].astype(str).str.replace(".0", "", regex=False)

    # Strip and lower case topic names to remove issues where model is randomly capitalising topics/sentiment
    topic_with_response_df["General Topic"] = topic_with_response_df["General Topic"].astype(str).str.strip().str.lower().str.capitalize()
    topic_with_response_df["Subtopic"] = topic_with_response_df["Subtopic"].astype(str).str.strip().str.lower().str.capitalize()
    topic_with_response_df["Sentiment"] = topic_with_response_df["Sentiment"].astype(str).str.strip().str.lower().str.capitalize()
    
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
        
        # Filter out references that are outside the valid range
        if references:
            try:
                # Convert all references to integers and keep only those within valid range
                ref_numbers = [int(ref) for ref in references]
                references = [str(ref) for ref in ref_numbers if 1 <= ref <= batch_size_number]
            except ValueError:
                # If any reference can't be converted to int, skip this row
                print("Response value could not be converted to number:", references)
                continue
        
        topic = row.iloc[0] if pd.notna(row.iloc[0]) else ""
        subtopic = row.iloc[1] if pd.notna(row.iloc[1]) else ""
        sentiment = row.iloc[2] if pd.notna(row.iloc[2]) else ""
        summary = row.iloc[4] if pd.notna(row.iloc[4]) else ""
        # If the reference response column is very long, and there's nothing in the summary column, assume that the summary was put in the reference column
        if not summary and (len(str(row.iloc[3])) > 30):
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

    #print("new_reference_df:", new_reference_df)
    
    # Append on old reference data
    out_reference_df = pd.concat([new_reference_df, existing_reference_df]).dropna(how='all')

    # Remove duplicate Response References for the same topic
    out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    # Try converting response references column to int, keep as string if fails
    try:
        out_reference_df["Response References"] = out_reference_df["Response References"].astype(int)
    except Exception as e:
        print("Could not convert Response References column to integer due to", e)
        print("out_reference_df['Response References']:", out_reference_df["Response References"].head())

    out_reference_df.sort_values(["Start row of group", "Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    # Each topic should only be associated with each individual response once
    out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    # Save the new DataFrame to CSV
    reference_table_out_path = output_folder + batch_file_path_details + "_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"    

    # Table of all unique topics with descriptions
    #print("topic_with_response_df:", topic_with_response_df)
    new_topic_summary_df = topic_with_response_df[["General Topic", "Subtopic", "Sentiment"]]

    new_topic_summary_df = new_topic_summary_df.rename(columns={new_topic_summary_df.columns[0]: "General Topic", new_topic_summary_df.columns[1]: "Subtopic", new_topic_summary_df.columns[2]: "Sentiment"})
    
    # Join existing and new unique topics
    out_topic_summary_df = pd.concat([new_topic_summary_df, existing_topics_df]).dropna(how='all')

    out_topic_summary_df = out_topic_summary_df.rename(columns={out_topic_summary_df.columns[0]: "General Topic", out_topic_summary_df.columns[1]: "Subtopic", out_topic_summary_df.columns[2]: "Sentiment"})

    #print("out_topic_summary_df:", out_topic_summary_df)

    out_topic_summary_df = out_topic_summary_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment"]).\
            drop(["Number of responses", "Summary"], axis = 1, errors="ignore") 

    # Get count of rows that refer to particular topics
    reference_counts = out_reference_df.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': ' <br> '.join
    }).reset_index()

    # Join the counts to existing_topic_summary_df
    out_topic_summary_df = out_topic_summary_df.merge(reference_counts, how='left', on=["General Topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    out_topic_summary_df = out_topic_summary_df.rename(columns={"Response References":"Number of responses"}, errors="ignore")

    topic_summary_df_out_path = output_folder + batch_file_path_details + "_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

    return topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_topic_summary_df, batch_file_path_details, is_error

def generate_zero_shot_topics_df(zero_shot_topics:pd.DataFrame,
                                 force_zero_shot_radio:str="No",
                                 create_revised_general_topics:bool=False,
                                 max_topic_no:int=120):

    # Max 120 topics allowed
    if zero_shot_topics.shape[0] > max_topic_no:
        print("Maximum", max_topic_no, "topics allowed to fit within large language model context limits.")
        zero_shot_topics = zero_shot_topics.iloc[:max_topic_no, :]

    # Forward slashes in the topic names seems to confuse the model
    if zero_shot_topics.shape[1] >= 1:  # Check if there is at least one column                       
        for x in zero_shot_topics.columns:
            if not zero_shot_topics[x].isnull().all():
                zero_shot_topics[x] = zero_shot_topics[x].apply(initial_clean)

                zero_shot_topics.loc[:, x] = (
                zero_shot_topics.loc[:, x]
                .str.strip()
                .str.replace('\n', ' ')
                .str.replace('\r', ' ')
                .str.replace('/', ' or ')
                .str.lower()
                .str.capitalize())            

        #print("zero_shot_topics:", zero_shot_topics)

        # If number of columns is 1, keep only subtopics
        if zero_shot_topics.shape[1] == 1 and "General topic" not in zero_shot_topics.columns: 
            zero_shot_topics_gen_topics_list = [""] * zero_shot_topics.shape[0]
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 0])                    
        # Allow for possibility that the user only wants to set general topics and not subtopics
        elif zero_shot_topics.shape[1] == 1 and "General topic" in zero_shot_topics.columns: 
            zero_shot_topics_gen_topics_list = list(zero_shot_topics["General Topic"])
            zero_shot_topics_subtopics_list = [""] * zero_shot_topics.shape[0]
        # If general topic and subtopic are specified
        elif set(["General topic", "Subtopic"]).issubset(zero_shot_topics.columns):
            print("Found General topic and Subtopic in zero shot topics")
            zero_shot_topics_gen_topics_list = list(zero_shot_topics["General topic"])
            zero_shot_topics_subtopics_list = list(zero_shot_topics["Subtopic"])

        # If number of columns is at least 2, keep general topics and subtopics
        elif zero_shot_topics.shape[1] >= 2 and "Description" not in zero_shot_topics.columns: 
            zero_shot_topics_gen_topics_list = list(zero_shot_topics.iloc[:, 0])
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 1])
        else:
            # If there are more columns, just assume that the first column was meant to be a subtopic
            zero_shot_topics_gen_topics_list = [""] * zero_shot_topics.shape[0]
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 0])

        # Add a description if column is present
        # print("zero_shot_topics.shape[1]:", zero_shot_topics.shape[1])                         
        if "Description" in zero_shot_topics.columns:
            zero_shot_topics_description_list = list(zero_shot_topics["Description"])
            #print("Description found in topic title. List is:", zero_shot_topics_description_list)        
        elif zero_shot_topics.shape[1] >= 3:
            zero_shot_topics_description_list = list(zero_shot_topics.iloc[:, 2]) # Assume the third column is description
        else:
            zero_shot_topics_description_list = [""] * zero_shot_topics.shape[0]

        # If the responses are being forced into zero shot topics, allow an option for nothing relevant
        if force_zero_shot_radio == "Yes":
            zero_shot_topics_gen_topics_list.append("")
            zero_shot_topics_subtopics_list.append("No relevant topic")
            zero_shot_topics_description_list.append("")                     

        if create_revised_general_topics == True:
            pass

            # The following currently doesn't really work. Excluded for now.

            # topic_summary_df = pd.DataFrame(data={
            #     "General Topic":zero_shot_topics_gen_topics_list,
            #     "Subtopic":zero_shot_topics_subtopics_list,
            #     "Description": zero_shot_topics_description_list
            #     })
            # unique_topics_markdown = topic_summary_df.to_markdown()

            # #print("unique_topics_markdown:", unique_topics_markdown)
            
            # formatted_general_topics_system_prompt = create_general_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

            # # Format the general_topics prompt with the topics
            # formatted_general_topics_prompt = create_general_topics_prompt.format(topics=unique_topics_markdown)

            # if "gemma" in model_choice:
            #     formatted_general_topics_prompt = llama_cpp_prefix + formatted_general_topics_system_prompt + "\n" + formatted_general_topics_prompt + llama_cpp_suffix

            # formatted_general_topics_prompt_list = [formatted_general_topics_prompt]

            # whole_conversation = []

            # general_topic_response, general_topic_conversation_history, general_topic_conversation, general_topic_conversation_metadata, response_text = call_llm_with_markdown_table_checks(batch_prompts, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, MAX_OUTPUT_VALIDATION_ATTEMPTS, master = True)

            # # Convert response text to a markdown table
            # try:
            #     zero_shot_topics_df, is_error = convert_response_text_to_markdown_table(response_text, table_type = "Revised topics table")
            #     print("Output revised zero shot topics table is:", zero_shot_topics_df)

            #     zero_shot_revised_path = output_folder + "zero_shot_topics_with_general_topics.csv"
            #     out_file_paths.append(zero_shot_revised_path)

            # except Exception as e:
            #     print("Error in parsing markdown table from response text:", e, "Not adding revised General Topics to table")

            # if zero_shot_topics_df.empty:
            #     print("Creation of revised general topics df failed, reverting to original list")
        else:
            pass
        
        # Add description or not
        zero_shot_topics_df = pd.DataFrame(data={
                "General Topic":zero_shot_topics_gen_topics_list,
                "Subtopic":zero_shot_topics_subtopics_list,
                "Description": zero_shot_topics_description_list
                })
        
        #if not zero_shot_topics_df["Description"].isnull().all():
        #    zero_shot_topics_df["Description"] = zero_shot_topics_df["Description"].apply(initial_clean)
        
        return zero_shot_topics_df

@spaces.GPU
def extract_topics(in_data_file,
              file_data:pd.DataFrame,
              existing_topics_table:pd.DataFrame,
              existing_reference_df:pd.DataFrame,
              existing_topic_summary_df:pd.DataFrame,
              unique_table_df_display_table_markdown:str,
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
              sentiment_checkbox:str = "Negative, Neutral, or Positive",
              force_zero_shot_radio:str = "No",
              in_excel_sheets:List[str] = [],
              force_single_topic_radio:str = "No",
              output_folder:str=OUTPUT_FOLDER,
              force_single_topic_prompt:str=force_single_topic_prompt,
              max_tokens:int=max_tokens,
              model_name_map:dict=model_name_map,              
              max_time_for_loop:int=max_time_for_loop,
              CHOSEN_LOCAL_MODEL_TYPE:str=CHOSEN_LOCAL_MODEL_TYPE,
              progress=Progress(track_tqdm=True)):

    '''
    Query an LLM (local, (Gemma 2B Instruct, Gemini or Anthropic-based on AWS) with up to three prompts about a table of open text data. Up to 'batch_size' rows will be queried at a time.

    Parameters:
    - in_data_file (gr.File): Gradio file object containing input data
    - file_data (pd.DataFrame): Pandas dataframe containing the consultation response data.
    - existing_topics_table (pd.DataFrame): Pandas dataframe containing the latest master topic table that has been iterated through batches.
    - existing_reference_df (pd.DataFrame): Pandas dataframe containing the list of Response reference numbers alongside the derived topics and subtopics.
    - existing_topic_summary_df (pd.DataFrame): Pandas dataframe containing the unique list of topics, subtopics, sentiment and summaries until this point.
    - unique_table_df_display_table_markdown (str): Table for display in markdown format.
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
    - sentiment_checkbox (str, optional): What type of sentiment analysis should the topic modeller do?
    - force_zero_shot_radio (str, optional): Should responses be forced into a zero shot topic or not.
    - in_excel_sheets (List[str], optional): List of excel sheets to load from input file.
    - force_single_topic_radio (str, optional): Should the model be forced to assign only one single topic to each response (effectively a classifier).
    - output_folder (str, optional): Output folder where results will be stored.
    - force_single_topic_prompt (str, optional): The prompt for forcing the model to assign only one single topic to each response.
    - max_tokens (int): The maximum number of tokens for the model.
    - model_name_map (dict, optional): A dictionary mapping full model name to shortened.
    - max_time_for_loop (int, optional): The number of seconds maximum that the function should run for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there).
    - CHOSEN_LOCAL_MODEL_TYPE (str, optional): The name of the chosen local model.
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

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            in_colnames_drop, in_excel_sheets, file_name = put_columns_in_df(in_data_file)
            file_data, file_name, num_batches = load_in_data_file(in_data_file, chosen_cols, batch_size_default, in_excel_sheets)
        except:
            # Check if files and text exist
            out_message = "Please enter a data file to summarise."
            print(out_message)
            raise Exception(out_message)


    #model_choice_clean = replace_punctuation_with_underscore(model_choice)
    model_choice_clean = model_name_map[model_choice]    

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("This is the first time through the loop, resetting latest_batch_completed to 0")
        if (latest_batch_completed == 999) | (latest_batch_completed == 0):
            latest_batch_completed = 0
            out_message = []
            out_file_paths = []

            if (model_choice == CHOSEN_LOCAL_MODEL_TYPE) & (RUN_LOCAL_MODEL == "1"):
                progress(0.1, f"Loading in local model: {CHOSEN_LOCAL_MODEL_TYPE}")
                local_model, tokenizer = load_model(local_model_type=CHOSEN_LOCAL_MODEL_TYPE, repo_id=LOCAL_REPO_ID, model_filename=LOCAL_MODEL_FILE, model_dir=LOCAL_MODEL_FOLDER)

    if num_batches > 0:
        progress_measure = round(latest_batch_completed / num_batches, 1)
        progress(progress_measure, desc="Querying large language model")
    else:
        progress(0.1, desc="Querying large language model")

    if latest_batch_completed < num_batches:

        # Load file
        # If out message or out_file_paths are blank, change to a list so it can be appended to
        if isinstance(out_message, str):
            out_message = [out_message]

        if not out_file_paths:
            out_file_paths = []
    
        
        if "anthropic.claude-3-sonnet" in model_choice and file_data.shape[1] > 300:
            out_message = "Your data has more than 300 rows, using the Sonnet model will be too expensive. Please choose the Haiku model instead."
            print(out_message)
            raise Exception(out_message)    
        
        if sentiment_checkbox == "Negative, Neutral, or Positive": sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative, Neutral, or Positive"
        elif sentiment_checkbox == "Negative or Positive": sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative or Positive"
        elif sentiment_checkbox == "Do not assess sentiment": sentiment_prompt = "Create a third column containing only the text 'Not assessed'"
        else: sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative, Neutral, or Positive"
        
        topics_loop_description = "Extracting topics from response batches (each batch of " + str(batch_size) + " responses)."
        topics_loop = tqdm(range(latest_batch_completed, num_batches), desc = topics_loop_description, unit="batches remaining")

        for i in topics_loop:       
            #for latest_batch_completed in range(num_batches):
            reported_batch_no = latest_batch_completed + 1  
            print("Running query batch", str(reported_batch_no))

            # Call the function to prepare the input table
            simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, batch_basic_response_df = data_file_to_markdown_table(file_data, file_name, chosen_cols, latest_batch_completed, batch_size)
            #log_files_output_paths.append(simplified_csv_table_path)

            # Conversation history
            conversation_history = []

            #print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

            # If the latest batch of responses contains at least one instance of text
            if not batch_basic_response_df.empty:

                # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
                if latest_batch_completed >= 1 or candidate_topics is not None:

                    # Prepare Gemini models before query       
                    if "gemini" in model_choice:
                        print("Using Gemini model:", model_choice)
                        model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=add_existing_topics_system_prompt, max_tokens=max_tokens)
                    elif "anthropic.claude" in model_choice:
                        print("Using AWS Bedrock model:", model_choice)
                    else:
                        print("Using local model:", model_choice)

                    # Preparing candidate topics if no topics currently exist
                    if candidate_topics and existing_topic_summary_df.empty:
                        progress(0.1, "Creating revised zero shot topics table")

                        # 'Zero shot topics' are those supplied by the user
                        max_topic_no = 120
                        zero_shot_topics = read_file(candidate_topics.name)                       
                            
                        zero_shot_topics_df = generate_zero_shot_topics_df(zero_shot_topics, force_zero_shot_radio, create_revised_general_topics, max_topic_no)

                        #print("zero_shot_topics_df:", zero_shot_topics_df)

                        # This part concatenates all zero shot and new topics together, so that for the next prompt the LLM will have the full list available
                        if not existing_topic_summary_df.empty and force_zero_shot_radio != "Yes":
                            existing_topic_summary_df = pd.concat([existing_topic_summary_df, zero_shot_topics_df]).drop_duplicates("Subtopic")
                        else:
                            existing_topic_summary_df = zero_shot_topics_df

                    if candidate_topics and not zero_shot_topics_df.empty:
                        # If you have already created revised zero shot topics, concat to the current
                        existing_topic_summary_df = pd.concat([existing_topic_summary_df, zero_shot_topics_df])

                    #all_topic_tables_df_merged = existing_topic_summary_df
                    existing_topic_summary_df["Number of responses"] = ""
                    existing_topic_summary_df.fillna("", inplace=True)
                    existing_topic_summary_df["General Topic"] = existing_topic_summary_df["General Topic"].str.replace('(?i)^Nan$', '', regex=True)
                    existing_topic_summary_df["Subtopic"] = existing_topic_summary_df["Subtopic"].str.replace('(?i)^Nan$', '', regex=True)
                    existing_topic_summary_df = existing_topic_summary_df.drop_duplicates()
                    if "Description" in existing_topic_summary_df:
                        if existing_topic_summary_df['Description'].isnull().all():
                            existing_topic_summary_df.drop("Description", axis = 1, inplace = True)

                    # print("existing_topic_summary_df:", existing_topic_summary_df)

                    # If user has chosen to try to force zero shot topics, then the prompt is changed to ask the model not to deviate at all from submitted topic list.
                    keep_cols = [
                        col for col in ["General Topic", "Subtopic", "Description"]
                        if col in existing_topic_summary_df.columns
                        and not existing_topic_summary_df[col].replace(r'^\s*$', pd.NA, regex=True).isna().all()
                        ]
                    
                    if force_zero_shot_radio == "Yes":                        
                        topics_df_for_markdown = existing_topic_summary_df[keep_cols].drop_duplicates(keep_cols)
                        unique_topics_markdown = topics_df_for_markdown.to_markdown(index=False)
                        topic_assignment_prompt = force_existing_topics_prompt
                    else:
                        topics_df_for_markdown = existing_topic_summary_df[keep_cols].drop_duplicates(keep_cols)
                        unique_topics_markdown = topics_df_for_markdown.to_markdown(index=False)
                        topic_assignment_prompt = allow_new_topics_prompt  

                    # Should the outputs force only one single topic assignment per response?
                    if force_single_topic_radio != "Yes": force_single_topic_prompt = ""
                    else:
                        topic_assignment_prompt = topic_assignment_prompt.replace("Assign topics", "Assign a topic").replace("assign Subtopics", "assign a Subtopic").replace("Subtopics", "Subtopic").replace("Topics", "Topic").replace("topics", "a topic")         

                    # Format the summary prompt with the response table and topics
                    formatted_system_prompt = add_existing_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)
                    formatted_summary_prompt = add_existing_topics_prompt.format(response_table=normalised_simple_markdown_table, topics=unique_topics_markdown, topic_assignment=topic_assignment_prompt, force_single_topic=force_single_topic_prompt, sentiment_choices=sentiment_prompt)
                    

                    if "gemma" in model_choice:
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

                    if "gemma" in model_choice:
                        summary_prompt_list = [full_prompt] # Includes system prompt
                    else:
                        summary_prompt_list = [formatted_summary_prompt]

                    # print("master_summary_prompt_list:", summary_prompt_list[0])

                    conversation_history = []
                    whole_conversation = []

                    # Process requests to large language model
                    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(summary_prompt_list, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, MAX_OUTPUT_VALIDATION_ATTEMPTS, master = True)

                    # Return output tables
                    topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, new_topic_df, new_markdown_table, new_reference_df, new_topic_summary_df, master_batch_out_file_part, is_error =  write_llm_output_and_logs(responses, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_topic_summary_df, batch_size, chosen_cols, first_run=False, output_folder=output_folder)

                    # Write final output to text file for logging purposes
                    try:
                        final_table_output_path = output_folder + master_batch_out_file_part + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                        if isinstance(responses[-1], ResponseObject):
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                        elif "choices" in responses[-1]:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1]["choices"][0]['text'])
                        else:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)

                    except Exception as e:
                        print("Error in returning model response:", e)
                    

                    # If error in table parsing, leave function
                    if is_error == True:
                        final_message_out = "Could not complete summary, error in LLM output."
                        raise Exception(final_message_out)

                    # Write outputs to csv
                    ## Topics with references
                    new_topic_df.to_csv(topic_table_out_path, index=None)
                    log_files_output_paths.append(topic_table_out_path)

                    ## Reference table mapping response numbers to topics
                    new_reference_df.to_csv(reference_table_out_path, index=None)
                    out_file_paths.append(reference_table_out_path)

                    ## Unique topic list
                    new_topic_summary_df = pd.concat([new_topic_summary_df, existing_topic_summary_df]).drop_duplicates('Subtopic')

                    new_topic_summary_df.to_csv(topic_summary_df_out_path, index=None)
                    out_file_paths.append(topic_summary_df_out_path)
                    
                    # Outputs for markdown table output
                    unique_table_df_display_table = new_topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
                    unique_table_df_display_table_markdown = unique_table_df_display_table[["General Topic", "Subtopic", "Sentiment", "Number of responses", "Summary"]].to_markdown(index=False)

                    #whole_conversation_metadata.append(whole_conversation_metadata_str)
                    whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)
                    

                    #out_file_paths = [col for col in out_file_paths if latest_batch_number_string in col]
                    #log_files_output_paths = [col for col in log_files_output_paths if latest_batch_number_string in col]

                    out_file_paths = [col for col in out_file_paths if str(reported_batch_no) in col]
                    log_files_output_paths = [col for col in out_file_paths if str(reported_batch_no) in col]

                    #print("out_file_paths at end of loop:", out_file_paths)

                # If this is the first batch, run this
                else:
                    #system_prompt = system_prompt + normalised_simple_markdown_table

                    # Prepare Gemini models before query       
                    if "gemini" in model_choice:
                        print("Using Gemini model:", model_choice)
                        model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
                    elif model_choice == CHOSEN_LOCAL_MODEL_TYPE:
                        print("Using local model:", model_choice)
                    else:
                        print("Using AWS Bedrock model:", model_choice)

                    formatted_initial_table_system_prompt = system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                    formatted_initial_table_prompt = initial_table_prompt.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)

                    if prompt2: formatted_prompt2 = prompt2.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)
                    else: formatted_prompt2 = prompt2
                    
                    if prompt3: formatted_prompt3 = prompt3.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)
                    else: formatted_prompt3 = prompt3

                    if "gemma" in model_choice:
                        formatted_initial_table_prompt = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_initial_table_prompt + llama_cpp_suffix
                        formatted_prompt2 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt2 + llama_cpp_suffix
                        formatted_prompt3 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt3 + llama_cpp_suffix

                    batch_prompts = [formatted_initial_table_prompt, formatted_prompt2, formatted_prompt3][:number_of_prompts_used]  # Adjust this list to send fewer requests 
                    
                    whole_conversation = [formatted_initial_table_system_prompt] 


                    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(batch_prompts, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, MAX_OUTPUT_VALIDATION_ATTEMPTS)


                    topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_table_df, markdown_table, reference_df, new_topic_summary_df, batch_file_path_details, is_error =  write_llm_output_and_logs(responses, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_topic_summary_df, batch_size, chosen_cols, first_run=True, output_folder=output_folder)

                    # If error in table parsing, leave function
                    if is_error == True:
                        raise Exception("Error in output table parsing")
                        # unique_table_df_display_table_markdown, new_topic_df, new_topic_summary_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths#, final_message_out
                    
                    
                    #all_topic_tables_df.append(topic_table_df)

                    topic_table_df.to_csv(topic_table_out_path, index=None)
                    out_file_paths.append(topic_table_out_path)

                    reference_df.to_csv(reference_table_out_path, index=None)
                    out_file_paths.append(reference_table_out_path)

                    ## Unique topic list

                    new_topic_summary_df = pd.concat([new_topic_summary_df, existing_topic_summary_df]).drop_duplicates('Subtopic')

                    new_topic_summary_df.to_csv(topic_summary_df_out_path, index=None)
                    out_file_paths.append(topic_summary_df_out_path)
                    
                    #all_markdown_topic_tables.append(markdown_table)

                    whole_conversation_metadata.append(whole_conversation_metadata_str)
                    whole_conversation_metadata_str = '. '.join(whole_conversation_metadata)
                    
                    # Write final output to text file also
                    try:
                        final_table_output_path = output_folder + batch_file_path_details + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                        if isinstance(responses[-1], ResponseObject):
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                            unique_table_df_display_table_markdown = responses[-1].text
                        elif "choices" in responses[-1]:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1]["choices"][0]['text'])
                            unique_table_df_display_table_markdown =responses[-1]["choices"][0]['text']
                        else:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                            unique_table_df_display_table_markdown = responses[-1].text

                        log_files_output_paths.append(final_table_output_path)

                    except Exception as e:
                        print("Error in returning model response:", e)
                    
                    new_topic_df = topic_table_df
                    new_reference_df = reference_df

            else:
                print("Current batch of responses contains no text, moving onto next. Batch number:", str(latest_batch_completed + 1), ". Start row:", start_row, ". End row:", end_row)

            # Increase latest file completed count unless we are over the last batch number
            if latest_batch_completed <= num_batches:
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
            existing_topic_summary_df = new_topic_summary_df.dropna(how='all')
            existing_topics_table = new_topic_df.dropna(how='all')

            # The topic table that can be modified does not need the summary column
            modifiable_topic_summary_df = existing_topic_summary_df.drop("Summary", axis=1)

        out_time = f"{final_time:0.1f} seconds."
        
        out_message.append('All queries successfully completed in')

        final_message_out = '\n'.join(out_message)
        final_message_out = final_message_out + " " + out_time  

        print(final_message_out)

    # If we have extracted topics from the last batch, return the input out_message and file list to the relevant components
    if latest_batch_completed >= num_batches:
        print("Last batch reached, returning batch:", str(latest_batch_completed))
        # Set to a very high number so as not to mess with subsequent file processing by the user
        #latest_batch_completed = 999

        join_file_paths = []

        toc = time.perf_counter()
        final_time = (toc - tic) + time_taken
        out_time = f"Everything finished in {round(final_time,1)} seconds."
        print(out_time)

        print("All summaries completed. Creating outputs.")

        model_choice_clean = model_name_map[model_choice]   
        # Example usage
        in_column_cleaned = clean_column_name(chosen_cols, max_length=20)

        # Need to reduce output file names as full length files may be too long
        file_name = clean_column_name(file_name, max_length=30)    

        # Save outputs for each batch. If master file created, label file as master
        file_path_details = f"{file_name}_col_{in_column_cleaned}"

        # Create a pivoted reference table
        existing_reference_df_pivot = convert_reference_table_to_pivot_table(existing_reference_df)

        # Save the new DataFrame to CSV
        #topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        reference_table_out_pivot_path = output_folder + file_path_details + "_final_reference_table_pivot_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        reference_table_out_path = output_folder + file_path_details + "_final_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv" 
        topic_summary_df_out_path = output_folder + file_path_details + "_final_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        basic_response_data_out_path = output_folder + file_path_details + "_simplified_data_file_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

        ## Reference table mapping response numbers to topics
        existing_reference_df.to_csv(reference_table_out_path, index=None)
        out_file_paths.append(reference_table_out_path)
        join_file_paths.append(reference_table_out_path)

        # Create final unique topics table from reference table to ensure consistent numbers
        final_out_topic_summary_df = create_topic_summary_df_from_reference_table(existing_reference_df)

        ## Unique topic list
        final_out_topic_summary_df.to_csv(topic_summary_df_out_path, index=None, encoding='utf-8')
        out_file_paths.append(topic_summary_df_out_path)

        # Ensure that we are only returning the final results to outputs
        out_file_paths = [x for x in out_file_paths if '_final_' in x]

        ## Reference table mapping response numbers to topics
        existing_reference_df_pivot.to_csv(reference_table_out_pivot_path, index = None, encoding='utf-8')
        log_files_output_paths.append(reference_table_out_pivot_path)

        ## Create a dataframe for missing response references:
        # Assuming existing_reference_df and file_data are already defined
        # Simplify table to just responses column and the Response reference number        

        basic_response_data = get_basic_response_data(file_data, chosen_cols)


        # Save simplified file data to log outputs
        pd.DataFrame(basic_response_data).to_csv(basic_response_data_out_path, index=None, encoding='utf-8')
        log_files_output_paths.append(basic_response_data_out_path)


        # Step 1: Identify missing references
        missing_references = basic_response_data[~basic_response_data['Reference'].astype(str).isin(existing_reference_df['Response References'].astype(str).unique())]

        # Step 2: Create a new DataFrame with the same columns as existing_reference_df
        missing_df = pd.DataFrame(columns=existing_reference_df.columns)

        # Step 3: Populate the new DataFrame
        missing_df['Response References'] = missing_references['Reference']
        missing_df = missing_df.fillna(np.nan) #.infer_objects(copy=False)  # Fill other columns with NA

        # Display the new DataFrame
        #print("missing_df:", missing_df)

        missing_df_out_path = output_folder + file_path_details + "_missing_references_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        missing_df.to_csv(missing_df_out_path, index=None, encoding='utf-8')
        log_files_output_paths.append(missing_df_out_path)

        out_file_paths = list(set(out_file_paths))
        log_files_output_paths = list(set(log_files_output_paths))        

        final_out_file_paths = [file_path for file_path in out_file_paths if "final_" in file_path]
 
        # The topic table that can be modified does not need the summary column
        modifiable_topic_summary_df = final_out_topic_summary_df.drop("Summary", axis=1)

        print("latest_batch_completed at end of batch iterations to return is", latest_batch_completed)

        return unique_table_df_display_table_markdown, existing_topics_table, final_out_topic_summary_df, existing_reference_df, final_out_file_paths, final_out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_out_file_paths, final_out_file_paths, gr.Dataframe(value=modifiable_topic_summary_df, headers=None, col_count=(modifiable_topic_summary_df.shape[1], "fixed"), row_count = (modifiable_topic_summary_df.shape[0], "fixed"), visible=True, type="pandas"), final_out_file_paths, join_file_paths


    return unique_table_df_display_table_markdown, existing_topics_table, existing_topic_summary_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths, gr.Dataframe(value=modifiable_topic_summary_df, headers=None, col_count=(modifiable_topic_summary_df.shape[1], "fixed"), row_count = (modifiable_topic_summary_df.shape[0], "fixed"), visible=True, type="pandas"), out_file_paths, join_file_paths

def join_modified_topic_names_to_ref_table(modified_topic_summary_df:pd.DataFrame, original_topic_summary_df:pd.DataFrame, reference_df:pd.DataFrame):
    '''
    Take a unique topic table that has been modified by the user, and apply the topic name changes to the long-form reference table.
    '''

    # Drop rows where Number of responses is either NA or null
    modified_topic_summary_df = modified_topic_summary_df[~modified_topic_summary_df["Number of responses"].isnull()]
    modified_topic_summary_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment", "Topic_number"], inplace=True)

    # First, join the modified topics to the original topics dataframe based on index to have the modified names alongside the original names
    original_topic_summary_df_m = original_topic_summary_df.merge(modified_topic_summary_df[["General Topic", "Subtopic", "Sentiment", "Topic_number"]], on="Topic_number", how="left", suffixes=("", "_mod"))

    original_topic_summary_df_m.drop_duplicates(["General Topic", "Subtopic", "Sentiment", "Topic_number"], inplace=True)


    # Then, join these new topic names onto the reference_df, merge based on the original names
    modified_reference_df = reference_df.merge(original_topic_summary_df_m[["Topic_number", "General Topic_mod", "Subtopic_mod", "Sentiment_mod"]], on=["Topic_number"], how="left")

    
    modified_reference_df.drop(["General Topic", "Subtopic", "Sentiment"], axis=1, inplace=True, errors="ignore")
    
    modified_reference_df.rename(columns={"General Topic_mod":"General Topic",
                                                                 "Subtopic_mod":"Subtopic",
                                                                 "Sentiment_mod":"Sentiment"}, inplace=True)

    modified_reference_df.drop(["General Topic_mod", "Subtopic_mod", "Sentiment_mod"], inplace=True, errors="ignore")  
    

    #modified_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    modified_reference_df.sort_values(["Start row of group", "Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    modified_reference_df = modified_reference_df.loc[:, ["Response References", "General Topic", "Subtopic", "Sentiment", "Summary", "Start row of group", "Topic_number"]]

    # Drop rows where Response References is either NA or null
    modified_reference_df = modified_reference_df[~modified_reference_df["Response References"].isnull()]

    return modified_reference_df

# MODIFY EXISTING TABLE
def modify_existing_output_tables(original_topic_summary_df:pd.DataFrame, modifiable_topic_summary_df:pd.DataFrame, reference_df:pd.DataFrame, text_output_file_list_state:List[str], output_folder:str=OUTPUT_FOLDER) -> Tuple:
    '''
    Take a unique_topics table that has been modified, apply these new topic names to the long-form reference_df, and save both tables to file.
    '''

    # Ensure text_output_file_list_state is a flat list
    if any(isinstance(i, list) for i in text_output_file_list_state):
        text_output_file_list_state = [item for sublist in text_output_file_list_state for item in sublist]  # Flatten list

    # Extract file paths safely
    reference_files = [x for x in text_output_file_list_state if 'reference' in x]
    unique_files = [x for x in text_output_file_list_state if 'unique' in x]

    # Ensure files exist before accessing
    reference_file_path = os.path.basename(reference_files[0]) if reference_files else None
    unique_table_file_path = os.path.basename(unique_files[0]) if unique_files else None

    output_file_list = []

    if reference_file_path and unique_table_file_path:

        reference_df = join_modified_topic_names_to_ref_table(modifiable_topic_summary_df, original_topic_summary_df, reference_df)

        ## Reference table mapping response numbers to topics
        reference_table_file_name = reference_file_path.replace(".csv", "_mod") 
        new_reference_df_file_path = output_folder + reference_table_file_name  + ".csv"
        reference_df.to_csv(new_reference_df_file_path, index=None, encoding='utf-8')
        output_file_list.append(new_reference_df_file_path)

        # Drop rows where Number of responses is NA or null
        modifiable_topic_summary_df = modifiable_topic_summary_df[~modifiable_topic_summary_df["Number of responses"].isnull()]

        # Convert 'Number of responses' to numeric (forcing errors to NaN if conversion fails)
        modifiable_topic_summary_df["Number of responses"] = pd.to_numeric(
            modifiable_topic_summary_df["Number of responses"], errors='coerce'
        )

        # Drop any rows where conversion failed (original non-numeric values)
        modifiable_topic_summary_df.dropna(subset=["Number of responses"], inplace=True)

        # Sort values
        modifiable_topic_summary_df.sort_values(["Number of responses"], ascending=False, inplace=True)

        unique_table_file_name = unique_table_file_path.replace(".csv", "_mod")
        modified_unique_table_file_path = output_folder + unique_table_file_name + ".csv"
        modifiable_topic_summary_df.to_csv(modified_unique_table_file_path, index=None, encoding='utf-8')
        output_file_list.append(modified_unique_table_file_path)
    
    else:
        output_file_list = text_output_file_list_state
        reference_table_file_name = reference_file_path
        unique_table_file_name = unique_table_file_path
        raise Exception("Reference and unique topic tables not found.")
    
    # Outputs for markdown table output
    unique_table_df_revised_display = modifiable_topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
    deduplicated_unique_table_markdown = unique_table_df_revised_display.to_markdown(index=False)
    

    return modifiable_topic_summary_df, reference_df, output_file_list, output_file_list, output_file_list, output_file_list, reference_table_file_name, unique_table_file_name, deduplicated_unique_table_markdown
