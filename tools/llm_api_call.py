import os
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
from typing import List, Tuple, Any
from io import StringIO
GradioFileData = gr.FileData

from tools.prompts import initial_table_prompt, prompt2, prompt3, initial_table_system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt,  force_existing_topics_prompt, allow_new_topics_prompt, force_single_topic_prompt, add_existing_topics_assistant_prefill, initial_table_assistant_prefill, structured_summary_prompt
from tools.helper_functions import read_file, put_columns_in_df, wrap_text, initial_clean, load_in_data_file, load_in_file, create_topic_summary_df_from_reference_table, convert_reference_table_to_pivot_table, get_basic_response_data, clean_column_name, load_in_previous_data_files, create_batch_file_path_details
from tools.llm_funcs import ResponseObject, construct_gemini_generative_model, call_llm_with_markdown_table_checks, create_missing_references_df, calculate_tokens_from_metadata, construct_azure_client
from tools.config import RUN_LOCAL_MODEL, AWS_REGION, MAX_COMMENT_CHARS, MAX_OUTPUT_VALIDATION_ATTEMPTS, MAX_TOKENS, TIMEOUT_WAIT, NUMBER_OF_RETRY_ATTEMPTS, MAX_TIME_FOR_LOOP, BATCH_SIZE_DEFAULT, DEDUPLICATION_THRESHOLD, model_name_map, OUTPUT_FOLDER, CHOSEN_LOCAL_MODEL_TYPE, LOCAL_REPO_ID, LOCAL_MODEL_FILE, LOCAL_MODEL_FOLDER, LLM_SEED, MAX_GROUPS, REASONING_SUFFIX, AZURE_INFERENCE_ENDPOINT
from tools.aws_functions import connect_to_bedrock_runtime

if RUN_LOCAL_MODEL == "1":
    from tools.llm_funcs import load_model

max_tokens = MAX_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS
random_seed = LLM_SEED
reasoning_suffix = REASONING_SUFFIX

### HELPER FUNCTIONS

def normalise_string(text:str):
    # Replace two or more dashes with a single dash
    text = re.sub(r'-{2,}', '-', text)
    
    # Replace two or more spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text


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
    start_row = (batch_number * batch_size)
    if start_row > file_len + 1:
        print("Start row greater than file row length")
        return simplified_csv_table_path, normalised_simple_markdown_table, file_name
    if start_row < 0:
        raise Exception("Start row is below 0")

    if ((start_row + batch_size) - 1) <= file_len + 1:
        end_row = ((start_row + batch_size) - 1)
    else:
        end_row = file_len + 1

    batch_basic_response_data = basic_response_data.loc[start_row:end_row, ["Reference", "Response", "Original Reference"]]  # Select the current batch

    # Now replace the reference numbers with numbers starting from 1
    batch_basic_response_data.loc[:, "Reference"] = batch_basic_response_data["Reference"] - start_row

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

    simple_markdown_table = batch_basic_response_data[["Reference", "Response"]].to_markdown(index=None)

    normalised_simple_markdown_table = normalise_string(simple_markdown_table)

    #print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

    return simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, batch_basic_response_data

def replace_punctuation_with_underscore(input_string:str):
    # Create a translation table where each punctuation character maps to '_'
    translation_table = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    
    # Translate the input string using the translation table
    return input_string.translate(translation_table)

### INITIAL TOPIC MODEL DEVELOPMENT FUNCTIONS

def clean_markdown_table(text: str):
    # Split text into lines
    lines = text.splitlines()
    
    # Step 1: Identify table structure and process line continuations
    table_rows = list()
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
    formatted_rows = list()
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

# Convert output table to markdown and then to a pandas dataframe to csv
def remove_before_last_term(input_string: str) -> str:
    # Use regex to find the last occurrence of the term
    match = re.search(r'(\| ?General topic)', input_string)
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

        print("clean_md_text:", clean_md_text)
        
        # Read Markdown table into a DataFrame
        df = pd.read_csv(pd.io.common.StringIO(clean_md_text), sep="|", skipinitialspace=True,
            dtype={'Response References': str})
        
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
                    <th>General topic</th>
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
                    <th>General topic</th>
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

def convert_response_text_to_dataframe(response_text:str, table_type:str = "Main table"):
    is_error = False
    start_of_table_response = remove_before_last_term(response_text)

    cleaned_response = clean_markdown_table(start_of_table_response)

    # Add a space after commas between numbers (e.g., "1,2" -> "1, 2")
    cleaned_response = re.sub(r'(\d),(\d)', r'\1, \2', cleaned_response)    

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

def write_llm_output_and_logs(response_text: str,
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
                              batch_basic_response_df:pd.DataFrame,
                              model_name_map:dict,
                              group_name:str = "All",
                              produce_structures_summary_radio:str = "No",                      
                              first_run: bool = False,
                              output_folder:str=OUTPUT_FOLDER) -> Tuple:
    """
    Writes the output of the large language model requests and logs to files.

    Parameters:
    - response_text (str): The text of the response from the model.
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
    - existing_topics_df (pd.DataFrame): The existing unique topics dataframe.
    - batch_size_number (int): The size of batches in terms of number of responses.
    - in_column (str): The name of the open text column that is being analysed.
    - batch_basic_response_df (pd.DataFrame): The dataframe that contains the response data.
    - model_name_map (dict): The dictionary that maps the model choice to the model name.
    - group_name (str, optional): The name of the current group.
    - produce_structures_summary_radio (str, optional): Whether the option to produce structured summaries has been selected.
    - first_run (bool): A boolean indicating if this is the first run through this function in this process. Defaults to False.
    - output_folder (str): The name of the folder where output files are saved.
    """
    topic_summary_df_out_path = list()
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    topic_summary_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame()
    out_reference_df = pd.DataFrame()
    out_topic_summary_df = pd.DataFrame()  
    is_error = False # If there was an error in parsing, return boolean saying error
    # Convert conversation to string and add to log outputs
    whole_conversation_str = '\n'.join(whole_conversation)
    whole_conversation_metadata_str = '\n'.join(whole_conversation_metadata)
    start_row_reported = start_row + 1

    batch_file_path_details = create_batch_file_path_details(file_name)

    # Need to reduce output file names as full length files may be too long
    model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)
    # in_column_cleaned = clean_column_name(in_column, max_length=20)    
    # file_name_clean = clean_column_name(file_name, max_length=20, front_characters=True)


    # # Save outputs for each batch. If master file created, label file as master
    # batch_file_path_details = f"{file_name_clean}_batch_{latest_batch_completed + 1}_size_{batch_size_number}_col_{in_column_cleaned}"
    row_number_string_start = f"Rows {start_row_reported} to {end_row + 1}: "

    whole_conversation_path = output_folder + batch_file_path_details + "_full_conversation_" + model_choice_clean_short + ".txt"

    whole_conversation_path_meta = output_folder + batch_file_path_details + "_metadata_" + model_choice_clean_short + ".txt"

    with open(whole_conversation_path, "w", encoding='utf-8-sig', errors='replace') as f: f.write(whole_conversation_str)

    with open(whole_conversation_path_meta, "w", encoding='utf-8-sig', errors='replace') as f: f.write(whole_conversation_metadata_str)

    log_files_output_paths.append(whole_conversation_path_meta)

    # Convert response text to a markdown table
    try:
        topic_with_response_df, is_error = convert_response_text_to_dataframe(response_text)
    except Exception as e:
        print("Error in parsing markdown table from response text:", e)
        return topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_with_response_df, out_reference_df, out_topic_summary_df, batch_file_path_details, is_error

    # Rename columns to ensure consistent use of data frames later in code
    new_column_names = {
    topic_with_response_df.columns[0]: "General topic",
    topic_with_response_df.columns[1]: "Subtopic",
    topic_with_response_df.columns[2]: "Sentiment",
    topic_with_response_df.columns[3]: "Response References",
    topic_with_response_df.columns[4]: "Summary"
    }

    topic_with_response_df = topic_with_response_df.rename(columns=new_column_names)

    # Fill in NA rows with values from above (topics seem to be included only on one row):
    topic_with_response_df = topic_with_response_df.ffill()

    # For instances where you end up with float values in Response References
    topic_with_response_df["Response References"] = topic_with_response_df["Response References"].astype(str).str.replace(".0", "", regex=False)

    # Strip and lower case topic names to remove issues where model is randomly capitalising topics/sentiment
    topic_with_response_df["General topic"] = topic_with_response_df["General topic"].astype(str).str.strip().str.lower().str.capitalize()
    topic_with_response_df["Subtopic"] = topic_with_response_df["Subtopic"].astype(str).str.strip().str.lower().str.capitalize()
    topic_with_response_df["Sentiment"] = topic_with_response_df["Sentiment"].astype(str).str.strip().str.lower().str.capitalize()
    
    topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean_short  + ".csv"

    # Table to map references to topics
    reference_data = list()

    batch_basic_response_df["Reference"] = batch_basic_response_df["Reference"].astype(str)

    # Iterate through each row in the original DataFrame
    for index, row in topic_with_response_df.iterrows():
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

        if produce_structures_summary_radio != "Yes": summary = row_number_string_start + summary

        # Create a new entry for each reference number
        for ref in references:
            # Add start_row back onto reference_number
            if batch_basic_response_df.empty:
                try:
                    response_ref_no =  str(int(ref) + int(start_row))
                except ValueError:
                    print("Reference is not a number")
                    continue
            else:
                try:                    
                    response_ref_no =  batch_basic_response_df.loc[batch_basic_response_df["Reference"]==str(ref), "Original Reference"].iloc[0]
                except ValueError:
                    print("Reference is not a number")
                    continue

            reference_data.append({
                'Response References': response_ref_no,
                'General topic': topic,
                'Subtopic': subtopic,
                'Sentiment': sentiment,
                'Summary': summary,
                "Start row of group": start_row_reported
            })

    # Create a new DataFrame from the reference data
    if reference_data:
        new_reference_df = pd.DataFrame(reference_data)
    else:
        new_reference_df = pd.DataFrame(columns=["Response References", "General topic", "Subtopic", "Sentiment", "Summary", "Start row of group"])
    
    # Append on old reference data
    if not new_reference_df.empty:
        out_reference_df = pd.concat([new_reference_df, existing_reference_df]).dropna(how='all')
    else:
        out_reference_df = existing_reference_df

    # Remove duplicate Response References for the same topic
    out_reference_df.drop_duplicates(["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)

    # Try converting response references column to int, keep as string if fails
    try:
        out_reference_df["Response References"] = out_reference_df["Response References"].astype(int)
    except Exception as e:
        print("Could not convert Response References column to integer due to", e)
        print("out_reference_df['Response References']:", out_reference_df["Response References"].head())

    out_reference_df.sort_values(["Start row of group", "Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)

    # Each topic should only be associated with each individual response once
    out_reference_df.drop_duplicates(["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)
    out_reference_df["Group"] = group_name

    # Save the new DataFrame to CSV
    reference_table_out_path = output_folder + batch_file_path_details + "_reference_table_" + model_choice_clean_short + ".csv"    

    # Table of all unique topics with descriptions
    new_topic_summary_df = topic_with_response_df[["General topic", "Subtopic", "Sentiment"]]

    new_topic_summary_df = new_topic_summary_df.rename(columns={new_topic_summary_df.columns[0]: "General topic", new_topic_summary_df.columns[1]: "Subtopic", new_topic_summary_df.columns[2]: "Sentiment"})
    
    # Join existing and new unique topics
    out_topic_summary_df = pd.concat([new_topic_summary_df, existing_topics_df]).dropna(how='all')

    out_topic_summary_df = out_topic_summary_df.rename(columns={out_topic_summary_df.columns[0]: "General topic", out_topic_summary_df.columns[1]: "Subtopic", out_topic_summary_df.columns[2]: "Sentiment"})

    #print("out_topic_summary_df:", out_topic_summary_df)

    out_topic_summary_df = out_topic_summary_df.drop_duplicates(["General topic", "Subtopic", "Sentiment"]).\
            drop(["Number of responses", "Summary"], axis = 1, errors="ignore") 

    # Get count of rows that refer to particular topics
    reference_counts = out_reference_df.groupby(["General topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': ' <br> '.join
    }).reset_index()

    # Join the counts to existing_topic_summary_df
    out_topic_summary_df = out_topic_summary_df.merge(reference_counts, how='left', on=["General topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    out_topic_summary_df = out_topic_summary_df.rename(columns={"Response References":"Number of responses"}, errors="ignore")

    out_topic_summary_df["Group"] = group_name

    topic_summary_df_out_path = output_folder + batch_file_path_details + "_unique_topics_" + model_choice_clean_short + ".csv"

    return topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_with_response_df, out_reference_df, out_topic_summary_df, batch_file_path_details, is_error

def generate_zero_shot_topics_df(zero_shot_topics:pd.DataFrame,
                                 force_zero_shot_radio:str="No",
                                 create_revised_general_topics:bool=False,
                                 max_topic_no:int=120):
    """
    Preprocesses a DataFrame of zero-shot topics, cleaning and formatting them
    for use with a large language model. It handles different column configurations
    (e.g., only subtopics, general topics and subtopics, or subtopics with descriptions)
    and enforces a maximum number of topics.

    Args:
        zero_shot_topics (pd.DataFrame): A DataFrame containing the initial zero-shot topics.
                                         Expected columns can vary, but typically include
                                         "General topic", "Subtopic", and/or "Description".
        force_zero_shot_radio (str, optional): A string indicating whether to force
                                               the use of zero-shot topics. Defaults to "No".
                                               (Currently not used in the function logic, but kept for signature consistency).
        create_revised_general_topics (bool, optional): A boolean indicating whether to
                                                        create revised general topics. Defaults to False.
                                                        (Currently not used in the function logic, but kept for signature consistency).
        max_topic_no (int, optional): The maximum number of topics allowed to fit within
                                      LLM context limits. If `zero_shot_topics` exceeds this,
                                      it will be truncated. Defaults to 120.

    Returns:
        tuple: A tuple containing:
            - zero_shot_topics_gen_topics_list (list): A list of cleaned general topics.
            - zero_shot_topics_subtopics_list (list): A list of cleaned subtopics.
            - zero_shot_topics_description_list (list): A list of cleaned topic descriptions.
    """

    zero_shot_topics_gen_topics_list = list()
    zero_shot_topics_subtopics_list = list()
    zero_shot_topics_description_list = list()

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

        # If number of columns is 1, keep only subtopics
        if zero_shot_topics.shape[1] == 1 and "General topic" not in zero_shot_topics.columns:
            print("Found only Subtopic in zero shot topics")
            zero_shot_topics_gen_topics_list = [""] * zero_shot_topics.shape[0]
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 0])                    
        # Allow for possibility that the user only wants to set general topics and not subtopics
        elif zero_shot_topics.shape[1] == 1 and "General topic" in zero_shot_topics.columns: 
            print("Found only General topic in zero shot topics")
            zero_shot_topics_gen_topics_list = list(zero_shot_topics["General topic"])
            zero_shot_topics_subtopics_list = [""] * zero_shot_topics.shape[0]
        # If general topic and subtopic are specified
        elif set(["General topic", "Subtopic"]).issubset(zero_shot_topics.columns):
            print("Found General topic and Subtopic in zero shot topics")
            zero_shot_topics_gen_topics_list = list(zero_shot_topics["General topic"])
            zero_shot_topics_subtopics_list = list(zero_shot_topics["Subtopic"])
        # If subtopic and description are specified
        elif set(["Subtopic", "Description"]).issubset(zero_shot_topics.columns):
            print("Found Subtopic and Description in zero shot topics")
            zero_shot_topics_gen_topics_list = [""] * zero_shot_topics.shape[0]
            zero_shot_topics_subtopics_list = list(zero_shot_topics["Subtopic"])
            zero_shot_topics_description_list = list(zero_shot_topics["Description"])

        # If number of columns is at least 2, keep general topics and subtopics
        elif zero_shot_topics.shape[1] >= 2 and "Description" not in zero_shot_topics.columns: 
            zero_shot_topics_gen_topics_list = list(zero_shot_topics.iloc[:, 0])
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 1])
        else:
            # If there are more columns, just assume that the first column was meant to be a subtopic
            zero_shot_topics_gen_topics_list = [""] * zero_shot_topics.shape[0]
            zero_shot_topics_subtopics_list = list(zero_shot_topics.iloc[:, 0])

        # Add a description if column is present 
        if not zero_shot_topics_description_list:
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
        
        # Add description or not
        zero_shot_topics_df = pd.DataFrame(data={
                "General topic":zero_shot_topics_gen_topics_list,
                "Subtopic":zero_shot_topics_subtopics_list,
                "Description": zero_shot_topics_description_list
                })
        
        return zero_shot_topics_df

@spaces.GPU(duration=300)
def extract_topics(in_data_file: GradioFileData,
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
              out_message:List= list(),
              out_file_paths:List = list(),
              log_files_output_paths:List = list(),
              first_loop_state:bool=False,
              whole_conversation_metadata_str:str="",
              initial_table_prompt:str=initial_table_prompt,
              prompt2:str=prompt2,
              prompt3:str=prompt3,
              initial_table_system_prompt:str=initial_table_system_prompt,
              add_existing_topics_system_prompt:str=add_existing_topics_system_prompt,
              add_existing_topics_prompt:str=add_existing_topics_prompt,
              number_of_prompts_used:int=1,
              batch_size:int=5,
              context_textbox:str="",
              time_taken:float = 0,
              sentiment_checkbox:str = "Negative, Neutral, or Positive",
              force_zero_shot_radio:str = "No",
              in_excel_sheets:List[str] = list(),
              force_single_topic_radio:str = "No",
              output_folder:str=OUTPUT_FOLDER,
              force_single_topic_prompt:str=force_single_topic_prompt,
              group_name:str="All",
              produce_structures_summary_radio:str="No",
              aws_access_key_textbox:str='',
              aws_secret_key_textbox:str='',
              hf_api_key_textbox:str='',
              azure_api_key_textbox:str='',
              max_tokens:int=max_tokens,
              model_name_map:dict=model_name_map,              
              max_time_for_loop:int=max_time_for_loop,
              CHOSEN_LOCAL_MODEL_TYPE:str=CHOSEN_LOCAL_MODEL_TYPE,
              reasoning_suffix:str=reasoning_suffix,
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
    - in_api_key (str): The API key for authentication (Google Gemini).
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
    - initial_table_system_prompt (str): The system prompt for the model.
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
    - produce_structures_summary_radio (str, optional): Should the model create a structured summary instead of extracting topics.
    - output_folder (str, optional): Output folder where results will be stored.
    - force_single_topic_prompt (str, optional): The prompt for forcing the model to assign only one single topic to each response.
    - aws_access_key_textbox (str, optional): AWS access key for account with Bedrock permissions.
    - aws_secret_key_textbox (str, optional): AWS secret key for account with Bedrock permissions.
    - hf_api_key_textbox (str, optional): Hugging Face API key for account with Hugging Face permissions.
    - max_tokens (int): The maximum number of tokens for the model.
    - model_name_map (dict, optional): A dictionary mapping full model name to shortened.
    - max_time_for_loop (int, optional): The number of seconds maximum that the function should run for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there).
    - CHOSEN_LOCAL_MODEL_TYPE (str, optional): The name of the chosen local model.
    - reasoning_suffix (str, optional): The suffix for the reasoning system prompt.
    - progress (Progress): A progress tracker.
    '''

    tic = time.perf_counter()
    google_client = list()
    google_config = {}
    final_time = 0.0
    whole_conversation_metadata = list()
    is_error = False
    create_revised_general_topics = False
    local_model = list()
    tokenizer = list()
    zero_shot_topics_df = pd.DataFrame()
    missing_df = pd.DataFrame()
    new_reference_df = pd.DataFrame(columns=["Response References",	"General topic",	"Subtopic",	"Sentiment",	"Start row of group",	"Group"	,"Topic_number",	"Summary"])
    new_topic_summary_df = pd.DataFrame(columns=["General topic","Subtopic","Sentiment","Group","Number of responses","Summary"])
    new_topic_df = pd.DataFrame()

    # Need to reduce output file names as full length files may be too long
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)
    in_column_cleaned = clean_column_name(chosen_cols, max_length=20)    
    file_name_clean = clean_column_name(file_name, max_length=20, front_characters=False)  

    # For Gemma models
    #llama_cpp_prefix = "<start_of_turn>user\n"
    #llama_cpp_suffix = "<end_of_turn>\n<start_of_turn>model\n"

    # For GPT OSS
    #llama_cpp_prefix = "<|start|>assistant<|channel|>analysis<|message|>\n"
    #llama_cpp_suffix = "<|start|>assistant<|channel|>final<|message|>"

    # Blank
    llama_cpp_prefix = ""
    llama_cpp_suffix = ""

    #print("output_folder:", output_folder)

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

    model_choice_clean = model_name_map[model_choice]['short_name']
    model_source = model_name_map[model_choice]["source"]

    bedrock_runtime = connect_to_bedrock_runtime(model_name_map, model_choice, aws_access_key_textbox, aws_secret_key_textbox)

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        if (latest_batch_completed == 999) | (latest_batch_completed == 0):
            latest_batch_completed = 0
            out_message = list()
            out_file_paths = list()
            final_time = 0

            if (model_source == "Local") & (RUN_LOCAL_MODEL == "1"):
                progress(0.1, f"Loading in local model: {CHOSEN_LOCAL_MODEL_TYPE}")
                local_model, tokenizer = load_model(local_model_type=CHOSEN_LOCAL_MODEL_TYPE, repo_id=LOCAL_REPO_ID, model_filename=LOCAL_MODEL_FILE, model_dir=LOCAL_MODEL_FOLDER, hf_token=hf_api_key_textbox)

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
            out_file_paths = list()
    
        
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
            reported_batch_no = latest_batch_completed + 1  
            print("Running response batch:", reported_batch_no)

            # Call the function to prepare the input table
            simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, batch_basic_response_df = data_file_to_markdown_table(file_data, file_name, chosen_cols, latest_batch_completed, batch_size)

            # Conversation history
            conversation_history = list()

            # If the latest batch of responses contains at least one instance of text
            if not batch_basic_response_df.empty:

                # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
                if latest_batch_completed >= 1 or candidate_topics is not None:

                    formatted_system_prompt = add_existing_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                    # Prepare clients before query       
                    if "Gemini" in model_source:
                        print("Using Gemini model:", model_choice)
                        google_client, google_config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=formatted_system_prompt, max_tokens=max_tokens)
                    elif "Azure" in model_source:
                        print("Using Azure AI Inference model:", model_choice)
                        # If provided, set env for downstream calls too
                        if azure_api_key_textbox:
                            os.environ["AZURE_INFERENCE_CREDENTIAL"] = azure_api_key_textbox
                        google_client, google_config = construct_azure_client(in_api_key=azure_api_key_textbox, endpoint=AZURE_INFERENCE_ENDPOINT)
                    elif "anthropic.claude" in model_choice:
                        print("Using AWS Bedrock model:", model_choice)
                    else:
                        print("Using local model:", model_choice)

                    # Preparing candidate topics if no topics currently exist
                    if candidate_topics and existing_topic_summary_df.empty:
                        #progress(0.1, "Creating revised zero shot topics table")

                        # 'Zero shot topics' are those supplied by the user
                        max_topic_no = 120
                        zero_shot_topics = read_file(candidate_topics.name)
                        zero_shot_topics = zero_shot_topics.fillna("")           # Replace NaN with empty string
                        zero_shot_topics = zero_shot_topics.astype(str)
                            
                        zero_shot_topics_df = generate_zero_shot_topics_df(zero_shot_topics, force_zero_shot_radio, create_revised_general_topics, max_topic_no)

                        

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
                    existing_topic_summary_df["General topic"] = existing_topic_summary_df["General topic"].str.replace('(?i)^Nan$', '', regex=True)
                    existing_topic_summary_df["Subtopic"] = existing_topic_summary_df["Subtopic"].str.replace('(?i)^Nan$', '', regex=True)
                    existing_topic_summary_df = existing_topic_summary_df.drop_duplicates()
                    if "Description" in existing_topic_summary_df:
                        if existing_topic_summary_df['Description'].isnull().all():
                            existing_topic_summary_df.drop("Description", axis = 1, inplace = True)

                    # If user has chosen to try to force zero shot topics, then the prompt is changed to ask the model not to deviate at all from submitted topic list.
                    keep_cols = [
                        col for col in ["General topic", "Subtopic", "Description"]
                        if col in existing_topic_summary_df.columns
                        and not existing_topic_summary_df[col].replace(r'^\s*$', pd.NA, regex=True).isna().all()
                        ]
                    
                    # Create topics table to be presented to LLM
                    topics_df_for_markdown = existing_topic_summary_df[keep_cols].drop_duplicates(keep_cols)
                    if "General topic" in topics_df_for_markdown.columns and "Subtopic" in topics_df_for_markdown.columns:
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(["General topic", "Subtopic"])

                    if produce_structures_summary_radio == "Yes":
                        if "General topic" in topics_df_for_markdown.columns:
                            topics_df_for_markdown = topics_df_for_markdown.rename(columns={"General topic":"Main Heading"})
                        if "Subtopic" in topics_df_for_markdown.columns:
                            topics_df_for_markdown = topics_df_for_markdown.rename(columns={"Subtopic":"Subheading"})

                    unique_topics_markdown = topics_df_for_markdown.to_markdown(index=False)
                    
                    if force_zero_shot_radio == "Yes": topic_assignment_prompt = force_existing_topics_prompt
                    else: topic_assignment_prompt = allow_new_topics_prompt  

                    # Should the outputs force only one single topic assignment per response?
                    if force_single_topic_radio != "Yes": force_single_topic_prompt = ""
                    else:
                        topic_assignment_prompt = topic_assignment_prompt.replace("Assign topics", "Assign a topic").replace("assign Subtopics", "assign a Subtopic").replace("Subtopics", "Subtopic").replace("Topics", "Topic").replace("topics", "a topic")         

                    # Format the summary prompt with the response table and topics
                    if produce_structures_summary_radio != "Yes":
                        formatted_summary_prompt = add_existing_topics_prompt.format(response_table=normalised_simple_markdown_table,
                                                                                     topics=unique_topics_markdown,
                                                                                     topic_assignment=topic_assignment_prompt, force_single_topic=force_single_topic_prompt, sentiment_choices=sentiment_prompt)
                    else:
                        formatted_summary_prompt = structured_summary_prompt.format(response_table=normalised_simple_markdown_table,
                                                                                    topics=unique_topics_markdown)
                    
                    if model_source == "Local":
                        #formatted_summary_prompt = llama_cpp_prefix + formatted_system_prompt + "\n" + formatted_summary_prompt + llama_cpp_suffix
                        full_prompt = formatted_system_prompt + "\n" + formatted_summary_prompt
                    else:
                        full_prompt = formatted_system_prompt + "\n" + formatted_summary_prompt

                     

                    # Save outputs for each batch. If master file created, label file as master
                    batch_file_path_details = f"{file_name_clean}_batch_{latest_batch_completed + 1}_size_{batch_size}_col_{in_column_cleaned}"        

                    # Define the output file path for the formatted prompt
                    formatted_prompt_output_path = output_folder + batch_file_path_details +  "_full_prompt_" + model_choice_clean_short + ".txt"

                    # Write the formatted prompt to the specified file
                    try:
                        with open(formatted_prompt_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                            f.write(full_prompt)
                    except Exception as e:
                        print(f"Error writing prompt to file {formatted_prompt_output_path}: {e}")

                    #if "Local" in model_source:
                    #    summary_prompt_list = [full_prompt] # Includes system prompt
                    #else:
                    summary_prompt_list = [formatted_summary_prompt]

                    if "Local" in model_source and reasoning_suffix: formatted_system_prompt = formatted_system_prompt + "\n" + reasoning_suffix

                    conversation_history = list()
                    whole_conversation = list()

                    # Process requests to large language model
                    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(summary_prompt_list, formatted_system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, google_client, google_config, model_choice, temperature, reported_batch_no, local_model, tokenizer, bedrock_runtime, model_source, MAX_OUTPUT_VALIDATION_ATTEMPTS, assistant_prefill=add_existing_topics_assistant_prefill,  master = True)

                    # Return output tables
                    topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, new_topic_df, new_reference_df, new_topic_summary_df, master_batch_out_file_part, is_error = write_llm_output_and_logs(response_text, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_topic_summary_df, batch_size, chosen_cols, batch_basic_response_df, model_name_map, group_name, produce_structures_summary_radio, first_run=False, output_folder=output_folder)                   
                    
                    # Write final output to text file for logging purposes
                    try:
                        final_table_output_path = output_folder + master_batch_out_file_part + "_full_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                        if isinstance(responses[-1], ResponseObject):
                            with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                                #f.write(responses[-1].text)
                                f.write(response_text)
                        elif "choices" in responses[-1]:
                            with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                                #f.write(responses[-1]["choices"][0]['text'])
                                f.write(response_text)
                        else:
                            with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                                #f.write(responses[-1].text)
                                f.write(response_text)

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

                    new_topic_summary_df["Group"] = group_name

                    new_topic_summary_df.to_csv(topic_summary_df_out_path, index=None)
                    out_file_paths.append(topic_summary_df_out_path)
                    
                    # Outputs for markdown table output
                    unique_table_df_display_table = new_topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
                    unique_table_df_display_table_markdown = unique_table_df_display_table[["General topic", "Subtopic", "Sentiment", "Number of responses", "Summary"]].to_markdown(index=False)

                    whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)

                    out_file_paths = [col for col in out_file_paths if str(reported_batch_no) in col]
                    log_files_output_paths = [col for col in out_file_paths if str(reported_batch_no) in col]
                # If this is the first batch, run this
                else:
                    formatted_initial_table_system_prompt = initial_table_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                    # Prepare Gemini models before query       
                    if model_source == "Gemini":
                        print("Using Gemini model:", model_choice)
                        google_client, google_config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=formatted_initial_table_system_prompt, max_tokens=max_tokens)
                    elif model_source == "Azure":
                        print("Using Azure AI Inference model:", model_choice)
                        if azure_api_key_textbox:
                            os.environ["AZURE_INFERENCE_CREDENTIAL"] = azure_api_key_textbox
                        google_client, google_config = construct_azure_client(in_api_key=azure_api_key_textbox, endpoint=AZURE_INFERENCE_ENDPOINT)
                    elif model_choice == CHOSEN_LOCAL_MODEL_TYPE:
                        print("Using local model:", model_choice)
                    else:
                        print("Using AWS Bedrock model:", model_choice)

                    # Format the summary prompt with the response table and topics
                    if produce_structures_summary_radio != "Yes":
                        formatted_initial_table_prompt = initial_table_prompt.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)
                    else:
                        unique_topics_markdown="No suggested headings for this summary"
                        formatted_initial_table_prompt = structured_summary_prompt.format(response_table=normalised_simple_markdown_table, topics=unique_topics_markdown)

                    if prompt2: formatted_prompt2 = prompt2.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)
                    else: formatted_prompt2 = prompt2
                    
                    if prompt3: formatted_prompt3 = prompt3.format(response_table=normalised_simple_markdown_table, sentiment_choices=sentiment_prompt)
                    else: formatted_prompt3 = prompt3

                    #if "Local" in model_source:
                    #formatted_initial_table_prompt = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_initial_table_prompt + llama_cpp_suffix
                    #formatted_prompt2 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt2 + llama_cpp_suffix
                    #formatted_prompt3 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt3 + llama_cpp_suffix

                    batch_prompts = [formatted_initial_table_prompt, formatted_prompt2, formatted_prompt3][:number_of_prompts_used]  # Adjust this list to send fewer requests

                    if "Local" in model_source and reasoning_suffix: formatted_initial_table_system_prompt = formatted_initial_table_system_prompt + "\n" + reasoning_suffix
                    
                    whole_conversation = list()

                    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(batch_prompts, formatted_initial_table_system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, google_client, google_config, model_choice, temperature, reported_batch_no, local_model, tokenizer,bedrock_runtime, model_source, MAX_OUTPUT_VALIDATION_ATTEMPTS, assistant_prefill=initial_table_assistant_prefill)
                    
                    topic_table_out_path, reference_table_out_path, topic_summary_df_out_path, topic_table_df, reference_df, new_topic_summary_df, batch_file_path_details, is_error =  write_llm_output_and_logs(response_text, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_topic_summary_df, batch_size, chosen_cols, batch_basic_response_df, model_name_map, group_name, produce_structures_summary_radio, first_run=True, output_folder=output_folder)

                    # If error in table parsing, leave function
                    if is_error == True: raise Exception("Error in output table parsing")                    

                    topic_table_df.to_csv(topic_table_out_path, index=None)
                    out_file_paths.append(topic_table_out_path)

                    reference_df.to_csv(reference_table_out_path, index=None)
                    out_file_paths.append(reference_table_out_path)

                    ## Unique topic list

                    new_topic_summary_df = pd.concat([new_topic_summary_df, existing_topic_summary_df]).drop_duplicates('Subtopic')

                    new_topic_summary_df["Group"] = group_name

                    new_topic_summary_df.to_csv(topic_summary_df_out_path, index=None)
                    out_file_paths.append(topic_summary_df_out_path)                    

                    whole_conversation_metadata.append(whole_conversation_metadata_str)
                    whole_conversation_metadata_str = '. '.join(whole_conversation_metadata)
                    
                    # Write final output to text file for logging purposes
                    try:
                        final_table_output_path = output_folder + batch_file_path_details + "_full_response_" + model_choice_clean + ".txt"

                        with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                            f.write(response_text)

                        # if isinstance(responses[-1], ResponseObject):
                        #     with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                        #         #f.write(responses[-1].text)
                        #         f.write(response_text)
                        # elif "choices" in responses[-1]:
                        #     with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                        #         #f.write(responses[-1]["choices"][0]['text'])
                        #         f.write(response_text)
                        # else:
                        #     with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                        #         #f.write(responses[-1].text)
                        #         f.write(response_text)

                    except Exception as e: print("Error in returning model response:", e)
                    
                    new_topic_df = topic_table_df
                    new_reference_df = reference_df

            else:
                print("Current batch of responses contains no text, moving onto next. Batch number:", str(latest_batch_completed + 1), ". Start row:", start_row, ". End row:", end_row)

            # Increase latest file completed count unless we are over the last batch number
            if latest_batch_completed <= num_batches:
                #print("Completed batch number:", str(reported_batch_no))
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

        join_file_paths = list()

        toc = time.perf_counter()
        final_time = (toc - tic) + time_taken
        out_time = f"Everything finished in {round(final_time,1)} seconds."
        print(out_time)

        print("All batches completed. Exporting outputs.")

        # model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)
        # in_column_cleaned = clean_column_name(chosen_cols, max_length=20)
        # file_name_cleaned = clean_column_name(file_name, max_length=20, front_characters=True)    

        # # Save outputs for each batch. If master file created, label file as master
        # file_path_details = f"{file_name_cleaned}_col_{in_column_cleaned}"

        file_path_details = create_batch_file_path_details(file_name)

        # Create a pivoted reference table
        existing_reference_df_pivot = convert_reference_table_to_pivot_table(existing_reference_df)        

        # Save the new DataFrame to CSV
        reference_table_out_pivot_path = output_folder + file_path_details + "_final_reference_table_pivot_" + model_choice_clean_short + "_temp_" + str(temperature) + ".csv"
        reference_table_out_path = output_folder + file_path_details + "_final_reference_table_" + model_choice_clean_short + "_temp_" + str(temperature) + ".csv" 
        topic_summary_df_out_path = output_folder + file_path_details + "_final_unique_topics_" + model_choice_clean_short + "_temp_" + str(temperature) + ".csv"
        basic_response_data_out_path = output_folder + file_path_details + "_simplified_data_file_" + model_choice_clean_short + "_temp_" + str(temperature) + ".csv"

        ## Reference table mapping response numbers to topics
        existing_reference_df.to_csv(reference_table_out_path, index=None)
        out_file_paths.append(reference_table_out_path)
        join_file_paths.append(reference_table_out_path)

        # Create final unique topics table from reference table to ensure consistent numbers
        final_out_topic_summary_df = create_topic_summary_df_from_reference_table(existing_reference_df)
        final_out_topic_summary_df["Group"] = group_name

        ## Unique topic list
        final_out_topic_summary_df.to_csv(topic_summary_df_out_path, index=None, encoding='utf-8-sig')
        out_file_paths.append(topic_summary_df_out_path)

        # Outputs for markdown table output
        unique_table_df_display_table = final_out_topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
        unique_table_df_display_table_markdown = unique_table_df_display_table[["General topic", "Subtopic", "Sentiment", "Number of responses", "Summary", "Group"]].to_markdown(index=False)

        # Ensure that we are only returning the final results to outputs
        out_file_paths = [x for x in out_file_paths if '_final_' in x]

        ## Reference table mapping response numbers to topics
        existing_reference_df_pivot["Group"] = group_name
        existing_reference_df_pivot.to_csv(reference_table_out_pivot_path, index = None, encoding='utf-8-sig')
        log_files_output_paths.append(reference_table_out_pivot_path)

        ## Create a dataframe for missing response references:
        # Assuming existing_reference_df and file_data are already defined
        # Simplify table to just responses column and the Response reference number
        basic_response_data = get_basic_response_data(file_data, chosen_cols)

        # Save simplified file data to log outputs
        pd.DataFrame(basic_response_data).to_csv(basic_response_data_out_path, index=None, encoding='utf-8-sig')
        log_files_output_paths.append(basic_response_data_out_path)

        # Create missing references dataframe
        missing_df = create_missing_references_df(basic_response_data, existing_reference_df)

        missing_df_out_path = output_folder + file_path_details + "_missing_references_" + model_choice_clean_short + "_temp_" + str(temperature) + ".csv"
        missing_df.to_csv(missing_df_out_path, index=None, encoding='utf-8-sig')
        log_files_output_paths.append(missing_df_out_path)

        out_file_paths = list(set(out_file_paths))
        log_files_output_paths = list(set(log_files_output_paths))        

        final_out_file_paths = [file_path for file_path in out_file_paths if "final_" in file_path]
 
        # The topic table that can be modified does not need the summary column
        modifiable_topic_summary_df = final_out_topic_summary_df.drop("Summary", axis=1)

        return unique_table_df_display_table_markdown, existing_topics_table, final_out_topic_summary_df, existing_reference_df, final_out_file_paths, final_out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_out_file_paths, final_out_file_paths, modifiable_topic_summary_df, final_out_file_paths, join_file_paths, existing_reference_df_pivot, missing_df

    return unique_table_df_display_table_markdown, existing_topics_table, existing_topic_summary_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths, modifiable_topic_summary_df, out_file_paths, join_file_paths, existing_reference_df_pivot, missing_df 

def wrapper_extract_topics_per_column_value(
    grouping_col: str,
    in_data_file: Any,
    file_data: pd.DataFrame,
    initial_existing_topics_table: pd.DataFrame,
    initial_existing_reference_df: pd.DataFrame,
    initial_existing_topic_summary_df: pd.DataFrame,
    initial_unique_table_df_display_table_markdown: str,
    original_file_name: str, # Original file name, to be modified per segment
    total_number_of_batches:int,
    in_api_key: str,        
    temperature: float,
    chosen_cols: List[str],
    model_choice: str,
    candidate_topics: GradioFileData = None,
    initial_first_loop_state: bool = True,
    initial_whole_conversation_metadata_str: str = '',
    initial_latest_batch_completed: int = 0, 
    initial_time_taken: float = 0,
    initial_table_prompt: str = initial_table_prompt,
    prompt2: str = prompt2,
    prompt3: str = prompt3,
    initial_table_system_prompt: str = initial_table_system_prompt,
    add_existing_topics_system_prompt: str = add_existing_topics_system_prompt,
    add_existing_topics_prompt: str = add_existing_topics_prompt,

    number_of_prompts_used: int = 1,
    batch_size: int = 50, # Crucial for calculating num_batches per segment
    context_textbox: str = "",
    sentiment_checkbox: str = "Negative, Neutral, or Positive",
    force_zero_shot_radio: str = "No",
    in_excel_sheets: List[str] = list(),
    force_single_topic_radio: str = "No",
    produce_structures_summary_radio: str = "No",
    aws_access_key_textbox:str="",
    aws_secret_key_textbox:str="",
    hf_api_key_textbox:str="",
    azure_api_key_textbox:str="",
    output_folder: str = OUTPUT_FOLDER,
    force_single_topic_prompt: str = force_single_topic_prompt,
    max_tokens: int = max_tokens,
    model_name_map: dict = model_name_map,
    max_time_for_loop: int = max_time_for_loop, # This applies per call to extract_topics
    reasoning_suffix: str = reasoning_suffix,
    CHOSEN_LOCAL_MODEL_TYPE: str = CHOSEN_LOCAL_MODEL_TYPE,
    progress=Progress(track_tqdm=True) # type: ignore
) -> Tuple: # Mimicking the return tuple structure of extract_topics
    """
    A wrapper function that iterates through unique values in a specified grouping column
    and calls the `extract_topics` function for each segment of the data.
    It accumulates results from each call and returns a consolidated output.

    :param grouping_col: The name of the column to group the data by.
    :param in_data_file: The input data file object (e.g., Gradio FileData).
    :param file_data: The full DataFrame containing all data.
    :param initial_existing_topics_table: Initial DataFrame of existing topics.
    :param initial_existing_reference_df: Initial DataFrame mapping responses to topics.
    :param initial_existing_topic_summary_df: Initial DataFrame summarizing topics.
    :param initial_unique_table_df_display_table_markdown: Initial markdown string for topic display.
    :param original_file_name: The original name of the input file.
    :param total_number_of_batches: The total number of batches across all data.
    :param in_api_key: API key for the chosen LLM.
    :param temperature: Temperature setting for the LLM.
    :param chosen_cols: List of columns from `file_data` to be processed.
    :param model_choice: The chosen LLM model (e.g., "Gemini", "AWS Claude").
    :param candidate_topics: Optional Gradio FileData for candidate topics (zero-shot).
    :param initial_first_loop_state: Boolean indicating if this is the very first loop iteration.
    :param initial_whole_conversation_metadata_str: Initial metadata string for the whole conversation.
    :param initial_latest_batch_completed: The batch number completed in the previous run.
    :param initial_time_taken: Initial time taken for processing.
    :param initial_table_prompt: The initial prompt for table summarization.
    :param prompt2: The second prompt for LLM interaction.
    :param prompt3: The third prompt for LLM interaction.
    :param initial_table_system_prompt: The initial system prompt for table summarization.
    :param add_existing_topics_system_prompt: System prompt for adding existing topics.
    :param add_existing_topics_prompt: Prompt for adding existing topics.
    :param number_of_prompts_used: Number of prompts used in the LLM call.
    :param batch_size: Number of rows to process in each batch for the LLM.
    :param context_textbox: Additional context provided by the user.
    :param sentiment_checkbox: Choice for sentiment assessment (e.g., "Negative, Neutral, or Positive").
    :param force_zero_shot_radio: Option to force responses into zero-shot topics.
    :param in_excel_sheets: List of Excel sheet names if applicable.
    :param force_single_topic_radio: Option to force a single topic per response.
    :param produce_structures_summary_radio: Option to produce a structured summary.
    :param aws_access_key_textbox: AWS access key for Bedrock.
    :param aws_secret_key_textbox: AWS secret key for Bedrock.
    :param hf_api_key_textbox: Hugging Face API key for local models.
    :param azure_api_key_textbox: Azure API key for Azure AI Inference.
    :param output_folder: The folder where output files will be saved.
    :param force_single_topic_prompt: Prompt for forcing a single topic.
    :param max_tokens: Maximum tokens for LLM generation.
    :param model_name_map: Dictionary mapping model names to their properties.
    :param max_time_for_loop: Maximum time allowed for the processing loop.
    :param reasoning_suffix: Suffix to append for reasoning.
    :param CHOSEN_LOCAL_MODEL_TYPE: Type of local model chosen.
    :param progress: Gradio Progress object for tracking progress.
    :return: A tuple containing consolidated results, mimicking the return structure of `extract_topics`.
    """
    
    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_number_of_calls = 0
    out_message = list()

    if grouping_col is None:
        print("No grouping column found")
        file_data["group_col"] = "All"
        grouping_col="group_col"

    if grouping_col not in file_data.columns:
        raise ValueError(f"Selected column '{grouping_col}' not found in file_data.")

    unique_values = file_data[grouping_col].unique()
    #print("unique_values:", unique_values)

    if len(unique_values) > MAX_GROUPS:
        print(f"Warning: More than {MAX_GROUPS} unique values found in '{grouping_col}'. Processing only the first {MAX_GROUPS}.")
        unique_values = unique_values[:MAX_GROUPS]

    # Initialize accumulators for results across all unique values
    # DataFrames are built upon iteratively
    acc_topics_table = initial_existing_topics_table.copy()
    acc_reference_df = initial_existing_reference_df.copy()
    acc_topic_summary_df = initial_existing_topic_summary_df.copy()
    acc_reference_df_pivot = pd.DataFrame()
    acc_missing_df = pd.DataFrame()

    # Lists are extended
    acc_out_file_paths = list()
    acc_log_files_output_paths = list()
    acc_join_file_paths = list() # join_file_paths seems to be overwritten, so maybe last one or extend? Let's extend.

    # Single value outputs - typically the last one is most relevant, or sum for time
    acc_markdown_output = initial_unique_table_df_display_table_markdown
    acc_latest_batch_completed = initial_latest_batch_completed # From the last segment processed
    acc_whole_conversation_metadata = initial_whole_conversation_metadata_str
    acc_total_time_taken = float(initial_time_taken)
    acc_gradio_df = gr.Dataframe(value=pd.DataFrame()) # type: ignore # Placeholder for the last Gradio DF

    wrapper_first_loop = initial_first_loop_state

    if len(unique_values) == 1:
        loop_object = enumerate(unique_values)
    else:
        loop_object = tqdm(enumerate(unique_values), desc=f"Analysing group", total=len(unique_values), unit="groups")


    for i, group_value in loop_object:
        print(f"\nProcessing group: {grouping_col} = {group_value} ({i+1}/{len(unique_values)})")
        
        filtered_file_data = file_data.copy()

        filtered_file_data = filtered_file_data[filtered_file_data[grouping_col] == group_value]

        if filtered_file_data.empty:
            print(f"No data for {grouping_col} = {group_value}. Skipping.")
            continue

        # Calculate num_batches for this specific segment
        current_num_batches = (len(filtered_file_data) + batch_size - 1) // batch_size
        
        # Modify file_name to be unique for this segment's outputs
        # _grp_{clean_column_name(grouping_col, max_length=15)}
        segment_file_name = f"{clean_column_name(original_file_name, max_length=15)}_{clean_column_name(str(group_value), max_length=15).replace(' ','_')}"

        # Determine first_loop_state for this call to extract_topics
        # It's True only if this is the very first segment *and* the wrapper was told it's the first loop.
        # For subsequent segments, it's False, as we are building on accumulated DFs.
        current_first_loop_state = wrapper_first_loop if i == 0 else False
        
        # latest_batch_completed for extract_topics should be 0 for each new segment,
        # as it processes the new filtered_file_data from its beginning.
        # However, if it's the very first call, respect initial_latest_batch_completed.
        current_latest_batch_completed = initial_latest_batch_completed if i == 0 and wrapper_first_loop else 0


        # Call extract_topics for the current segment
        try:
            (
                seg_markdown,
                seg_topics_table,
                seg_topic_summary_df,
                seg_reference_df,
                seg_out_files1,
                _seg_out_files2, # Often same as 1
                seg_batch_completed, # Specific to this segment's run
                seg_log_files1,
                _seg_log_files2, # Often same as 1
                seg_conversation_metadata,
                seg_time_taken,
                _seg_out_files3, # Often same as 1
                _seg_out_files4, # Often same as 1
                seg_gradio_df,
                _seg_out_files5, # Often same as 1
                seg_join_files,
                seg_reference_df_pivot,
                seg_missing_df
                        ) = extract_topics(
                in_data_file=in_data_file,
                file_data=filtered_file_data,
                existing_topics_table=pd.DataFrame(), #acc_topics_table.copy(), # Pass the accumulated table
                existing_reference_df=pd.DataFrame(),#acc_reference_df.copy(), # Pass the accumulated table
                existing_topic_summary_df=pd.DataFrame(),#acc_topic_summary_df.copy(), # Pass the accumulated table
                unique_table_df_display_table_markdown="", # extract_topics will generate this
                file_name=segment_file_name,
                num_batches=current_num_batches,
                in_api_key=in_api_key,
                temperature=temperature,
                chosen_cols=chosen_cols,
                model_choice=model_choice,
                candidate_topics=candidate_topics,
                latest_batch_completed=current_latest_batch_completed, # Reset for each new segment's internal batching
                out_message= list(), # Fresh for each call
                out_file_paths= list(),# Fresh for each call
                log_files_output_paths= list(),# Fresh for each call                
                first_loop_state=current_first_loop_state, # True only for the very first iteration of wrapper                
                whole_conversation_metadata_str="", # Fresh for each call
                initial_table_prompt=initial_table_prompt,
                prompt2=prompt2,
                prompt3=prompt3,
                initial_table_system_prompt=initial_table_system_prompt,
                add_existing_topics_system_prompt=add_existing_topics_system_prompt,
                add_existing_topics_prompt=add_existing_topics_prompt,
                number_of_prompts_used=number_of_prompts_used,
                batch_size=batch_size,
                context_textbox=context_textbox,
                time_taken=0, # Time taken for this specific call, wrapper sums it.
                sentiment_checkbox=sentiment_checkbox,
                force_zero_shot_radio=force_zero_shot_radio,
                in_excel_sheets=in_excel_sheets,
                force_single_topic_radio=force_single_topic_radio,
                output_folder=output_folder,
                force_single_topic_prompt=force_single_topic_prompt,
                group_name=group_value,
                produce_structures_summary_radio=produce_structures_summary_radio,
                aws_access_key_textbox=aws_access_key_textbox,
                aws_secret_key_textbox=aws_secret_key_textbox,
                hf_api_key_textbox=hf_api_key_textbox,
                azure_api_key_textbox=azure_api_key_textbox,
                max_tokens=max_tokens,
                model_name_map=model_name_map,
                max_time_for_loop=max_time_for_loop,
                CHOSEN_LOCAL_MODEL_TYPE=CHOSEN_LOCAL_MODEL_TYPE,
                reasoning_suffix=reasoning_suffix,
                progress=progress
            )

            # Aggregate results
            # The DFs returned by extract_topics are already cumulative for *its own run*.
            # We now make them cumulative for the *wrapper's run*.
            acc_reference_df = pd.concat([acc_reference_df, seg_reference_df])
            acc_topic_summary_df = pd.concat([acc_topic_summary_df, seg_topic_summary_df])
            acc_reference_df_pivot = pd.concat([acc_reference_df_pivot, seg_reference_df_pivot])
            acc_missing_df = pd.concat([acc_missing_df, seg_missing_df])
            
            # For lists, extend. Use set to remove duplicates if paths might be re-added.
            acc_out_file_paths.extend(f for f in seg_out_files1 if f not in acc_out_file_paths)
            acc_log_files_output_paths.extend(f for f in seg_log_files1 if f not in acc_log_files_output_paths)
            acc_join_file_paths.extend(f for f in seg_join_files if f not in acc_join_file_paths)

            acc_markdown_output = seg_markdown # Keep the latest markdown
            acc_latest_batch_completed = seg_batch_completed # Keep latest batch count
            acc_whole_conversation_metadata += (("\n---\n" if acc_whole_conversation_metadata else "") +
                                               f"Segment {grouping_col}={group_value}:\n" +
                                               seg_conversation_metadata)
            acc_total_time_taken += float(seg_time_taken)
            acc_gradio_df = seg_gradio_df # Keep the latest Gradio DF            

            print(f"Group {grouping_col} = {group_value} processed. Time: {seg_time_taken:.2f}s")

        except Exception as e:
            print(f"Error processing segment {grouping_col} = {group_value}: {e}")
            # Optionally, decide if you want to continue with other segments or stop
            # For now, it will continue
            continue

    if "Group" in acc_reference_df.columns:
        model_choice_clean = model_name_map[model_choice]["short_name"]
        model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)
        overall_file_name = clean_column_name(original_file_name, max_length=20)
        column_clean = clean_column_name(chosen_cols, max_length=20)
        
        acc_reference_df_path = output_folder + overall_file_name + "_col_" + column_clean + "_all_final_reference_table_" + model_choice_clean_short + ".csv"
        acc_topic_summary_df_path = output_folder + overall_file_name + "_col_" + column_clean +  "_all_final_unique_topics_" + model_choice_clean_short + ".csv"
        acc_reference_df_pivot_path = output_folder + overall_file_name + "_col_" + column_clean +  "_all_final_reference_pivot_" + model_choice_clean_short + ".csv"
        acc_missing_df_path = output_folder + overall_file_name + "_col_" + column_clean + "_all_missing_df_" + model_choice_clean_short + ".csv"        

        acc_reference_df.to_csv(acc_reference_df_path, index=None)
        acc_topic_summary_df.to_csv(acc_topic_summary_df_path, index=None)
        acc_reference_df_pivot.to_csv(acc_reference_df_pivot_path, index=None)
        acc_missing_df.to_csv(acc_missing_df_path, index=None)

        acc_log_files_output_paths.append(acc_missing_df_path)

        # Remove the existing output file list and replace with the updated concatenated outputs
        substring_list_to_remove = ["_final_reference_table_pivot_", "_final_reference_table_", "_final_unique_topics_"]
        acc_out_file_paths = [
            x for x in acc_out_file_paths
            if not any(sub in x for sub in substring_list_to_remove)
        ]

        acc_out_file_paths.extend([acc_reference_df_path, acc_topic_summary_df_path])

        # Outputs for markdown table output
        unique_table_df_display_table = acc_topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
        acc_markdown_output = unique_table_df_display_table[["General topic", "Subtopic", "Sentiment", "Number of responses", "Summary", "Group"]].to_markdown(index=False)

    acc_input_tokens, acc_output_tokens, acc_number_of_calls = calculate_tokens_from_metadata(acc_whole_conversation_metadata, model_choice, model_name_map)

    out_message = '\n'.join(out_message)
    out_message = out_message + " " + f"Topic extraction finished processing all groups. Total time: {acc_total_time_taken:.2f}s"
    print(out_message)

    # The return signature should match extract_topics.
    # The aggregated lists will be returned in the multiple slots.
    return (
        acc_markdown_output,
        acc_topics_table,
        acc_topic_summary_df,
        acc_reference_df,
        acc_out_file_paths, # Slot 1 for out_file_paths
        acc_out_file_paths, # Slot 2 for out_file_paths
        acc_latest_batch_completed, # From the last successfully processed segment
        acc_log_files_output_paths, # Slot 1 for log_files_output_paths
        acc_log_files_output_paths, # Slot 2 for log_files_output_paths
        acc_whole_conversation_metadata,
        acc_total_time_taken,
        acc_out_file_paths, # Slot 3
        acc_out_file_paths, # Slot 4
        acc_gradio_df,      # Last Gradio DF
        acc_out_file_paths, # Slot 5
        acc_join_file_paths,
        acc_missing_df,
        acc_input_tokens,
        acc_output_tokens,
        acc_number_of_calls,
        out_message
    )


def join_modified_topic_names_to_ref_table(modified_topic_summary_df:pd.DataFrame, original_topic_summary_df:pd.DataFrame, reference_df:pd.DataFrame):
    '''
    Take a unique topic table that has been modified by the user, and apply the topic name changes to the long-form reference table.
    '''

    # Drop rows where Number of responses is either NA or null
    modified_topic_summary_df = modified_topic_summary_df[~modified_topic_summary_df["Number of responses"].isnull()]
    modified_topic_summary_df.drop_duplicates(["General topic", "Subtopic", "Sentiment", "Topic_number"], inplace=True)

    # First, join the modified topics to the original topics dataframe based on index to have the modified names alongside the original names
    original_topic_summary_df_m = original_topic_summary_df.merge(modified_topic_summary_df[["General topic", "Subtopic", "Sentiment", "Topic_number"]], on="Topic_number", how="left", suffixes=("", "_mod"))

    original_topic_summary_df_m.drop_duplicates(["General topic", "Subtopic", "Sentiment", "Topic_number"], inplace=True)


    # Then, join these new topic names onto the reference_df, merge based on the original names
    modified_reference_df = reference_df.merge(original_topic_summary_df_m[["Topic_number", "General Topic_mod", "Subtopic_mod", "Sentiment_mod"]], on=["Topic_number"], how="left")

    
    modified_reference_df.drop(["General topic", "Subtopic", "Sentiment"], axis=1, inplace=True, errors="ignore")
    
    modified_reference_df.rename(columns={"General Topic_mod":"General topic",
                                                                 "Subtopic_mod":"Subtopic",
                                                                 "Sentiment_mod":"Sentiment"}, inplace=True)

    modified_reference_df.drop(["General Topic_mod", "Subtopic_mod", "Sentiment_mod"], inplace=True, errors="ignore")  
    

    #modified_reference_df.drop_duplicates(["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)

    modified_reference_df.sort_values(["Start row of group", "Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)

    modified_reference_df = modified_reference_df.loc[:, ["Response References", "General topic", "Subtopic", "Sentiment", "Summary", "Start row of group", "Topic_number"]]

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

    output_file_list = list()

    if reference_file_path and unique_table_file_path:

        reference_df = join_modified_topic_names_to_ref_table(modifiable_topic_summary_df, original_topic_summary_df, reference_df)

        ## Reference table mapping response numbers to topics
        reference_table_file_name = reference_file_path.replace(".csv", "_mod") 
        new_reference_df_file_path = output_folder + reference_table_file_name  + ".csv"
        reference_df.to_csv(new_reference_df_file_path, index=None, encoding='utf-8-sig')
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
        modifiable_topic_summary_df.to_csv(modified_unique_table_file_path, index=None, encoding='utf-8-sig')
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
