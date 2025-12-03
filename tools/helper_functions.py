import os
import re
import boto3
import gradio as gr
import pandas as pd
import numpy as np
from typing import List
import math
import codecs
from botocore.exceptions import ClientError
from tools.config import OUTPUT_FOLDER, INPUT_FOLDER, SESSION_OUTPUT_FOLDER, CUSTOM_HEADER, CUSTOM_HEADER_VALUE, AWS_USER_POOL_ID, MAXIMUM_ZERO_SHOT_TOPICS, model_name_map, model_full_names

def empty_output_vars_extract_topics():
    # Empty output objects before processing a new file

    master_topic_df_state = pd.DataFrame()
    master_topic_summary_df_state = pd.DataFrame()
    master_reference_df_state = pd.DataFrame()
    text_output_file = list()
    text_output_file_list_state = list()
    latest_batch_completed = 0
    log_files_output = list()
    log_files_output_list_state = list()
    conversation_metadata_textbox = ""
    estimated_time_taken_number = 0
    file_data_state = pd.DataFrame()
    reference_data_file_name_textbox = ""
    display_topic_table_markdown = ""
    summary_output_file_list = list()
    summary_input_file_list = list()
    overall_summarisation_input_files = list()
    overall_summary_output_files = list()

    return master_topic_df_state, master_topic_summary_df_state, master_reference_df_state, text_output_file, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, reference_data_file_name_textbox, display_topic_table_markdown, summary_output_file_list, summary_input_file_list, overall_summarisation_input_files, overall_summary_output_files

def empty_output_vars_summarise():
    # Empty output objects before summarising files

    summary_reference_table_sample_state = pd.DataFrame()
    master_topic_summary_df_revised_summaries_state = pd.DataFrame()
    master_reference_df_revised_summaries_state = pd.DataFrame()
    summary_output_files = list()
    summarised_outputs_list = list()
    latest_summary_completed_num = 0
    overall_summarisation_input_files = list()

    return summary_reference_table_sample_state, master_topic_summary_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, overall_summarisation_input_files

def get_or_create_env_var(var_name:str, default_value:str):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value

def get_file_path_with_extension(file_path:str):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Return the basename with its extension
    return basename

def get_file_name_no_ext(file_path:str):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def detect_file_type(filename:str):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pdf'):
        return 'pdf'
    elif filename.endswith('.jpg'):
        return 'jpg'
    elif filename.endswith('.jpeg'):
        return 'jpeg'
    elif filename.endswith('.png'):
        return 'png'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename:str, sheet:str=""):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        if sheet:
            return pd.read_excel(filename, sheet_name=sheet)
        else:
            return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)
    
def load_in_file(file_path: str, colnames:List[str]="", excel_sheet:str=""):
    """
    Loads in a tabular data file and returns data and file name.

    Parameters:
    - file_path (str): The path to the file to be processed.
    - colnames (List[str], optional): list of colnames to load in
    """

    file_name = get_file_name_no_ext(file_path)
    file_data = read_file(file_path, excel_sheet)

    if colnames and isinstance(colnames, list):
        col_list = colnames
    else:
        col_list = list(file_data.columns)

    if not isinstance(col_list, List):
        col_list = [col_list]

    col_list = [item for item in col_list if item not in ["", "NA"]]

    for col in col_list:
        file_data[col] = file_data[col].fillna("")
        file_data[col] = file_data[col].astype(str).str.replace("\bnan\b", "", regex=True)  
        
        #print(file_data[colnames])

    return file_data, file_name

def load_in_data_file(file_paths:List[str], in_colnames:List[str], batch_size:int=5, in_excel_sheets:str=""):
    '''Load in data table, work out how many batches needed.'''

    if not isinstance(in_colnames, list):
        in_colnames = [in_colnames]

    #print("in_colnames:", in_colnames)

    try:
        file_data, file_name = load_in_file(file_paths[0], colnames=in_colnames, excel_sheet=in_excel_sheets)
        num_batches = math.ceil(len(file_data) / batch_size)
        print("Total number of batches:", num_batches)

    except Exception as e:
        print(e)
        file_data = pd.DataFrame()
        file_name = ""
        num_batches = 1  
    
    return file_data, file_name, num_batches

def clean_column_name(column_name:str, max_length:int=20, front_characters:bool=True):
    # Convert to string
    column_name = str(column_name)  
    # Replace non-alphanumeric characters (except underscores) with underscores
    column_name = re.sub(r'\W+', '_', column_name)  
    # Remove leading/trailing underscores
    column_name = column_name.strip('_')  
    # Ensure the result is not empty; fall back to "column" if necessary
    column_name = column_name if column_name else "column"
    # Truncate to max_length
    if front_characters == True:
        output_text = column_name[:max_length]
    else:
        output_text = column_name[-max_length:]
    return output_text

def load_in_previous_reference_file(file:str):
    '''Load in data table from a partially completed consultation summary to continue it.'''

    reference_file_data = pd.DataFrame()
    reference_file_name = ""
    out_message = ""

    #for file in file_paths:

    print("file:", file)

    # If reference table
    if 'reference_table' in file:
        try:
            reference_file_data, reference_file_name = load_in_file(file)
            #print("reference_file_data:", reference_file_data.head(2))
            out_message = out_message + " Reference file load successful."
        except Exception as e:
            out_message = "Could not load reference file data:" + str(e)
            raise Exception("Could not load reference file data:", e)

    if reference_file_data.empty:
        out_message = out_message + " No reference data table provided."
        raise Exception(out_message) 

    print(out_message)
         
    return reference_file_data, reference_file_name

def load_in_previous_data_files(file_paths_partial_output:List[str], for_modified_table:bool=False):
    '''Load in data table from a partially completed consultation summary to continue it.'''

    reference_file_data = pd.DataFrame()
    reference_file_name = ""
    unique_file_data = pd.DataFrame()
    unique_file_name = ""
    out_message = ""
    latest_batch = 0

    if not file_paths_partial_output:
        out_message = out_message + " No reference or unique data table provided."
        return reference_file_data, unique_file_data, latest_batch, out_message, reference_file_name, unique_file_name

    if not isinstance(file_paths_partial_output, list):
        file_paths_partial_output = [file_paths_partial_output]

    for file in file_paths_partial_output:

        if isinstance(file, gr.FileData):
            name = file.name
        else:
            name = file

        # If reference table
        if 'reference_table' in name:
            try:
                reference_file_data, reference_file_name = load_in_file(file)
                #print("reference_file_data:", reference_file_data.head(2))
                out_message = out_message + " Reference file load successful."

            except Exception as e:
                out_message = "Could not load reference file data:" + str(e)
                raise Exception("Could not load reference file data:", e)
        # If unique table
        if 'unique_topic' in name:
            try:
                unique_file_data, unique_file_name = load_in_file(file)
                #print("unique_topics_file:", unique_file_data.head(2))
                out_message = out_message + " Unique table file load successful."
            except Exception as e:
                out_message = "Could not load unique table file data:" + str(e)
                raise Exception("Could not load unique table file data:", e)
        if 'batch_' in name:
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
        reference_file_data.drop("Topic number", axis=1, inplace=True, errors="ignore")

        unique_file_data = create_topic_summary_df_from_reference_table(reference_file_data)

        unique_file_data.drop("Summary",axis=1, inplace=True)

        # Then merge the topic numbers back to the original dataframe
        reference_file_data = reference_file_data.merge(
            unique_file_data[['General topic', 'Subtopic', 'Sentiment', 'Topic number']],
            on=['General topic', 'Subtopic', 'Sentiment'],
            how='left'
        )        

        out_file_names = [reference_file_name + ".csv"]
        out_file_names.append(unique_file_name + ".csv")

        return unique_file_data, reference_file_data, unique_file_data, reference_file_name, unique_file_name, out_file_names # gr.Dataframe(value=unique_file_data, headers=None, column_count=(unique_file_data.shape[1], "fixed"), row_count = (unique_file_data.shape[0], "fixed"), visible=True, type="pandas")

def join_cols_onto_reference_df(reference_df:pd.DataFrame, original_data_df:pd.DataFrame, join_columns:List[str], original_file_name:str, output_folder:str=OUTPUT_FOLDER):

    #print("original_data_df columns:", original_data_df.columns)
    #print("original_data_df:", original_data_df)

    original_data_df.reset_index(names="Response References", inplace=True)    
    original_data_df["Response References"] += 1

    #print("reference_df columns:", reference_df.columns)
    #print("reference_df:", reference_df)

    join_columns.append("Response References")

    reference_df["Response References"] = reference_df["Response References"].fillna("-1").astype(int) 

    save_file_name = output_folder + original_file_name + "_j.csv"

    out_reference_df = reference_df.merge(original_data_df[join_columns], on = "Response References", how="left")
    out_reference_df.to_csv(save_file_name, index=None)    

    file_data_outputs = [save_file_name]

    return out_reference_df, file_data_outputs

def get_basic_response_data(file_data:pd.DataFrame, chosen_cols:List[str], verify_titles:bool=False) -> pd.DataFrame:

    if not isinstance(chosen_cols, list):
        chosen_cols = [chosen_cols]

    if chosen_cols[0] not in file_data.columns:
        print("Column:", chosen_cols[0], "not found in file_data columns:", file_data.columns)

    basic_response_data = file_data[[chosen_cols[0]]]
    basic_response_data = basic_response_data.rename(columns={basic_response_data.columns[0]:"Response"})
    basic_response_data = basic_response_data.reset_index(names="Original Reference")#.reset_index(drop=True) #
    # Try to convert to int, if it fails, return a range of 1 to last row + 1
    try:
        basic_response_data["Original Reference"] = basic_response_data["Original Reference"].astype(int) + 1
    except (ValueError, TypeError):
            basic_response_data["Original Reference"] = range(1, len(basic_response_data) + 1)

    basic_response_data["Reference"] = basic_response_data.index.astype(int) + 1

    if verify_titles == True:
        basic_response_data = basic_response_data.rename(columns={chosen_cols[1]: "Title"})
        basic_response_data["Title"] = basic_response_data["Title"].str.strip()
        basic_response_data["Title"] = basic_response_data["Title"].apply(initial_clean)
    else:
        basic_response_data = basic_response_data[['Reference', 'Response', 'Original Reference']]

    basic_response_data["Response"] = basic_response_data["Response"].str.strip()
    basic_response_data["Response"] = basic_response_data["Response"].apply(initial_clean)

    return basic_response_data

def convert_reference_table_to_pivot_table(df:pd.DataFrame, basic_response_data:pd.DataFrame=pd.DataFrame()):

    df_in = df[['Response References', 'General topic', 'Subtopic', 'Sentiment']].copy()

    df_in['Response References'] = df_in['Response References'].astype(int)

    # Create a combined category column
    df_in['Category'] = df_in['General topic'] + ' - ' + df_in['Subtopic'] + ' - ' + df_in['Sentiment']
    
    # Create pivot table counting occurrences of each unique combination
    pivot_table = pd.crosstab(
        index=df_in['Response References'],
        columns=[df_in['General topic'], df_in['Subtopic'], df_in['Sentiment']],
        margins=True
    )
    
    # Flatten column names to make them more readable
    pivot_table.columns = [' - '.join(col) for col in pivot_table.columns]

    pivot_table.reset_index(inplace=True)

    if not basic_response_data.empty:
        pivot_table = basic_response_data.merge(pivot_table, right_on="Response References", left_on="Reference", how="left")

        pivot_table.drop("Response References", axis=1, inplace=True)    

    pivot_table.columns = pivot_table.columns.str.replace("Not assessed - ", "").str.replace("- Not assessed", "")

    return pivot_table

def create_topic_summary_df_from_reference_table(reference_df:pd.DataFrame):

    if "Group" not in reference_df.columns:
        reference_df["Group"] = "All"

    # Ensure 'Start row of group' column is numeric to avoid comparison errors
    if 'Start row of group' in reference_df.columns:
        reference_df['Start row of group'] = pd.to_numeric(reference_df['Start row of group'], errors='coerce')
    
    out_topic_summary_df = (reference_df.groupby(["General topic", "Subtopic", "Sentiment", "Group"])
            .agg({
                'Response References': 'size',  # Count the number of references
                'Summary': lambda x: '<br>'.join(
                    sorted(set(x), key=lambda summary: reference_df.loc[reference_df['Summary'] == summary, 'Start row of group'].min())
                )
            })
            .reset_index()
            #.sort_values('Response References', ascending=False)  # Sort by size, biggest first
        )
    
    out_topic_summary_df = out_topic_summary_df.rename(columns={"Response References": "Number of responses"}, errors="ignore")

    # Sort the dataframe first
    out_topic_summary_df = out_topic_summary_df.sort_values(["Group", "Number of responses", "General topic", "Subtopic", "Sentiment"], ascending=[True, False, True, True, True])

    # Then assign Topic number based on the final sorted order
    out_topic_summary_df = out_topic_summary_df.assign(Topic_number=lambda df: np.arange(1, len(df) + 1))

    out_topic_summary_df.rename(columns={"Topic_number":"Topic number"}, inplace=True)

    return out_topic_summary_df

# Wrap text in each column to the specified max width, including whole words
def wrap_text(text:str, max_width=100, max_text_length=None):
    if not isinstance(text, str):
        return text
        
    # If max_text_length is set, truncate the text and add ellipsis
    if max_text_length and len(text) > max_text_length:
        text = text[:max_text_length] + '...'
    
    text = text.replace('\r\n', '<br>').replace('\n', '<br>')
    
    words = text.split()
    if not words:
        return text
        
    # First pass: initial word wrapping
    wrapped_lines = list()
    current_line = list()
    current_length = 0
    
    def add_line():
        if current_line:
            wrapped_lines.append(' '.join(current_line))
            current_line.clear()
    
    for i, word in enumerate(words):
        word_length = len(word)
        
        # Handle words longer than max_width
        if word_length > max_width:
            add_line()
            wrapped_lines.append(word)
            current_length = 0
            continue
            
        # Calculate space needed for this word
        space_needed = word_length if not current_line else word_length + 1
        
        # Check if adding this word would exceed max_width
        if current_length + space_needed > max_width:
            add_line()
            current_line.append(word)
            current_length = word_length
        else:
            current_line.append(word)
            current_length += space_needed
    
    add_line()  # Add any remaining text
    
    # Second pass: redistribute words from lines following single-word lines
    def can_fit_in_previous_line(prev_line, word):
        return len(prev_line) + 1 + len(word) <= max_width
    
    i = 0
    while i < len(wrapped_lines) - 1:
        words_in_line = wrapped_lines[i].split()
        next_line_words = wrapped_lines[i + 1].split()
        
        # If current line has only one word and isn't too long
        if len(words_in_line) == 1 and len(words_in_line[0]) < max_width * 0.8:
            # Try to bring words back from the next line
            words_to_bring_back = list()
            remaining_words = list()
            current_length = len(words_in_line[0])
            
            for word in next_line_words:
                if current_length + len(word) + 1 <= max_width:
                    words_to_bring_back.append(word)
                    current_length += len(word) + 1
                else:
                    remaining_words.append(word)
            
            if words_to_bring_back:
                # Update current line with additional words
                wrapped_lines[i] = ' '.join(words_in_line + words_to_bring_back)
                
                # Update next line with remaining words
                if remaining_words:
                    wrapped_lines[i + 1] = ' '.join(remaining_words)
                else:
                    wrapped_lines.pop(i + 1)
                    continue  # Don't increment i if we removed a line
        i += 1
    
    return '<br>'.join(wrapped_lines)

def initial_clean(text:str):
    #### Some of my cleaning functions
    html_pattern_regex = r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\xa0|&nbsp;'
    html_start_pattern_end_dots_regex = r'<(.*?)\.\.'
    non_ascii_pattern = r'[^\x00-\x7F]+'
    multiple_spaces_regex = r'\s{2,}'
        
    # Define a list of patterns and their replacements
    patterns = [
        (html_pattern_regex, ' '),
        (html_start_pattern_end_dots_regex, ' '),
        (non_ascii_pattern, ' '),
        (multiple_spaces_regex, ' ')
    ]
    
    # Apply each regex replacement
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text

def view_table(file_path: str):  # Added max_width parameter
    df = pd.read_csv(file_path)

    df_cleaned = df.replace('\n', ' ', regex=True)    

    # Use apply with axis=1 to apply wrap_text to each element
    df_cleaned = df_cleaned.apply(lambda col: col.map(wrap_text))

    table_out = df_cleaned.to_markdown(index=False)

    return table_out

def ensure_output_folder_exists():
    """Checks if the 'output/' folder exists, creates it if not."""

    folder_name = "output/"

    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Created the 'output/' folder.")
    else:
        print(f"The 'output/' folder already exists.")

def put_columns_in_df(in_file:List[str]):
    new_choices = list()
    concat_choices = list()
    all_sheet_names = list()
    number_of_excel_files = 0

    if not in_file:
        return gr.Dropdown(choices=list()), gr.Dropdown(choices=list()), "", gr.Dropdown(choices=list()), gr.Dropdown(choices=list())
    
    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        #print("File type is:", file_type)

        file_end = get_file_path_with_extension(file_name)

        if file_type == 'xlsx':
            number_of_excel_files += 1
            new_choices = list()
            print("Running through all xlsx sheets")
            anon_xlsx = pd.ExcelFile(file_name)
            new_sheet_names = anon_xlsx.sheet_names
            # Iterate through the sheet names
            for sheet_name in new_sheet_names:
                # Read each sheet into a DataFrame
                df = pd.read_excel(file_name, sheet_name=sheet_name)

                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        else:
            df = read_file(file_name)
            new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        
    # Drop duplicate columns
    concat_choices = sorted(set(concat_choices))

    if number_of_excel_files > 0:      
        return gr.Dropdown(choices=concat_choices, value=concat_choices[0]), \
        gr.Dropdown(choices=all_sheet_names, value=all_sheet_names[0], visible=True, interactive=True), \
        file_end, \
        gr.Dropdown(choices=concat_choices), \
        gr.Dropdown(choices=concat_choices)
    else:
        return gr.Dropdown(choices=concat_choices,
        value=concat_choices[0]), \
        gr.Dropdown(visible=False), \
        file_end, \
        gr.Dropdown(choices=concat_choices), \
        gr.Dropdown(choices=concat_choices)

# Following function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
def add_folder_to_path(folder_path: str):
    '''
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist.
    '''

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ['PATH']
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ['PATH'] = full_path_extension
            #print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")

# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    return gr.Radio(visible=True), gr.Textbox(visible=True), gr.Button(visible=True), gr.Markdown(visible=True)

def wipe_logs(feedback_logs_loc:str, usage_logs_loc:str):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)
    
async def get_connection_params(request: gr.Request,
                                output_folder_textbox:str=OUTPUT_FOLDER,
                                input_folder_textbox:str=INPUT_FOLDER,
                                session_output_folder:str=SESSION_OUTPUT_FOLDER):

    #print("Session hash:", request.session_hash)

    if CUSTOM_HEADER and CUSTOM_HEADER_VALUE:
            if CUSTOM_HEADER in request.headers:
                supplied_custom_header_value = request.headers[CUSTOM_HEADER]
                if supplied_custom_header_value == CUSTOM_HEADER_VALUE:
                    print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
                else:
                    print("Custom header value does not match expected value.")
                    raise ValueError("Custom header value does not match expected value.")
            else:
                print("Custom header value not found.")
                raise ValueError("Custom header value not found.")   

    # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

    if request.username:
        out_session_hash = request.username
        #print("Request username found:", out_session_hash)

    elif 'x-cognito-id' in request.headers:
        out_session_hash = request.headers['x-cognito-id']
        #print("Cognito ID found:", out_session_hash)

    elif 'x-amzn-oidc-identity' in request.headers:
        out_session_hash = request.headers['x-amzn-oidc-identity']

        # Fetch email address using Cognito client
        cognito_client = boto3.client('cognito-idp')
        try:
            response = cognito_client.admin_get_user(
                UserPoolId=AWS_USER_POOL_ID,  # Replace with your User Pool ID
                Username=out_session_hash
            )
            email = next(attr['Value'] for attr in response['UserAttributes'] if attr['Name'] == 'email')
            #print("Email address found:", email)

            out_session_hash = email
        except ClientError as e:
            print("Error fetching user details:", e)
            email = None

        print("Cognito ID found:", out_session_hash)

    else:
        out_session_hash = request.session_hash

    if session_output_folder == 'True' or session_output_folder == True:
        output_folder = output_folder_textbox + out_session_hash + "/"
        input_folder = input_folder_textbox + out_session_hash + "/"
    else:
        output_folder = output_folder_textbox
        input_folder = input_folder_textbox

    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(input_folder): os.mkdir(input_folder)

    return out_session_hash, output_folder, out_session_hash, input_folder

def load_in_default_cost_codes(cost_codes_path:str, default_cost_code:str=""):
    '''
    Load in the cost codes list from file.
    '''
    cost_codes_df = pd.read_csv(cost_codes_path)
    dropdown_choices = cost_codes_df.iloc[:, 0].astype(str).tolist()

    # Avoid inserting duplicate or empty cost code values
    if default_cost_code and default_cost_code not in dropdown_choices:
        dropdown_choices.insert(0, default_cost_code)

    # Always have a blank option at the top
    if "" not in dropdown_choices:
        dropdown_choices.insert(0, "")

    out_dropdown = gr.Dropdown(
        value=default_cost_code if default_cost_code in dropdown_choices else "",
        label="Choose cost code for analysis",
        choices=dropdown_choices,
        allow_custom_value=False
    )
    
    return cost_codes_df, cost_codes_df, out_dropdown

def update_cost_code_dataframe_from_dropdown_select(cost_dropdown_selection:str, cost_code_df:pd.DataFrame):
    cost_code_df = cost_code_df.loc[cost_code_df.iloc[:,0] == cost_dropdown_selection, :]
    return cost_code_df

def df_select_callback_cost(df: pd.DataFrame, evt: gr.SelectData):
    row_value_code = evt.row_value[0] # This is the value for cost code

    return row_value_code

def update_cost_code_dataframe_from_dropdown_select(cost_dropdown_selection:str, cost_code_df:pd.DataFrame):
    cost_code_df = cost_code_df.loc[cost_code_df.iloc[:,0] == cost_dropdown_selection, :]
    return cost_code_df

def reset_base_dataframe(df:pd.DataFrame):
    return df

def enforce_cost_codes(enforce_cost_code_textbox:str, cost_code_choice:str, cost_code_df:pd.DataFrame, verify_cost_codes:bool=True):
    '''
    Check if the enforce cost codes variable is set to true, and then check that a cost cost has been chosen. If not, raise an error. Then, check against the values in the cost code dataframe to ensure that the cost code exists.
    '''

    if enforce_cost_code_textbox == "True":
        if not cost_code_choice:
            raise Exception("Please choose a cost code before continuing")
        
        if verify_cost_codes == True:
            if cost_code_df.empty:
                # Warn but don't block - cost code is still required above
                print("Warning: Cost code dataframe is empty. Verification skipped. Please ensure cost codes are loaded for full validation.")
            else:
                valid_cost_codes_list = list(cost_code_df.iloc[:,0].unique())

                if not cost_code_choice in valid_cost_codes_list:
                    raise Exception("Selected cost code not found in list. Please contact Finance if you cannot find the correct cost code from the given list of suggestions.")
    return

def _get_env_list(env_var_name: str, strip_strings:bool=True) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('\"', '').replace("\'","")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    if strip_strings:
        return [s.strip() for s in value.split(',') if s.strip()]
    else:
        return [codecs.decode(s, 'unicode_escape') for s in value.split(',') if s]

def create_batch_file_path_details(reference_data_file_name: str, latest_batch_completed:int=None, batch_size_number:int=None, in_column:str=None) -> str:
            """
            Creates a standardised batch file path detail string from a reference data filename.
            
            Args:
                reference_data_file_name (str): Name of the reference data file
                latest_batch_completed (int, optional): Latest batch completed. Defaults to None.
                batch_size_number (int, optional): Batch size number. Defaults to None.
                in_column (str, optional): In column. Defaults to None.
            Returns:
                str: Formatted batch file path detail string
            """
            
            # Extract components from filename using regex
            file_name = re.search(r'(.*?)(?:_all_|_final_|_batch_|_col_)', reference_data_file_name).group(1) if re.search(r'(.*?)(?:_all_|_final_|_batch_|_col_)', reference_data_file_name) else reference_data_file_name
            latest_batch_completed = int(re.search(r'batch_(\d+)_', reference_data_file_name).group(1)) if 'batch_' in reference_data_file_name else latest_batch_completed
            batch_size_number = int(re.search(r'size_(\d+)_', reference_data_file_name).group(1)) if 'size_' in reference_data_file_name else batch_size_number
            in_column = re.search(r'col_(.*?)_reference', reference_data_file_name).group(1) if 'col_' in reference_data_file_name else in_column

            # Clean the extracted names
            file_name_cleaned = clean_column_name(file_name, max_length=20)
            in_column_cleaned = clean_column_name(in_column, max_length=20)

            # Create batch file path details string
            if latest_batch_completed:
                return f"{file_name_cleaned}_batch_{latest_batch_completed}_size_{batch_size_number}_col_{in_column_cleaned}"
            return f"{file_name_cleaned}_col_{in_column_cleaned}"


def move_overall_summary_output_files_to_front_page(overall_summary_output_files_xlsx:List[str]):
    return overall_summary_output_files_xlsx

def generate_zero_shot_topics_df(zero_shot_topics:pd.DataFrame,
                                 force_zero_shot_radio:str="No",
                                 create_revised_general_topics:bool=False,
                                 max_topic_no:int=MAXIMUM_ZERO_SHOT_TOPICS):
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
        out_message = "Maximum " + str(max_topic_no) + " zero-shot topics allowed according to application configuration."
        print(out_message)
        raise Exception(out_message)        

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
                .str.replace('&', ' and ')
                .str.replace(' s ', 's ')
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

        # Filter out duplicate General topic and subtopic names
        zero_shot_topics_df = zero_shot_topics_df.drop_duplicates(["General topic", "Subtopic"], keep="first")

        # Sort the dataframe by General topic and subtopic
        zero_shot_topics_df = zero_shot_topics_df.sort_values(["General topic", "Subtopic"], ascending=[True, True])

        return zero_shot_topics_df

def update_model_choice(model_source):
    # Filter models by source and return the first matching model name
    matching_models = [model_name for model_name, model_info in model_name_map.items() 
                    if model_info["source"] == model_source]
                    
    output_model = matching_models[0] if matching_models else model_full_names[0]

    return gr.Dropdown(value = output_model, choices = matching_models, label="Large language model for topic extraction and summarisation", multiselect=False)