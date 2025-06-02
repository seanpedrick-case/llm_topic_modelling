import os
import re
import gradio as gr
import pandas as pd
from typing import List
import math

def empty_output_vars_extract_topics():
    # Empty output objects before processing a new file

    master_topic_df_state = pd.DataFrame()
    master_unique_topics_df_state = pd.DataFrame()
    master_reference_df_state = pd.DataFrame()
    text_output_file = []
    text_output_file_list_state = []
    latest_batch_completed = 0
    log_files_output = []
    log_files_output_list_state = []
    conversation_metadata_textbox = ""
    estimated_time_taken_number = 0
    file_data_state = pd.DataFrame()
    reference_data_file_name_textbox = ""
    display_topic_table_markdown = ""

    return master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, text_output_file, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, reference_data_file_name_textbox, display_topic_table_markdown

def empty_output_vars_summarise():
    # Empty output objects before summarising files

    summary_reference_table_sample_state = pd.DataFrame()
    master_unique_topics_df_revised_summaries_state = pd.DataFrame()
    master_reference_df_revised_summaries_state = pd.DataFrame()
    summary_output_files = []
    summarised_outputs_list = []
    latest_summary_completed_num = 0
    conversation_metadata_textbox = ""

    return summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox


def get_or_create_env_var(var_name, default_value):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "1")
print(f'The value of RUN_AWS_FUNCTIONS is {RUN_AWS_FUNCTIONS}')

RUN_LOCAL_MODEL = get_or_create_env_var("RUN_LOCAL_MODEL", "1")
print(f'The value of RUN_LOCAL_MODEL is {RUN_LOCAL_MODEL}')

RUN_GEMINI_MODELS = get_or_create_env_var("RUN_GEMINI_MODELS", "1")
print(f'The value of RUN_GEMINI_MODELS is {RUN_GEMINI_MODELS}')

GEMINI_API_KEY = get_or_create_env_var('GEMINI_API_KEY', '')

# Build up options for models
model_full_names = []
model_short_names = []

if RUN_LOCAL_MODEL == "1":
    model_full_names.append("gemma_2b_it_local")
    model_short_names.append("gemma_local")

if RUN_AWS_FUNCTIONS == "1":
    model_full_names.extend(["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"])
    model_short_names.extend(["haiku", "sonnet"])

if RUN_GEMINI_MODELS == "1":
    model_full_names.extend(["gemini-2.0-flash-001", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-exp-05-06" ]) # , # Gemini pro No longer available on free tier
    model_short_names.extend(["gemini_flash_2", "gemini_flash_2.5", "gemini_pro"])

print("model_short_names:", model_short_names)
print("model_full_names:", model_full_names)

model_name_map = {short: full for short, full in zip(model_full_names, model_short_names)}

# Retrieving or setting output folder
env_var_name = 'GRADIO_OUTPUT_FOLDER'
default_value = 'output/'

output_folder = get_or_create_env_var(env_var_name, default_value)
print(f'The value of {env_var_name} is {output_folder}')

def get_file_path_with_extension(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Return the basename with its extension
    return basename

def get_file_name_no_ext(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def detect_file_type(filename):
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

    #file_type = detect_file_type(file_path)
    #print("File type is:", file_type)

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

def load_in_data_file(file_paths:List[str], in_colnames:List[str], batch_size:int=50, in_excel_sheets:str=""):
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

def join_cols_onto_reference_df(reference_df:pd.DataFrame, original_data_df:pd.DataFrame, join_columns:List[str], original_file_name:str, output_folder:str=output_folder):

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

# Wrap text in each column to the specified max width, including whole words
def wrap_text(text:str, max_width=60, max_text_length=None):
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
    wrapped_lines = []
    current_line = []
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
            words_to_bring_back = []
            remaining_words = []
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

def initial_clean(text):
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
    new_choices = []
    concat_choices = []
    all_sheet_names = []
    number_of_excel_files = 0
    
    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        #print("File type is:", file_type)

        file_end = get_file_path_with_extension(file_name)

        if file_type == 'xlsx':
            number_of_excel_files += 1
            new_choices = []
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
        return gr.Dropdown(choices=concat_choices, value=concat_choices[0]), gr.Dropdown(choices=all_sheet_names, value=all_sheet_names[0], visible=True, interactive=True), file_end, gr.Dropdown(choices=concat_choices)
    else:
        return gr.Dropdown(choices=concat_choices, value=concat_choices[0]), gr.Dropdown(visible=False), file_end, gr.Dropdown(choices=concat_choices)

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

def wipe_logs(feedback_logs_loc, usage_logs_loc):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)
   
async def get_connection_params(request: gr.Request):
    base_folder = ""

    if request:
        #print("request user:", request.username)

        #request_data = await request.json()  # Parse JSON body
        #print("All request data:", request_data)
        #context_value = request_data.get('context') 
        #if 'context' in request_data:
        #     print("Request context dictionary:", request_data['context'])

        # print("Request headers dictionary:", request.headers)
        # print("All host elements", request.client)           
        # print("IP address:", request.client.host)
        # print("Query parameters:", dict(request.query_params))
        # To get the underlying FastAPI items you would need to use await and some fancy @ stuff for a live query: https://fastapi.tiangolo.com/vi/reference/request/
        #print("Request dictionary to object:", request.request.body())
        print("Session hash:", request.session_hash)

        # Retrieving or setting CUSTOM_CLOUDFRONT_HEADER
        CUSTOM_CLOUDFRONT_HEADER_var = get_or_create_env_var('CUSTOM_CLOUDFRONT_HEADER', '')
        #print(f'The value of CUSTOM_CLOUDFRONT_HEADER is {CUSTOM_CLOUDFRONT_HEADER_var}')

        # Retrieving or setting CUSTOM_CLOUDFRONT_HEADER_VALUE
        CUSTOM_CLOUDFRONT_HEADER_VALUE_var = get_or_create_env_var('CUSTOM_CLOUDFRONT_HEADER_VALUE', '')
        #print(f'The value of CUSTOM_CLOUDFRONT_HEADER_VALUE_var is {CUSTOM_CLOUDFRONT_HEADER_VALUE_var}')

        if CUSTOM_CLOUDFRONT_HEADER_var and CUSTOM_CLOUDFRONT_HEADER_VALUE_var:
            if CUSTOM_CLOUDFRONT_HEADER_var in request.headers:
                supplied_cloudfront_custom_value = request.headers[CUSTOM_CLOUDFRONT_HEADER_var]
                if supplied_cloudfront_custom_value == CUSTOM_CLOUDFRONT_HEADER_VALUE_var:
                    print("Custom Cloudfront header found:", supplied_cloudfront_custom_value)
                else:
                    raise(ValueError, "Custom Cloudfront header value does not match expected value.")

        # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

        if request.username:
            out_session_hash = request.username
            base_folder = "user-files/"
            print("Request username found:", out_session_hash)

        elif 'x-cognito-id' in request.headers:
            out_session_hash = request.headers['x-cognito-id']
            base_folder = "user-files/"
            print("Cognito ID found:", out_session_hash)

        else:
            out_session_hash = request.session_hash
            base_folder = "temp-files/"
            # print("Cognito ID not found. Using session hash as save folder:", out_session_hash)

        output_folder = base_folder + out_session_hash + "/"
        #if bucket_name:
        #    print("S3 output folder is: " + "s3://" + bucket_name + "/" + output_folder)

        return out_session_hash, output_folder, out_session_hash
    else:
        print("No session parameters found.")
        return "",""