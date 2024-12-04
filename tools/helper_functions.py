import os
import gradio as gr
import pandas as pd



def get_or_create_env_var(var_name, default_value):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "0")
print(f'The value of RUN_AWS_FUNCTIONS is {RUN_AWS_FUNCTIONS}')

if RUN_AWS_FUNCTIONS == "1":
    model_full_names = ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]
    model_short_names = ["haiku", "sonnet", "gemini_flash", "gemini_pro"]
else:
    model_full_names = ["gemini-1.5-flash-002", "gemini-1.5-pro-002"]
    model_short_names = ["gemini_flash", "gemini_pro"]

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

def get_file_path_end(file_path):
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

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def view_table(file_path: str, max_width: int = 60):  # Added max_width parameter
    df = pd.read_csv(file_path)

    df_cleaned = df.replace('\n', ' ', regex=True)

    # Wrap text in each column to the specified max width, including whole words
    def wrap_text(text):
        if isinstance(text, str):
            words = text.split(' ')
            wrapped_lines = []
            current_line = ""

            for word in words:
                # Check if adding the next word exceeds the max width
                if len(current_line) + len(word) + 1 > max_width:  # +1 for the space
                    wrapped_lines.append(current_line)
                    current_line = word  # Start a new line with the current word
                else:
                    if current_line:  # If current_line is not empty, add a space
                        current_line += ' '
                    current_line += word

            # Add any remaining text in current_line to wrapped_lines
            if current_line:
                wrapped_lines.append(current_line)

            return '<br>'.join(wrapped_lines)  # Join lines with <br>
        return text

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

def put_columns_in_df(in_file):
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

                # Process the DataFrame (e.g., print its contents)
                print(f"Sheet Name: {sheet_name}")
                print(df.head())  # Print the first few rows

                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        else:
            df = read_file(file_name)
            new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        
    # Drop duplicate columns
    concat_choices = list(set(concat_choices))

    if number_of_excel_files > 0:      
        return gr.Dropdown(choices=concat_choices, value=concat_choices[0]), gr.Dropdown(choices=all_sheet_names, value=all_sheet_names[0], visible=True), file_end
    else:
        return gr.Dropdown(choices=concat_choices, value=concat_choices[0]), gr.Dropdown(visible=False), file_end

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