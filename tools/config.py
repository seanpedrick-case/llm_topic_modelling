import os
import tempfile
import socket
import logging
from datetime import datetime
from dotenv import load_dotenv

today_rev = datetime.now().strftime("%Y%m%d")
HOST_NAME = socket.gethostname()

# Set or retrieve configuration variables for the redaction app

def get_or_create_env_var(var_name:str, default_value:str, print_val:bool=False):
    '''
    Get an environmental variable, and set it to a default value if it doesn't exist
    '''
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set the environment variable to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    if print_val == True:
        print(f'The value of {var_name} is {value}')
    
    return value

def ensure_folder_exists(output_folder:str):
    """Checks if the specified folder exists, creates it if not."""   

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")

def add_folder_to_path(folder_path: str):
    '''
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
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


###
# LOAD CONFIG FROM ENV FILE
###

CONFIG_FOLDER = get_or_create_env_var('CONFIG_FOLDER', 'config/')

ensure_folder_exists(CONFIG_FOLDER)

# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. 'config/app_config.env'
APP_CONFIG_PATH = get_or_create_env_var('APP_CONFIG_PATH', CONFIG_FOLDER + 'app_config.env') # e.g. config/app_config.env

if APP_CONFIG_PATH:
    if os.path.exists(APP_CONFIG_PATH):
        print(f"Loading app variables from config file {APP_CONFIG_PATH}")
        load_dotenv(APP_CONFIG_PATH)
    else: print("App config file not found at location:", APP_CONFIG_PATH)

###
# AWS OPTIONS
###

# If you have an aws_config env file in the config folder, you can load in AWS keys this way, e.g. 'env/aws_config.env'
AWS_CONFIG_PATH = get_or_create_env_var('AWS_CONFIG_PATH', '') # e.g. config/aws_config.env

if AWS_CONFIG_PATH:
    if os.path.exists(AWS_CONFIG_PATH):
        print(f"Loading AWS variables from config file {AWS_CONFIG_PATH}")
        load_dotenv(AWS_CONFIG_PATH)
    else: print("AWS config file not found at location:", AWS_CONFIG_PATH)

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "1")

AWS_REGION = get_or_create_env_var('AWS_REGION', '')

AWS_CLIENT_ID = get_or_create_env_var('AWS_CLIENT_ID', '')

AWS_CLIENT_SECRET = get_or_create_env_var('AWS_CLIENT_SECRET', '')

AWS_USER_POOL_ID = get_or_create_env_var('AWS_USER_POOL_ID', '')

AWS_ACCESS_KEY = get_or_create_env_var('AWS_ACCESS_KEY', '')
if AWS_ACCESS_KEY: print(f'AWS_ACCESS_KEY found in environment variables')

AWS_SECRET_KEY = get_or_create_env_var('AWS_SECRET_KEY', '')
if AWS_SECRET_KEY: print(f'AWS_SECRET_KEY found in environment variables')

CONSULTATION_SUMMARY_BUCKET = get_or_create_env_var('CONSULTATION_SUMMARY_BUCKET', '')

# Custom headers e.g. if routing traffic through Cloudfront
# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var('CUSTOM_HEADER', '')

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var('CUSTOM_HEADER_VALUE', '')

###
# File I/O
###
SESSION_OUTPUT_FOLDER = get_or_create_env_var('SESSION_OUTPUT_FOLDER', 'False') # i.e. do you want your input and output folders saved within a subfolder based on session hash value within output/input folders 

OUTPUT_FOLDER = get_or_create_env_var('GRADIO_OUTPUT_FOLDER', 'output/') # 'output/'
INPUT_FOLDER = get_or_create_env_var('GRADIO_INPUT_FOLDER', 'input/') # 'input/'

ensure_folder_exists(OUTPUT_FOLDER)
ensure_folder_exists(INPUT_FOLDER)

# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP": 
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f'Temporary directory created at: {temp_dir}')

        if OUTPUT_FOLDER == "TEMP": OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP": INPUT_FOLDER = temp_dir + "/"


GRADIO_TEMP_DIR = get_or_create_env_var('GRADIO_TEMP_DIR', 'tmp/gradio_tmp/') # Default Gradio temp folder
MPLCONFIGDIR = get_or_create_env_var('MPLCONFIGDIR', 'tmp/matplotlib_cache/') # Matplotlib cache folder

ensure_folder_exists(GRADIO_TEMP_DIR)
ensure_folder_exists(MPLCONFIGDIR)

# TLDEXTRACT_CACHE = get_or_create_env_var('TLDEXTRACT_CACHE', 'tmp/tld/')
# try:
#     extract = TLDExtract(cache_dir=TLDEXTRACT_CACHE)
# except:
#     extract = TLDExtract(cache_dir=None)

###
# LOGGING OPTIONS
###

# By default, logs are put into a subfolder of today's date and the host name of the instance running the app. This is to avoid at all possible the possibility of log files from one instance overwriting the logs of another instance on S3. If running the app on one system always, or just locally, it is not necessary to make the log folders so specific.
# Another way to address this issue would be to write logs to another type of storage, e.g. database such as dynamodb. I may look into this in future.

SAVE_LOGS_TO_CSV = get_or_create_env_var('SAVE_LOGS_TO_CSV', 'True')

USE_LOG_SUBFOLDERS = get_or_create_env_var('USE_LOG_SUBFOLDERS', 'True')

if USE_LOG_SUBFOLDERS == "True":
    day_log_subfolder = today_rev + '/'
    host_name_subfolder = HOST_NAME + '/'
    full_log_subfolder = day_log_subfolder + host_name_subfolder
else:
    full_log_subfolder = ""

FEEDBACK_LOGS_FOLDER = get_or_create_env_var('FEEDBACK_LOGS_FOLDER', 'feedback/' + full_log_subfolder)
ACCESS_LOGS_FOLDER = get_or_create_env_var('ACCESS_LOGS_FOLDER', 'logs/' + full_log_subfolder)
USAGE_LOGS_FOLDER = get_or_create_env_var('USAGE_LOGS_FOLDER', 'usage/' + full_log_subfolder)

ensure_folder_exists(FEEDBACK_LOGS_FOLDER)
ensure_folder_exists(ACCESS_LOGS_FOLDER)
ensure_folder_exists(USAGE_LOGS_FOLDER)

# Should the redacted file name be included in the logs? In some instances, the names of the files themselves could be sensitive, and should not be disclosed beyond the app. So, by default this is false.
DISPLAY_FILE_NAMES_IN_LOGS = get_or_create_env_var('DISPLAY_FILE_NAMES_IN_LOGS', 'False')

# Further customisation options for CSV logs

CSV_ACCESS_LOG_HEADERS = get_or_create_env_var('CSV_ACCESS_LOG_HEADERS', '') # If blank, uses component labels
CSV_FEEDBACK_LOG_HEADERS = get_or_create_env_var('CSV_FEEDBACK_LOG_HEADERS', '') # If blank, uses component labels
CSV_USAGE_LOG_HEADERS = get_or_create_env_var('CSV_USAGE_LOG_HEADERS', '["session_hash_textbox", "doc_full_file_name_textbox", "data_full_file_name_textbox", "actual_time_taken_number",	"total_page_count",	"textract_query_number", "pii_detection_method", "comprehend_query_number",  "cost_code", "textract_handwriting_signature", "host_name_textbox", "text_extraction_method", "is_this_a_textract_api_call"]') # If blank, uses component labels

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table

SAVE_LOGS_TO_DYNAMODB = get_or_create_env_var('SAVE_LOGS_TO_DYNAMODB', 'False')

ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('ACCESS_LOG_DYNAMODB_TABLE_NAME', 'redaction_access_log')
DYNAMODB_ACCESS_LOG_HEADERS = get_or_create_env_var('DYNAMODB_ACCESS_LOG_HEADERS', '')

FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('FEEDBACK_LOG_DYNAMODB_TABLE_NAME', 'redaction_feedback')
DYNAMODB_FEEDBACK_LOG_HEADERS = get_or_create_env_var('DYNAMODB_FEEDBACK_LOG_HEADERS', '')

USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('USAGE_LOG_DYNAMODB_TABLE_NAME', 'redaction_usage')
DYNAMODB_USAGE_LOG_HEADERS = get_or_create_env_var('DYNAMODB_USAGE_LOG_HEADERS', '')

# Report logging to console?
LOGGING = get_or_create_env_var('LOGGING', 'False')

if LOGGING == 'True':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOG_FILE_NAME = get_or_create_env_var('LOG_FILE_NAME', 'log.csv')

###
# LLM variables
###

MAX_TOKENS = int(get_or_create_env_var('MAX_TOKENS', '4096')) # Maximum number of output tokens
TIMEOUT_WAIT = int(get_or_create_env_var('TIMEOUT_WAIT', '30')) # AWS now seems to have a 60 second minimum wait between API calls
NUMBER_OF_RETRY_ATTEMPTS = int(get_or_create_env_var('NUMBER_OF_RETRY_ATTEMPTS', '5'))
# Try up to 3 times to get a valid markdown table response with LLM calls, otherwise retry with temperature changed
MAX_OUTPUT_VALIDATION_ATTEMPTS = int(get_or_create_env_var('MAX_OUTPUT_VALIDATION_ATTEMPTS', '3'))
MAX_TIME_FOR_LOOP = int(get_or_create_env_var('MAX_TIME_FOR_LOOP', '99999'))
BATCH_SIZE_DEFAULT = int(get_or_create_env_var('BATCH_SIZE_DEFAULT', '5'))
DEDUPLICATION_THRESHOLD = int(get_or_create_env_var('DEDUPLICATION_THRESHOLD', '90'))
MAX_COMMENT_CHARS = int(get_or_create_env_var('MAX_COMMENT_CHARS', '14000'))

RUN_LOCAL_MODEL = get_or_create_env_var("RUN_LOCAL_MODEL", "1")
RUN_GEMINI_MODELS = get_or_create_env_var("RUN_GEMINI_MODELS", "1")
GEMINI_API_KEY = get_or_create_env_var('GEMINI_API_KEY', '')

# Build up options for models

model_full_names = []
model_short_names = []

CHOSEN_LOCAL_MODEL_TYPE = get_or_create_env_var("CHOSEN_LOCAL_MODEL_TYPE", "Gemma 2b") # Gemma 3 1B #  "Gemma 2b"

if RUN_LOCAL_MODEL == "1" and CHOSEN_LOCAL_MODEL_TYPE:
    model_full_names.append(CHOSEN_LOCAL_MODEL_TYPE)
    model_short_names.append(CHOSEN_LOCAL_MODEL_TYPE)

if RUN_AWS_FUNCTIONS == "1":
    model_full_names.extend(["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"])
    model_short_names.extend(["haiku", "sonnet"])

if RUN_GEMINI_MODELS == "1":
    model_full_names.extend(["gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-exp-05-06" ]) # , # Gemini pro No longer available on free tier
    model_short_names.extend(["gemini_flash_lite_2.5", "gemini_flash_2.5", "gemini_pro"])

print("model_short_names:", model_short_names)
print("model_full_names:", model_full_names)

model_name_map = {short: full for short, full in zip(model_full_names, model_short_names)}

# HF token may or may not be needed for downloading models from Hugging Face
HF_TOKEN = get_or_create_env_var('HF_TOKEN', '')

GEMMA2_REPO_ID = get_or_create_env_var("GEMMA2_2B_REPO_ID", "lmstudio-community/gemma-2-2b-it-GGUF")# "bartowski/Llama-3.2-3B-Instruct-GGUF") # "lmstudio-community/gemma-2-2b-it-GGUF")#"QuantFactory/Phi-3-mini-128k-instruct-GGUF")
GEMMA2_MODEL_FILE = get_or_create_env_var("GEMMA2_2B_MODEL_FILE", "gemma-2-2b-it-Q8_0.gguf") # )"Llama-3.2-3B-Instruct-Q5_K_M.gguf") #"gemma-2-2b-it-Q8_0.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")
GEMMA2_MODEL_FOLDER = get_or_create_env_var("GEMMA2_2B_MODEL_FOLDER", "model/gemma") #"model/phi"  # Assuming this is your intended directory

GEMMA3_REPO_ID = get_or_create_env_var("GEMMA3_REPO_ID", "ggml-org/gemma-3-1b-it-GGUF")# "bartowski/Llama-3.2-3B-Instruct-GGUF") # "lmstudio-community/gemma-2-2b-it-GGUF")#"QuantFactory/Phi-3-mini-128k-instruct-GGUF")
GEMMA3_MODEL_FILE = get_or_create_env_var("GEMMA3_MODEL_FILE", "gemma-3-1b-it-Q8_0.gguf") # )"Llama-3.2-3B-Instruct-Q5_K_M.gguf") #"gemma-2-2b-it-Q8_0.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")
GEMMA3_MODEL_FOLDER = get_or_create_env_var("GEMMA3_MODEL_FOLDER", "model/gemma")

GEMMA3_4B_REPO_ID = get_or_create_env_var("GEMMA3_4B_REPO_ID", "ggml-org/gemma-3-4b-it-GGUF")# "bartowski/Llama-3.2-3B-Instruct-GGUF") # "lmstudio-community/gemma-2-2b-it-GGUF")#"QuantFactory/Phi-3-mini-128k-instruct-GGUF")
GEMMA3_4B_MODEL_FILE = get_or_create_env_var("GEMMA3_4B_MODEL_FILE", "gemma-3-4b-it-Q4_K_M.gguf") # )"Llama-3.2-3B-Instruct-Q5_K_M.gguf") #"gemma-2-2b-it-Q8_0.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")
GEMMA3_4B_MODEL_FOLDER = get_or_create_env_var("GEMMA3_4B_MODEL_FOLDER", "model/gemma3_4b")


if CHOSEN_LOCAL_MODEL_TYPE == "Gemma 2b":
    LOCAL_REPO_ID = GEMMA2_REPO_ID
    LOCAL_MODEL_FILE = GEMMA2_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA2_MODEL_FOLDER

elif CHOSEN_LOCAL_MODEL_TYPE == "Gemma 3 1B":
    LOCAL_REPO_ID = GEMMA3_REPO_ID
    LOCAL_MODEL_FILE = GEMMA3_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA3_MODEL_FOLDER

elif CHOSEN_LOCAL_MODEL_TYPE == "Gemma 3 4B":
    LOCAL_REPO_ID = GEMMA3_4B_REPO_ID
    LOCAL_MODEL_FILE = GEMMA3_4B_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA3_4B_MODEL_FOLDER

    print("CHOSEN_LOCAL_MODEL_TYPE:", CHOSEN_LOCAL_MODEL_TYPE)
    print("LOCAL_REPO_ID:", LOCAL_REPO_ID)
    print("LOCAL_MODEL_FILE:", LOCAL_MODEL_FILE)
    print("LOCAL_MODEL_FOLDER:", LOCAL_MODEL_FOLDER)

LLM_MAX_GPU_LAYERS = int(get_or_create_env_var('MAX_GPU_LAYERS','-1'))
LLM_TEMPERATURE = float(get_or_create_env_var('LLM_TEMPERATURE', '0.1'))
LLM_TOP_K = int(get_or_create_env_var('LLM_TOP_K','3'))
LLM_TOP_P = float(get_or_create_env_var('LLM_TOP_P', '1'))
LLM_REPETITION_PENALTY = float(get_or_create_env_var('LLM_REPETITION_PENALTY', '1.2')) # Mild repetition penalty to prevent repeating table rows
LLM_LAST_N_TOKENS = int(get_or_create_env_var('LLM_LAST_N_TOKENS', '512'))
LLM_MAX_NEW_TOKENS = int(get_or_create_env_var('LLM_MAX_NEW_TOKENS', '4096'))
LLM_SEED = int(get_or_create_env_var('LLM_SEED', '42'))
LLM_RESET = get_or_create_env_var('LLM_RESET', 'True')
LLM_STREAM = get_or_create_env_var('LLM_STREAM', 'False')
LLM_THREADS = int(get_or_create_env_var('LLM_THREADS', '4'))
LLM_BATCH_SIZE = int(get_or_create_env_var('LLM_BATCH_SIZE', '256'))
LLM_CONTEXT_LENGTH = int(get_or_create_env_var('LLM_CONTEXT_LENGTH', '16384'))
LLM_SAMPLE = get_or_create_env_var('LLM_SAMPLE', 'True')

###
# Gradio app variables
###

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')

RUN_DIRECT_MODE = get_or_create_env_var('RUN_DIRECT_MODE', '0')

MAX_QUEUE_SIZE = int(get_or_create_env_var('MAX_QUEUE_SIZE', '5'))

MAX_FILE_SIZE = get_or_create_env_var('MAX_FILE_SIZE', '250mb')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7860'))

ROOT_PATH = get_or_create_env_var('ROOT_PATH', '')

DEFAULT_CONCURRENCY_LIMIT = get_or_create_env_var('DEFAULT_CONCURRENCY_LIMIT', '3')

GET_DEFAULT_ALLOW_LIST = get_or_create_env_var('GET_DEFAULT_ALLOW_LIST', '')

ALLOW_LIST_PATH = get_or_create_env_var('ALLOW_LIST_PATH', '') # config/default_allow_list.csv

S3_ALLOW_LIST_PATH = get_or_create_env_var('S3_ALLOW_LIST_PATH', '') # default_allow_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

if ALLOW_LIST_PATH: OUTPUT_ALLOW_LIST_PATH = ALLOW_LIST_PATH
else: OUTPUT_ALLOW_LIST_PATH = 'config/default_allow_list.csv'

FILE_INPUT_HEIGHT = get_or_create_env_var('FILE_INPUT_HEIGHT', '200')
