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
#if AWS_ACCESS_KEY: print(f'AWS_ACCESS_KEY found in environment variables')

AWS_SECRET_KEY = get_or_create_env_var('AWS_SECRET_KEY', '')
#if AWS_SECRET_KEY: print(f'AWS_SECRET_KEY found in environment variables')

# Should the app prioritise using AWS SSO over using API keys stored in environment variables/secrets (defaults to yes)
PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS = get_or_create_env_var('PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS', '1')

S3_LOG_BUCKET = get_or_create_env_var('S3_LOG_BUCKET', '')

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


# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP": 
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f'Temporary directory created at: {temp_dir}')

        if OUTPUT_FOLDER == "TEMP": OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP": INPUT_FOLDER = temp_dir + "/"


GRADIO_TEMP_DIR = get_or_create_env_var('GRADIO_TEMP_DIR', 'tmp/gradio_tmp/') # Default Gradio temp folder
MPLCONFIGDIR = get_or_create_env_var('MPLCONFIGDIR', 'tmp/matplotlib_cache/') # Matplotlib cache folder

###
# LOGGING OPTIONS
###

# By default, logs are put into a subfolder of today's date and the host name of the instance running the app. This is to avoid at all possible the possibility of log files from one instance overwriting the logs of another instance on S3. If running the app on one system always, or just locally, it is not necessary to make the log folders so specific.
# Another way to address this issue would be to write logs to another type of storage, e.g. database such as dynamodb. I may look into this in future.

SAVE_LOGS_TO_CSV = get_or_create_env_var('SAVE_LOGS_TO_CSV', 'True')

USE_LOG_SUBFOLDERS = get_or_create_env_var('USE_LOG_SUBFOLDERS', 'True')

FEEDBACK_LOGS_FOLDER = get_or_create_env_var('FEEDBACK_LOGS_FOLDER', 'feedback/')
ACCESS_LOGS_FOLDER = get_or_create_env_var('ACCESS_LOGS_FOLDER', 'logs/')
USAGE_LOGS_FOLDER = get_or_create_env_var('USAGE_LOGS_FOLDER', 'usage/')

if USE_LOG_SUBFOLDERS == "True":
    day_log_subfolder = today_rev + '/'
    host_name_subfolder = HOST_NAME + '/'
    full_log_subfolder = day_log_subfolder + host_name_subfolder

    FEEDBACK_LOGS_FOLDER = FEEDBACK_LOGS_FOLDER + full_log_subfolder
    ACCESS_LOGS_FOLDER = ACCESS_LOGS_FOLDER + full_log_subfolder
    USAGE_LOGS_FOLDER = USAGE_LOGS_FOLDER + full_log_subfolder

S3_FEEDBACK_LOGS_FOLDER = get_or_create_env_var('S3_FEEDBACK_LOGS_FOLDER', 'feedback/' + full_log_subfolder)
S3_ACCESS_LOGS_FOLDER = get_or_create_env_var('S3_ACCESS_LOGS_FOLDER', 'logs/' + full_log_subfolder)
S3_USAGE_LOGS_FOLDER = get_or_create_env_var('S3_USAGE_LOGS_FOLDER', 'usage/' + full_log_subfolder)

LOG_FILE_NAME = get_or_create_env_var('LOG_FILE_NAME', 'log.csv')
USAGE_LOG_FILE_NAME = get_or_create_env_var('USAGE_LOG_FILE_NAME', LOG_FILE_NAME)
FEEDBACK_LOG_FILE_NAME = get_or_create_env_var('FEEDBACK_LOG_FILE_NAME', LOG_FILE_NAME)

# Should the redacted file name be included in the logs? In some instances, the names of the files themselves could be sensitive, and should not be disclosed beyond the app. So, by default this is false.
DISPLAY_FILE_NAMES_IN_LOGS = get_or_create_env_var('DISPLAY_FILE_NAMES_IN_LOGS', 'False')

# Further customisation options for CSV logs

CSV_ACCESS_LOG_HEADERS = get_or_create_env_var('CSV_ACCESS_LOG_HEADERS', '') # If blank, uses component labels
CSV_FEEDBACK_LOG_HEADERS = get_or_create_env_var('CSV_FEEDBACK_LOG_HEADERS', '') # If blank, uses component labels
CSV_USAGE_LOG_HEADERS = get_or_create_env_var('CSV_USAGE_LOG_HEADERS', '') # If blank, uses component labels

### DYNAMODB logs. Whether to save to DynamoDB, and the headers of the table
SAVE_LOGS_TO_DYNAMODB = get_or_create_env_var('SAVE_LOGS_TO_DYNAMODB', 'False')

ACCESS_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('ACCESS_LOG_DYNAMODB_TABLE_NAME', 'llm_topic_model_access_log')
DYNAMODB_ACCESS_LOG_HEADERS = get_or_create_env_var('DYNAMODB_ACCESS_LOG_HEADERS', '')

FEEDBACK_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('FEEDBACK_LOG_DYNAMODB_TABLE_NAME', 'llm_topic_model_feedback')
DYNAMODB_FEEDBACK_LOG_HEADERS = get_or_create_env_var('DYNAMODB_FEEDBACK_LOG_HEADERS', '')

USAGE_LOG_DYNAMODB_TABLE_NAME = get_or_create_env_var('USAGE_LOG_DYNAMODB_TABLE_NAME', 'llm_topic_model_usage')
DYNAMODB_USAGE_LOG_HEADERS = get_or_create_env_var('DYNAMODB_USAGE_LOG_HEADERS', '')

# Report logging to console?
LOGGING = get_or_create_env_var('LOGGING', 'False')

if LOGGING == 'True':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### 
# App run variables
###
OUTPUT_DEBUG_FILES = get_or_create_env_var('OUTPUT_DEBUG_FILES', 'True') # Whether to output debug files

TIMEOUT_WAIT = int(get_or_create_env_var('TIMEOUT_WAIT', '30')) # Maximum number of seconds to wait for a response from the LLM
NUMBER_OF_RETRY_ATTEMPTS = int(get_or_create_env_var('NUMBER_OF_RETRY_ATTEMPTS', '5')) # Maximum number of times to retry a request to the LLM
# Try up to 3 times to get a valid markdown table response with LLM calls, otherwise retry with temperature changed
MAX_OUTPUT_VALIDATION_ATTEMPTS = int(get_or_create_env_var('MAX_OUTPUT_VALIDATION_ATTEMPTS', '3'))
MAX_TIME_FOR_LOOP = int(get_or_create_env_var('MAX_TIME_FOR_LOOP', '99999')) # Maximum number of seconds to run the loop for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there)

MAX_COMMENT_CHARS = int(get_or_create_env_var('MAX_COMMENT_CHARS', '14000')) # Maximum number of characters in a comment
MAX_ROWS = int(get_or_create_env_var('MAX_ROWS', '5000')) # Maximum number of rows to process
MAX_GROUPS = int(get_or_create_env_var('MAX_GROUPS', '99')) # Maximum number of groups to process
BATCH_SIZE_DEFAULT = int(get_or_create_env_var('BATCH_SIZE_DEFAULT', '5')) # Default batch size for LLM calls
MAXIMUM_ZERO_SHOT_TOPICS = int(get_or_create_env_var('MAXIMUM_ZERO_SHOT_TOPICS', '120')) # Maximum number of zero shot topics to process
MAX_SPACES_GPU_RUN_TIME = int(get_or_create_env_var('MAX_SPACES_GPU_RUN_TIME', '240')) # Maximum number of seconds to run on GPU on Hugging Face Spaces

DEDUPLICATION_THRESHOLD = int(get_or_create_env_var('DEDUPLICATION_THRESHOLD', '90')) # Deduplication threshold for topic summary tables

###
# Model options
###

RUN_LOCAL_MODEL = get_or_create_env_var("RUN_LOCAL_MODEL", "1")

RUN_AWS_BEDROCK_MODELS = get_or_create_env_var("RUN_AWS_BEDROCK_MODELS", "1")

RUN_GEMINI_MODELS = get_or_create_env_var("RUN_GEMINI_MODELS", "1")
GEMINI_API_KEY = get_or_create_env_var('GEMINI_API_KEY', '')

# Azure AI Inference settings
RUN_AZURE_MODELS = get_or_create_env_var("RUN_AZURE_MODELS", "0")
AZURE_API_KEY = get_or_create_env_var('AZURE_API_KEY', '')
AZURE_INFERENCE_ENDPOINT = get_or_create_env_var('AZURE_INFERENCE_ENDPOINT', '')

# Build up options for models

model_full_names = list()
model_short_names = list()
model_source = list()

CHOSEN_LOCAL_MODEL_TYPE = get_or_create_env_var("CHOSEN_LOCAL_MODEL_TYPE", "Qwen 3 4B") # Gemma 3 1B #  "Gemma 2b" # "Gemma 3 4B"

if RUN_LOCAL_MODEL == "1" and CHOSEN_LOCAL_MODEL_TYPE:
    model_full_names.append(CHOSEN_LOCAL_MODEL_TYPE)
    model_short_names.append(CHOSEN_LOCAL_MODEL_TYPE)
    model_source.append("Local")

if RUN_AWS_BEDROCK_MODELS == "1":
    model_full_names.extend(["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-7-sonnet-20250219-v1:0", "amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"])
    model_short_names.extend(["haiku", "sonnet", "nova_micro", "nova_lite", "nova_pro"])
    model_source.extend(["AWS", "AWS", "AWS", "AWS", "AWS"])

if RUN_GEMINI_MODELS == "1":
    model_full_names.extend(["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"])
    model_short_names.extend(["gemini_flash_lite_2.5", "gemini_flash_2.5", "gemini_pro"])
    model_source.extend(["Gemini", "Gemini", "Gemini"])

# Register Azure AI models (model names must match your Azure deployments)
if RUN_AZURE_MODELS == "1":
    # Example deployments; adjust to the deployments you actually create in Azure
    model_full_names.extend(["gpt-5-mini"])
    model_short_names.extend(["gpt-5-mini"])
    model_source.extend(["Azure"])

model_name_map = {
    full: {"short_name": short, "source": source}
    for full, short, source in zip(model_full_names, model_short_names, model_source)
}

#print("model_name_map:", model_name_map)

# HF token may or may not be needed for downloading models from Hugging Face
HF_TOKEN = get_or_create_env_var('HF_TOKEN', '')

LOAD_LOCAL_MODEL_AT_START = get_or_create_env_var('LOAD_LOCAL_MODEL_AT_START', 'True')

# If you are using a system with low VRAM, you can set this to True to reduce the memory requirements
LOW_VRAM_SYSTEM = get_or_create_env_var('LOW_VRAM_SYSTEM', 'False')

if LOW_VRAM_SYSTEM == 'True':
    print("Changing settings for low VRAM system")
    USE_LLAMA_CPP = get_or_create_env_var('USE_LLAMA_CPP', 'True')
    LLM_MAX_NEW_TOKENS = int(get_or_create_env_var('LLM_MAX_NEW_TOKENS', '4096'))
    LLM_CONTEXT_LENGTH = int(get_or_create_env_var('LLM_CONTEXT_LENGTH', '8192'))
    LLM_BATCH_SIZE = int(get_or_create_env_var('LLM_BATCH_SIZE', '512'))
    KV_QUANT_LEVEL = int(get_or_create_env_var('KV_QUANT_LEVEL', '2')) # 2 is equivalent to q4_0, 8 is q8_0



USE_LLAMA_CPP = get_or_create_env_var('USE_LLAMA_CPP', 'True') # Llama.cpp or transformers with unsloth

GEMMA2_REPO_ID = get_or_create_env_var("GEMMA2_2B_REPO_ID", "unsloth/gemma-2-it-GGUF")
GEMMA2_REPO_TRANSFORMERS_ID = get_or_create_env_var("GEMMA2_2B_REPO_TRANSFORMERS_ID", "unsloth/gemma-2-2b-it-bnb-4bit")
if USE_LLAMA_CPP == "False":
    GEMMA2_REPO_ID = GEMMA2_REPO_TRANSFORMERS_ID

GEMMA2_MODEL_FILE = get_or_create_env_var("GEMMA2_2B_MODEL_FILE", "gemma-2-2b-it.q8_0.gguf")
GEMMA2_MODEL_FOLDER = get_or_create_env_var("GEMMA2_2B_MODEL_FOLDER", "model/gemma")

GEMMA3_REPO_ID = get_or_create_env_var("GEMMA3_REPO_ID", "unsloth/gemma-3-270m-it-qat-GGUF")
GEMMA3_REPO_TRANSFORMERS_ID = get_or_create_env_var("GEMMA3_REPO_TRANSFORMERS_ID", "unsloth/gemma-3-270m-it")
if USE_LLAMA_CPP == "False":
    GEMMA3_REPO_ID = GEMMA3_REPO_TRANSFORMERS_ID

GEMMA3_MODEL_FILE = get_or_create_env_var("GEMMA3_MODEL_FILE", "gemma-3-270m-it-qat-F16.gguf")
GEMMA3_MODEL_FOLDER = get_or_create_env_var("GEMMA3_MODEL_FOLDER", "model/gemma")

GEMMA3_4B_REPO_ID = get_or_create_env_var("GEMMA3_4B_REPO_ID", "unsloth/gemma-3-4b-it-qat-GGUF")
GEMMA3_4B_REPO_TRANSFORMERS_ID = get_or_create_env_var("GEMMA3_4B_REPO_TRANSFORMERS_ID", "unsloth/gemma-3-4b-it-qat" ) # "google/gemma-3-4b-it" # "unsloth/gemma-3-4b-it-qat-unsloth-bnb-4bit" # unsloth/gemma-3-4b-it-qat
if USE_LLAMA_CPP == "False":
    GEMMA3_4B_REPO_ID = GEMMA3_4B_REPO_TRANSFORMERS_ID

GEMMA3_4B_MODEL_FILE = get_or_create_env_var("GEMMA3_4B_MODEL_FILE", "gemma-3-4b-it-qat-Q4_K_M.gguf")
GEMMA3_4B_MODEL_FOLDER = get_or_create_env_var("GEMMA3_4B_MODEL_FOLDER", "model/gemma3_4b")

GPT_OSS_REPO_ID = get_or_create_env_var("GPT_OSS_REPO_ID", "unsloth/gpt-oss-20b-GGUF")
GPT_OSS_REPO_TRANSFORMERS_ID = get_or_create_env_var("GPT_OSS_REPO_TRANSFORMERS_ID", "unsloth/gpt-oss-20b-unsloth-bnb-4bit")
if USE_LLAMA_CPP == "False": GPT_OSS_REPO_ID = GPT_OSS_REPO_TRANSFORMERS_ID

GPT_OSS_MODEL_FILE = get_or_create_env_var("GPT_OSS_MODEL_FILE", "gpt-oss-20b-F16.gguf")
GPT_OSS_MODEL_FOLDER = get_or_create_env_var("GPT_OSS_MODEL_FOLDER", "model/gpt_oss")

USE_SPECULATIVE_DECODING = get_or_create_env_var("USE_SPECULATIVE_DECODING", "False")

if CHOSEN_LOCAL_MODEL_TYPE == "Gemma 3 4B": ASSISTANT_MODEL = get_or_create_env_var("ASSISTANT_MODEL", "unsloth/gemma-3-270m-it")
elif CHOSEN_LOCAL_MODEL_TYPE == "Qwen 3 4B": ASSISTANT_MODEL = get_or_create_env_var("ASSISTANT_MODEL", "unsloth/Qwen3-0.6B")

DRAFT_MODEL_LOC = get_or_create_env_var("DRAFT_MODEL_LOC", ".cache/llama.cpp/")

GEMMA3_DRAFT_MODEL_LOC = get_or_create_env_var("GEMMA3_DRAFT_MODEL_LOC", DRAFT_MODEL_LOC + "unsloth_gemma-3-270m-it-qat-GGUF_gemma-3-270m-it-qat-F16.gguf")

GEMMA3_4B_DRAFT_MODEL_LOC = get_or_create_env_var("GEMMA3_4B_DRAFT_MODEL_LOC", DRAFT_MODEL_LOC + "unsloth_gemma-3-4b-it-qat-GGUF_gemma-3-4b-it-qat-Q4_K_M.gguf")

QWEN3_4B_REPO_ID = get_or_create_env_var("QWEN3_4B_REPO_ID", "unsloth/Qwen3-4B-Instruct-2507-GGUF")
QWEN3_4B_REPO_TRANSFORMERS_ID = get_or_create_env_var("QWEN3_4B_REPO_TRANSFORMERS_ID", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
if USE_LLAMA_CPP == "False": QWEN3_4B_REPO_ID = QWEN3_4B_REPO_TRANSFORMERS_ID

QWEN3_4B_MODEL_FILE = get_or_create_env_var("QWEN3_4B_MODEL_FILE", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
QWEN3_4B_MODEL_FOLDER = get_or_create_env_var("QWEN3_4B_MODEL_FOLDER", "model/qwen")

QWEN3_DRAFT_MODEL_LOC = get_or_create_env_var("QWEN3_DRAFT_MODEL_LOC", DRAFT_MODEL_LOC + "Qwen3-0.6B-Q8_0.gguf")
QWEN3_4B_DRAFT_MODEL_LOC = get_or_create_env_var("QWEN3_4B_DRAFT_MODEL_LOC", DRAFT_MODEL_LOC + "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

if CHOSEN_LOCAL_MODEL_TYPE == "Gemma 2b":
    LOCAL_REPO_ID = GEMMA2_REPO_ID
    LOCAL_MODEL_FILE = GEMMA2_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA2_MODEL_FOLDER

# WARNING: In my testing, Gemma 3 1B was not capable enough of giving consistent output tables. I would strongly advise sticking with Gemma 3 4B
elif CHOSEN_LOCAL_MODEL_TYPE == "Gemma 3 1B":
    LOCAL_REPO_ID = GEMMA3_REPO_ID
    LOCAL_MODEL_FILE = GEMMA3_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA3_MODEL_FOLDER

elif CHOSEN_LOCAL_MODEL_TYPE == "Gemma 3 4B":
    LOCAL_REPO_ID = GEMMA3_4B_REPO_ID
    LOCAL_MODEL_FILE = GEMMA3_4B_MODEL_FILE
    LOCAL_MODEL_FOLDER = GEMMA3_4B_MODEL_FOLDER

elif CHOSEN_LOCAL_MODEL_TYPE == "Qwen 3 4B":
    LOCAL_REPO_ID = QWEN3_4B_REPO_ID
    LOCAL_MODEL_FILE = QWEN3_4B_MODEL_FILE
    LOCAL_MODEL_FOLDER = QWEN3_4B_MODEL_FOLDER

elif CHOSEN_LOCAL_MODEL_TYPE == "gpt-oss-20b":
    LOCAL_REPO_ID = GPT_OSS_REPO_ID
    LOCAL_MODEL_FILE = GPT_OSS_MODEL_FILE
    LOCAL_MODEL_FOLDER = GPT_OSS_MODEL_FOLDER

LLM_MAX_GPU_LAYERS = int(get_or_create_env_var('LLM_MAX_GPU_LAYERS','-1')) # Maximum possible
LLM_TEMPERATURE = float(get_or_create_env_var('LLM_TEMPERATURE', '0.6'))
LLM_TOP_K = int(get_or_create_env_var('LLM_TOP_K','64')) # https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune
LLM_MIN_P = float(get_or_create_env_var('LLM_MIN_P', '0'))
LLM_TOP_P = float(get_or_create_env_var('LLM_TOP_P', '0.95'))
LLM_REPETITION_PENALTY = float(get_or_create_env_var('LLM_REPETITION_PENALTY', '1.0'))

LLM_LAST_N_TOKENS = int(get_or_create_env_var('LLM_LAST_N_TOKENS', '512'))
LLM_MAX_NEW_TOKENS = int(get_or_create_env_var('LLM_MAX_NEW_TOKENS', '8192'))
LLM_SEED = int(get_or_create_env_var('LLM_SEED', '42'))
LLM_RESET = get_or_create_env_var('LLM_RESET', 'True')
LLM_STREAM = get_or_create_env_var('LLM_STREAM', 'True')
LLM_THREADS = int(get_or_create_env_var('LLM_THREADS', '-1'))
LLM_BATCH_SIZE = int(get_or_create_env_var('LLM_BATCH_SIZE', '512'))
LLM_CONTEXT_LENGTH = int(get_or_create_env_var('LLM_CONTEXT_LENGTH', '32768'))
LLM_SAMPLE = get_or_create_env_var('LLM_SAMPLE', 'True')
LLM_STOP_STRINGS = get_or_create_env_var('LLM_STOP_STRINGS', r"['                                          ','\n\n\n\n','---------------------------------------------]")
MULTIMODAL_PROMPT_FORMAT = get_or_create_env_var('MULTIMODAL_PROMPT_FORMAT', 'False')
SPECULATIVE_DECODING = get_or_create_env_var('SPECULATIVE_DECODING', 'False')
NUM_PRED_TOKENS = int(get_or_create_env_var('NUM_PRED_TOKENS', '2'))
KV_QUANT_LEVEL = int(get_or_create_env_var('KV_QUANT_LEVEL', '16'))




# If you are using e.g. gpt-oss, you can add a reasoning suffix to set reasoning level, or turn it off in the case of Qwen 3 4B
if CHOSEN_LOCAL_MODEL_TYPE == "gpt-oss-20b": REASONING_SUFFIX = get_or_create_env_var('REASONING_SUFFIX', 'Reasoning: low')
elif CHOSEN_LOCAL_MODEL_TYPE == "Qwen 3 4B" and USE_LLAMA_CPP == "False": REASONING_SUFFIX = get_or_create_env_var('REASONING_SUFFIX', '/nothink')
else: REASONING_SUFFIX = get_or_create_env_var('REASONING_SUFFIX', '') 

# Transformers variables
COMPILE_TRANSFORMERS = get_or_create_env_var('COMPILE_TRANSFORMERS', 'False') # Whether to compile transformers models
USE_BITSANDBYTES = get_or_create_env_var('USE_BITSANDBYTES', 'True') # Whether to use bitsandbytes for quantization
COMPILE_MODE = get_or_create_env_var('COMPILE_MODE', 'reduce-overhead') # alternatively 'max-autotune'
MODEL_DTYPE = get_or_create_env_var('MODEL_DTYPE', 'bfloat16') # alternatively 'bfloat16'
INT8_WITH_OFFLOAD_TO_CPU = get_or_create_env_var('INT8_WITH_OFFLOAD_TO_CPU', 'False') # Whether to offload to CPU


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

S3_ALLOW_LIST_PATH = get_or_create_env_var('S3_ALLOW_LIST_PATH', '') # default_allow_list.csv # This is a path within the named S3 bucket

if ALLOW_LIST_PATH: OUTPUT_ALLOW_LIST_PATH = ALLOW_LIST_PATH
else: OUTPUT_ALLOW_LIST_PATH = 'config/default_allow_list.csv'

FILE_INPUT_HEIGHT = get_or_create_env_var('FILE_INPUT_HEIGHT', '200')

SHOW_EXAMPLES = get_or_create_env_var('SHOW_EXAMPLES', 'True')

###
# COST CODE OPTIONS
###

SHOW_COSTS = get_or_create_env_var('SHOW_COSTS', 'False')

GET_COST_CODES = get_or_create_env_var('GET_COST_CODES', 'False')

DEFAULT_COST_CODE = get_or_create_env_var('DEFAULT_COST_CODE', '')

COST_CODES_PATH = get_or_create_env_var('COST_CODES_PATH', '') # 'config/COST_CENTRES.csv' # file should be a csv file with a single table in it that has two columns with a header. First column should contain cost codes, second column should contain a name or description for the cost code

S3_COST_CODES_PATH = get_or_create_env_var('S3_COST_CODES_PATH', '') # COST_CENTRES.csv # This is a path within the DOCUMENT_REDACTION_BUCKET  
    
# A default path in case s3 cost code location is provided but no local cost code location given
if COST_CODES_PATH: OUTPUT_COST_CODES_PATH = COST_CODES_PATH
else: OUTPUT_COST_CODES_PATH = 'config/cost_codes.csv'

ENFORCE_COST_CODES = get_or_create_env_var('ENFORCE_COST_CODES', 'False') # If you have cost codes listed, is it compulsory to choose one before redacting?

if ENFORCE_COST_CODES == 'True': GET_COST_CODES = 'True'