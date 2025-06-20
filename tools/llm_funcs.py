from typing import TypeVar
import torch.cuda
import os
import time
import boto3
import json
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from typing import List, Tuple
import google.generativeai as ai
import gradio as gr
from gradio import Progress

torch.cuda.empty_cache()

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

model_type = None # global variable setup
full_text = "" # Define dummy source text (full text) just to enable highlight function to load
model = [] # Define empty list for model functions to run
tokenizer = [] #[] # Define empty list for model functions to run


# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
# Check for torch cuda
print("Is CUDA enabled? ", torch.cuda.is_available())
print("Is a CUDA device available on this computer?", torch.backends.cudnn.enabled)
if torch.cuda.is_available():
    torch_device = "cuda"
    gpu_layers = -1
    os.system("nvidia-smi")
else: 
    torch_device =  "cpu"
    gpu_layers = 0

print("Device used is: ", torch_device)
print("Running on device:", torch_device)

from tools.config import RUN_AWS_FUNCTIONS, AWS_REGION, LLM_TEMPERATURE, LLM_TOP_K, LLM_TOP_P, LLM_REPETITION_PENALTY, LLM_LAST_N_TOKENS, LLM_MAX_NEW_TOKENS, LLM_SEED, LLM_RESET, LLM_STREAM, LLM_THREADS, LLM_BATCH_SIZE, LLM_CONTEXT_LENGTH, LLM_SAMPLE, MAX_TOKENS, TIMEOUT_WAIT, NUMBER_OF_RETRY_ATTEMPTS, MAX_TIME_FOR_LOOP, BATCH_SIZE_DEFAULT, DEDUPLICATION_THRESHOLD, MAX_COMMENT_CHARS, RUN_LOCAL_MODEL, CHOSEN_LOCAL_MODEL_TYPE, LOCAL_REPO_ID, LOCAL_MODEL_FILE, LOCAL_MODEL_FOLDER, HF_TOKEN

if RUN_LOCAL_MODEL == "1":
    print("Running local model - importing llama-cpp-python")
    from llama_cpp import Llama

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

if not LLM_THREADS:
    threads = torch.get_num_threads() # 8
else: threads = LLM_THREADS
print("CPU threads:", threads)

if LLM_RESET == 'True': reset = True
else: reset = False

if LLM_STREAM == 'True': stream = True
else: stream = False

if LLM_SAMPLE == 'True': sample = True
else: sample = False

temperature = LLM_TEMPERATURE
top_k = LLM_TOP_K
top_p = LLM_TOP_P
repetition_penalty = LLM_REPETITION_PENALTY # Mild repetition penalty to prevent repeating table rows
last_n_tokens = LLM_LAST_N_TOKENS
max_new_tokens: int = LLM_MAX_NEW_TOKENS
seed: int = LLM_SEED
reset: bool = reset
stream: bool = stream
threads: int = threads
batch_size:int = LLM_BATCH_SIZE
context_length:int = LLM_CONTEXT_LENGTH
sample = LLM_SAMPLE

class llama_cpp_init_config_gpu:
    def __init__(self,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 n_threads=threads,
                 n_batch=batch_size,
                 n_ctx=context_length,
                 n_gpu_layers=gpu_layers):

        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        # self.stop: list[str] = field(default_factory=lambda: [stop_string])

    def update_gpu(self, new_value):
        self.n_gpu_layers = new_value

    def update_context(self, new_value):
        self.n_ctx = new_value

class llama_cpp_init_config_cpu(llama_cpp_init_config_gpu):
    def __init__(self):
        super().__init__()
        self.n_gpu_layers = gpu_layers
        self.n_ctx=context_length

gpu_config = llama_cpp_init_config_gpu()
cpu_config = llama_cpp_init_config_cpu()


class LlamaCPPGenerationConfig:
    def __init__(self, temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repeat_penalty=repetition_penalty,
                 seed=seed,
                 stream=stream,
                 max_tokens=max_new_tokens
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens=max_tokens
        self.stream = stream

    def update_temp(self, new_value):
        self.temperature = new_value

# ResponseObject class for AWS Bedrock calls
class ResponseObject:
        def __init__(self, text, usage_metadata):
            self.text = text
            self.usage_metadata = usage_metadata

###
# LOCAL MODEL FUNCTIONS
###

def get_model_path(repo_id=LOCAL_REPO_ID, model_filename=LOCAL_MODEL_FILE, model_dir=LOCAL_MODEL_FOLDER, hf_token=HF_TOKEN):
    # Construct the expected local path
    local_path = os.path.join(model_dir, model_filename)

    print("local path for model load:", local_path)

    if os.path.exists(local_path):
        print(f"Model already exists at: {local_path}")

        return local_path
    else:
        print(f"Checking default Hugging Face folder. Downloading model from Hugging Face Hub if not found")
        if hf_token:
            downloaded_model_path = hf_hub_download(repo_id=repo_id, token=hf_token, filename=model_filename)
        else:
            downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

        return downloaded_model_path

def load_model(local_model_type:str=CHOSEN_LOCAL_MODEL_TYPE, gpu_layers:int=gpu_layers, max_context_length:int=context_length, gpu_config:llama_cpp_init_config_gpu=gpu_config, cpu_config:llama_cpp_init_config_cpu=cpu_config, torch_device:str=torch_device, repo_id=LOCAL_REPO_ID, model_filename=LOCAL_MODEL_FILE, model_dir=LOCAL_MODEL_FOLDER):
    '''
    Load in a model from Hugging Face hub via the transformers package, or using llama_cpp_python by downloading a GGUF file from Huggingface Hub. 
    '''
    print("Loading model ", local_model_type)
    model_path = get_model_path(repo_id=repo_id, model_filename=model_filename, model_dir=model_dir)  

    print("model_path:", model_path)

    # GPU mode    
    if torch_device == "cuda":
        gpu_config.update_gpu(gpu_layers)
        gpu_config.update_context(max_context_length)        

        try:
            print("GPU load variables:" , vars(gpu_config))
            llama_model = Llama(model_path=model_path, **vars(gpu_config)) #  type_k=8, type_v = 8, flash_attn=True, 
    
        except Exception as e:
            print("GPU load failed due to:", e)
            llama_model = Llama(model_path=model_path, type_k=8, **vars(cpu_config)) # type_v = 8, flash_attn=True, 
        
        print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU. And a maximum context length of ", gpu_config.n_ctx)
    
    # CPU mode
    else:
        gpu_config.update_gpu(gpu_layers)
        cpu_config.update_gpu(gpu_layers)

        # Update context length according to slider
        gpu_config.update_context(max_context_length)
        cpu_config.update_context(max_context_length)

        llama_model = Llama(model_path=model_path, type_k=8, **vars(cpu_config)) # type_v = 8, flash_attn=True, 

        print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU. And a maximum context length of ", gpu_config.n_ctx)
    
    tokenizer = []

    print("Finished loading model:", local_model_type)
    return llama_model, tokenizer

def call_llama_cpp_model(formatted_string:str, gen_config:str, model=model):
    """
    Calls your generation model with parameters from the LlamaCPPGenerationConfig object.

    Args:
        formatted_string (str): The formatted input text for the model.
        gen_config (LlamaCPPGenerationConfig): An object containing generation parameters.
    """
    # Extracting parameters from the gen_config object
    temperature = gen_config.temperature
    top_k = gen_config.top_k
    top_p = gen_config.top_p
    repeat_penalty = gen_config.repeat_penalty
    seed = gen_config.seed
    max_tokens = gen_config.max_tokens
    stream = gen_config.stream

    # Now you can call your model directly, passing the parameters:
    output = model(
        formatted_string, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        repeat_penalty=repeat_penalty, 
        seed=seed,
        max_tokens=max_tokens,
        stream=stream#,
        #stop=["<|eot_id|>", "\n\n"]
    )

    return output

# This function is not used in this app
def llama_cpp_streaming(history, full_prompt, temperature=temperature):

    gen_config = LlamaCPPGenerationConfig()
    gen_config.update_temp(temperature)

    print(vars(gen_config))

    # Pull the generated text from the streamer, and update the model output.
    start = time.time()
    NUM_TOKENS=0
    print('-'*4+'Start Generation'+'-'*4)

    output = model(
    full_prompt, **vars(gen_config))

    history[-1][1] = ""
    for out in output:

        if "choices" in out and len(out["choices"]) > 0 and "text" in out["choices"][0]:
            history[-1][1] += out["choices"][0]["text"]
            NUM_TOKENS+=1
            yield history
        else:
            print(f"Unexpected output structure: {out}") 

    time_generate = time.time() - start
    print('\n')
    print('-'*4+'End Generation'+'-'*4)
    print(f'Num of generated tokens: {NUM_TOKENS}')
    print(f'Time for complete generation: {time_generate}s')
    print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
    print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

###
# LLM FUNCTIONS
###

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
    #print("Metadata:", response.usage_metadata)
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

    progress_bar = range(0,number_of_api_retry_attempts)

    # Generate the model's response
    if "gemini" in model_choice:

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
    elif "anthropic.claude" in model_choice:
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
        # This is the local model
        for i in progress_bar:
            try:
                print("Calling local model, attempt", i + 1)

                gen_config = LlamaCPPGenerationConfig()
                gen_config.update_temp(temperature)

                response = call_llama_cpp_model(prompt, gen_config, model=local_model)

                #progress_bar.close()
                #tqdm._instances.clear()

                print("Successful call to local model. Response:", response)
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
