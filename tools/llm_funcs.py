
import os
import re
import time
import boto3
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from typing import List, Tuple, TypeVar
from google import genai as ai
from google.genai import types
from gradio import Progress
from openai import OpenAI

model_type = None # global variable setup
full_text = "" # Define dummy source text (full text) just to enable highlight function to load

# Global variables for model and tokenizer
_model = None
_tokenizer = None
_assistant_model = None

from tools.config import LLM_TEMPERATURE, LLM_TOP_K, LLM_MIN_P, LLM_TOP_P, LLM_REPETITION_PENALTY, LLM_LAST_N_TOKENS, LLM_MAX_NEW_TOKENS, LLM_SEED, LLM_RESET, LLM_STREAM, LLM_THREADS, LLM_BATCH_SIZE, LLM_CONTEXT_LENGTH, LLM_SAMPLE, TIMEOUT_WAIT, NUMBER_OF_RETRY_ATTEMPTS, MAX_TIME_FOR_LOOP, BATCH_SIZE_DEFAULT, DEDUPLICATION_THRESHOLD, MAX_COMMENT_CHARS, CHOSEN_LOCAL_MODEL_TYPE, LOCAL_REPO_ID, LOCAL_MODEL_FILE, LOCAL_MODEL_FOLDER, HF_TOKEN, LLM_SEED, LLM_MAX_GPU_LAYERS, SPECULATIVE_DECODING, NUM_PRED_TOKENS, USE_LLAMA_CPP, COMPILE_MODE, MODEL_DTYPE, USE_BITSANDBYTES, COMPILE_TRANSFORMERS, INT8_WITH_OFFLOAD_TO_CPU, LOAD_LOCAL_MODEL_AT_START, ASSISTANT_MODEL, LLM_STOP_STRINGS, MULTIMODAL_PROMPT_FORMAT, KV_QUANT_LEVEL
from tools.helper_functions import _get_env_list

if SPECULATIVE_DECODING == "True": SPECULATIVE_DECODING = True 
else: SPECULATIVE_DECODING = False


if isinstance(NUM_PRED_TOKENS, str): NUM_PRED_TOKENS = int(NUM_PRED_TOKENS)
if isinstance(LLM_MAX_GPU_LAYERS, str): LLM_MAX_GPU_LAYERS = int(LLM_MAX_GPU_LAYERS)
if isinstance(LLM_THREADS, str): LLM_THREADS = int(LLM_THREADS)

if LLM_RESET == 'True': reset = True
else: reset = False

if LLM_STREAM == 'True': stream = True
else: stream = False

if LLM_SAMPLE == 'True': sample = True
else: sample = False

if LLM_STOP_STRINGS: LLM_STOP_STRINGS = _get_env_list(LLM_STOP_STRINGS, strip_strings=False)

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS

temperature = LLM_TEMPERATURE
top_k = LLM_TOP_K
top_p = LLM_TOP_P
min_p = LLM_MIN_P
repetition_penalty = LLM_REPETITION_PENALTY
last_n_tokens = LLM_LAST_N_TOKENS
LLM_MAX_NEW_TOKENS: int = LLM_MAX_NEW_TOKENS
seed: int = LLM_SEED
reset: bool = reset
stream: bool = stream
batch_size:int = LLM_BATCH_SIZE
context_length:int = LLM_CONTEXT_LENGTH
sample = LLM_SAMPLE
stop_strings = LLM_STOP_STRINGS
speculative_decoding = SPECULATIVE_DECODING
if LLM_MAX_GPU_LAYERS != 0:
    gpu_layers = int(LLM_MAX_GPU_LAYERS)
    torch_device =  "cuda"
else:
    gpu_layers = 0
    torch_device =  "cpu"

if not LLM_THREADS: threads = 1
else: threads = LLM_THREADS

class llama_cpp_init_config_gpu:
    def __init__(self,
                 last_n_tokens=last_n_tokens,
                 seed=seed,
                 n_threads=threads,
                 n_batch=batch_size,
                 n_ctx=context_length,
                 n_gpu_layers=gpu_layers,
                 reset=reset):

        self.last_n_tokens = last_n_tokens
        self.seed = seed
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.reset = reset
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
                 min_p=min_p,
                 top_p=top_p,
                 repeat_penalty=repetition_penalty,
                 seed=seed,
                 stream=stream,
                 max_tokens=LLM_MAX_NEW_TOKENS,
                 reset=reset
                 ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.max_tokens=max_tokens
        self.stream = stream
        self.reset = reset
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

    try:
        if os.path.exists(local_path):
            print(f"Model already exists at: {local_path}")

            return local_path
        else:            
            if hf_token:
                print("Downloading model from Hugging Face Hub with HF token")
                downloaded_model_path = hf_hub_download(repo_id=repo_id, token=hf_token, filename=model_filename)

                return downloaded_model_path
            else:
                print("No HF token found, downloading model from Hugging Face Hub without token")
                downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

                return downloaded_model_path

    except Exception as e:
        print("Error loading model:", e)
        raise Warning("Error loading model:", e)
    
def load_model(local_model_type:str=CHOSEN_LOCAL_MODEL_TYPE,
    gpu_layers:int=gpu_layers,
    max_context_length:int=context_length,
    gpu_config:llama_cpp_init_config_gpu=gpu_config,
    cpu_config:llama_cpp_init_config_cpu=cpu_config,
    torch_device:str=torch_device,
    repo_id=LOCAL_REPO_ID,
    model_filename=LOCAL_MODEL_FILE,
    model_dir=LOCAL_MODEL_FOLDER,
    compile_mode=COMPILE_MODE,
    model_dtype=MODEL_DTYPE,
    hf_token=HF_TOKEN,
    speculative_decoding=speculative_decoding,
    model=None,
    tokenizer=None,
    assistant_model=None):
    '''
    Load in a model from Hugging Face hub via the transformers package, or using llama_cpp_python by downloading a GGUF file from Huggingface Hub.

    Args:
        local_model_type (str): The type of local model to load (e.g., "llama-cpp").
        gpu_layers (int): The number of GPU layers to offload to the GPU.
        max_context_length (int): The maximum context length for the model.
        gpu_config (llama_cpp_init_config_gpu): Configuration object for GPU-specific Llama.cpp parameters.
        cpu_config (llama_cpp_init_config_cpu): Configuration object for CPU-specific Llama.cpp parameters.
        torch_device (str): The device to load the model on ("cuda" for GPU, "cpu" for CPU).
        repo_id (str): The Hugging Face repository ID where the model is located.
        model_filename (str): The specific filename of the model to download from the repository.
        model_dir (str): The local directory where the model will be stored or downloaded.
        compile_mode (str): The compilation mode to use for the model.
        model_dtype (str): The data type to use for the model.
        hf_token (str): The Hugging Face token to use for the model.
        speculative_decoding (bool): Whether to use speculative decoding.
        model (Llama/transformers model): The model to load.
        tokenizer (list/transformers tokenizer): The tokenizer to load.
        assistant_model (transformers model): The assistant model for speculative decoding.
    Returns:
        tuple: A tuple containing:
            - model (Llama/transformers model): The loaded Llama.cpp/transformers model instance.
            - tokenizer (list/transformers tokenizer): An empty list (tokenizer is not used with Llama.cpp directly in this setup), or a transformers tokenizer.
            - assistant_model (transformers model): The assistant model for speculative decoding (if speculative_decoding is True).
    '''
    
    if model:
        return model, tokenizer, assistant_model

    print("Loading model:", local_model_type)

    # Verify the device and cuda settings
    # Check if CUDA is enabled
    
    import torch    

    torch.cuda.empty_cache()
    print("Is CUDA enabled? ", torch.cuda.is_available())
    print("Is a CUDA device available on this computer?", torch.backends.cudnn.enabled)
    if torch.cuda.is_available():
        torch_device = "cuda"
        gpu_layers = int(LLM_MAX_GPU_LAYERS)
        print("CUDA version:", torch.version.cuda)
        #try:
        #    os.system("nvidia-smi")
        #except Exception as e:
        #    print("Could not print nvidia-smi settings due to:", e)
    else: 
        torch_device =  "cpu"
        gpu_layers = 0

    print("Running on device:", torch_device)
    print("GPU layers assigned to cuda:", gpu_layers)

    if not LLM_THREADS:
        threads = torch.get_num_threads()
    else: threads = LLM_THREADS
    print("CPU threads:", threads)

    # GPU mode    
    if torch_device == "cuda":
        torch.cuda.empty_cache()
        gpu_config.update_gpu(gpu_layers)
        gpu_config.update_context(max_context_length)        

        if USE_LLAMA_CPP == "True":
            from llama_cpp import Llama
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

            model_path = get_model_path(repo_id=repo_id, model_filename=model_filename, model_dir=model_dir)

            try:
                print("GPU load variables:" , vars(gpu_config))
                if speculative_decoding:
                    model = Llama(model_path=model_path, type_k=KV_QUANT_LEVEL, type_v=KV_QUANT_LEVEL, flash_attn=True, draft_model=LlamaPromptLookupDecoding(num_pred_tokens=NUM_PRED_TOKENS), **vars(gpu_config)) 
                else:
                    model = Llama(model_path=model_path, type_k=KV_QUANT_LEVEL, type_v=KV_QUANT_LEVEL, flash_attn=True, **vars(gpu_config))    

            except Exception as e:
                print("GPU load failed due to:", e, "Loading model in CPU mode")
                # If fails, go to CPU mode
                model = Llama(model_path=model_path, **vars(cpu_config)) 
        
        else:
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            

            print("Loading model from transformers")
            # Use the official model ID for Gemma 3 4B
            model_id = repo_id
            # 1. Set Data Type (dtype)
            # For H200/Hopper: 'bfloat16'
            # For RTX 3060/Ampere: 'float16'
            dtype_str = model_dtype #os.environ.get("MODEL_DTYPE", "bfloat16").lower()
            if dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            elif dtype_str == "float16":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32 # A safe fallback

            # 2. Set Compilation Mode
            # 'max-autotune' is great for both but can be slow initially.
            # 'reduce-overhead' is a faster alternative for compiling.

            print(f"--- System Configuration ---")
            print(f"Using model id: {model_id}")
            print(f"Using dtype: {torch_dtype}")            
            print(f"Using compile mode: {compile_mode}")
            print(f"Using bitsandbytes: {USE_BITSANDBYTES}")
            print("--------------------------\n")

            # --- Load Tokenizer and Model ---   

            try:

                # Load Tokenizer and Model
                # tokenizer = AutoTokenizer.from_pretrained(model_id)

                
                if USE_BITSANDBYTES == "True":              

                    if INT8_WITH_OFFLOAD_TO_CPU == "True":
                        # This will be very slow. Requires at least 4GB of VRAM and 32GB of RAM
                        print("Using bitsandbytes for quantisation to 8 bits, with offloading to CPU")
                        max_memory={0: "4GB", "cpu": "32GB"}
                        quantisation_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        max_memory=max_memory,
                        llm_int8_enable_fp32_cpu_offload=True # Note: if bitsandbytes has to offload to CPU, inference will be slow
                        )
                    else:                        
                        # For Gemma 4B, requires at least 6GB of VRAM
                        print("Using bitsandbytes for quantisation to 4 bits")
                        quantisation_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4", # Use the modern NF4 quantisation for better performance
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True # Optional: uses a second quantisation step to save even more memory
                    )

                    #print("Loading model with bitsandbytes quantisation config:", quantisation_config)

                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_id,
                        max_seq_length=max_context_length,
                        dtype=torch_dtype,
                        device_map="auto",
                        load_in_4bit=True,
                        # quantization_config=quantisation_config, # Not actually used in Unsloth
                        token=hf_token
                    )

                    FastLanguageModel.for_inference(model)
                else:
                    print("Loading model without bitsandbytes quantisation")
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_id,
                        max_seq_length=max_context_length,
                        dtype=torch_dtype,
                        device_map="auto",
                        token=hf_token
                    )

                    FastLanguageModel.for_inference(model)

                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token    

            except Exception as e:
                print("Error loading model with bitsandbytes quantisation config:", e)
                raise Warning("Error loading model with bitsandbytes quantisation config:", e)

            # Compile the Model with the selected mode 🚀
            if COMPILE_TRANSFORMERS == "True":
                try:
                    model = torch.compile(model, mode=compile_mode, fullgraph=True)
                except Exception as e:
                    print(f"Could not compile model: {e}. Running in eager mode.")
        
        print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU and a maximum context length of", gpu_config.n_ctx)
    
    # CPU mode
    else:
        if USE_LLAMA_CPP == "False":
            raise Warning("Using transformers model in CPU mode is not supported. Please change your config variable USE_LLAMA_CPP to True if you want to do CPU inference.")

        model_path = get_model_path(repo_id=repo_id, model_filename=model_filename, model_dir=model_dir)

        #gpu_config.update_gpu(gpu_layers)
        cpu_config.update_gpu(gpu_layers)

        # Update context length according to slider
        #gpu_config.update_context(max_context_length)
        cpu_config.update_context(max_context_length)

        if speculative_decoding:
            model = Llama(model_path=model_path, draft_model=LlamaPromptLookupDecoding(num_pred_tokens=NUM_PRED_TOKENS), **vars(cpu_config))
        else:
            model = Llama(model_path=model_path, **vars(cpu_config)) 

        print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU and a maximum context length of", cpu_config.n_ctx)

    print("Finished loading model:", local_model_type)
    print("GPU layers assigned to cuda:", gpu_layers)

    # Load assistant model for speculative decoding if enabled
    if speculative_decoding and USE_LLAMA_CPP == "False" and torch_device == "cuda":
        print("Loading assistant model for speculative decoding:", ASSISTANT_MODEL)
        try:
            from transformers import AutoModelForCausalLM
            
            # Load the assistant model with the same configuration as the main model
            assistant_model = AutoModelForCausalLM.from_pretrained(
                ASSISTANT_MODEL,
                dtype=torch_dtype,
                device_map="auto",
                token=hf_token
            )

            #assistant_model.config._name_or_path = model.config._name_or_path
            
            # Compile the assistant model if compilation is enabled
            if COMPILE_TRANSFORMERS == "True":
                try:
                    assistant_model = torch.compile(assistant_model, mode=compile_mode, fullgraph=True)
                except Exception as e:
                    print(f"Could not compile assistant model: {e}. Running in eager mode.")
            
            print("Successfully loaded assistant model for speculative decoding")
            
        except Exception as e:
            print(f"Error loading assistant model: {e}")
            assistant_model = None
    else:
        assistant_model = None

    return model, tokenizer, assistant_model

def get_model():
    """Get the globally loaded model. Load it if not already loaded."""
    global _model, _tokenizer, _assistant_model
    if _model is None:
        _model, _tokenizer, _assistant_model = load_model(
            local_model_type=CHOSEN_LOCAL_MODEL_TYPE, 
            gpu_layers=gpu_layers, 
            max_context_length=context_length, 
            gpu_config=gpu_config, 
            cpu_config=cpu_config, 
            torch_device=torch_device, 
            repo_id=LOCAL_REPO_ID, 
            model_filename=LOCAL_MODEL_FILE, 
            model_dir=LOCAL_MODEL_FOLDER, 
            compile_mode=COMPILE_MODE, 
            model_dtype=MODEL_DTYPE, 
            hf_token=HF_TOKEN, 
            model=_model, 
            tokenizer=_tokenizer,
            assistant_model=_assistant_model
        )
    return _model

def get_tokenizer():
    """Get the globally loaded tokenizer. Load it if not already loaded."""
    global _model, _tokenizer, _assistant_model
    if _tokenizer is None:
        _model, _tokenizer, _assistant_model = load_model(
            local_model_type=CHOSEN_LOCAL_MODEL_TYPE, 
            gpu_layers=gpu_layers, 
            max_context_length=context_length, 
            gpu_config=gpu_config, 
            cpu_config=cpu_config, 
            torch_device=torch_device, 
            repo_id=LOCAL_REPO_ID, 
            model_filename=LOCAL_MODEL_FILE, 
            model_dir=LOCAL_MODEL_FOLDER, 
            compile_mode=COMPILE_MODE, 
            model_dtype=MODEL_DTYPE, 
            hf_token=HF_TOKEN, 
            model=_model, 
            tokenizer=_tokenizer,
            assistant_model=_assistant_model
        )
    return _tokenizer

def get_assistant_model():
    """Get the globally loaded assistant model. Load it if not already loaded."""
    global _model, _tokenizer, _assistant_model
    if _assistant_model is None:
        _model, _tokenizer, _assistant_model = load_model(
            local_model_type=CHOSEN_LOCAL_MODEL_TYPE, 
            gpu_layers=gpu_layers, 
            max_context_length=context_length, 
            gpu_config=gpu_config, 
            cpu_config=cpu_config, 
            torch_device=torch_device, 
            repo_id=LOCAL_REPO_ID, 
            model_filename=LOCAL_MODEL_FILE, 
            model_dir=LOCAL_MODEL_FOLDER, 
            compile_mode=COMPILE_MODE, 
            model_dtype=MODEL_DTYPE, 
            hf_token=HF_TOKEN, 
            model=_model, 
            tokenizer=_tokenizer,
            assistant_model=_assistant_model
        )
    return _assistant_model

def set_model(model, tokenizer, assistant_model=None):
    """Set the global model, tokenizer, and assistant model."""
    global _model, _tokenizer, _assistant_model
    _model = model
    _tokenizer = tokenizer
    _assistant_model = assistant_model

# Initialize model at startup if configured
if LOAD_LOCAL_MODEL_AT_START == "True":
    get_model()  # This will trigger loading

def call_llama_cpp_model(formatted_string:str, gen_config:str, model=None):
    """
    Calls your generation model with parameters from the LlamaCPPGenerationConfig object.

    Args:
        formatted_string (str): The formatted input text for the model.
        gen_config (LlamaCPPGenerationConfig): An object containing generation parameters.
        model: Optional model instance. If None, will use the globally loaded model.
    """
    if model is None:
        model = get_model()
    
    if model is None:
        raise ValueError("No model available. Either pass a model parameter or ensure LOAD_LOCAL_MODEL_AT_START is True.")
    
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

def call_llama_cpp_chatmodel(formatted_string:str, system_prompt:str, gen_config:LlamaCPPGenerationConfig, model=None):
    """
    Calls your Llama.cpp chat model with a formatted user message and system prompt,
    using generation parameters from the LlamaCPPGenerationConfig object.

    Args:
        formatted_string (str): The formatted input text for the user's message.
        system_prompt (str): The system-level instructions for the model.
        gen_config (LlamaCPPGenerationConfig): An object containing generation parameters.
        model: Optional model instance. If None, will use the globally loaded model.
    """
    if model is None:
        model = get_model()
    
    if model is None:
        raise ValueError("No model available. Either pass a model parameter or ensure LOAD_LOCAL_MODEL_AT_START is True.")
    
    # Extracting parameters from the gen_config object
    temperature = gen_config.temperature
    top_k = gen_config.top_k
    top_p = gen_config.top_p
    repeat_penalty = gen_config.repeat_penalty
    seed = gen_config.seed
    max_tokens = gen_config.max_tokens
    stream = gen_config.stream
    reset = gen_config.reset

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": formatted_string}
    ]

    input_tokens = len(model.tokenize((system_prompt + "\n" + formatted_string).encode("utf-8"), special=True))

    if stream:
        final_tokens = []
        output_tokens = 0
        for chunk in model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            max_tokens=max_tokens,
            stream=True,
            stop=stop_strings,
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content") or chunk["choices"][0].get("text") or ""
            if token:
                print(token, end="", flush=True)
                final_tokens.append(token)
                output_tokens += 1
        print()  # newline after stream finishes

        text = "".join(final_tokens)

        if reset:
            model.reset()

        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            # Provide a usage object so downstream code can read it
            "usage": {
                "prompt_tokens": input_tokens,         # unknown during streaming
                "completion_tokens": output_tokens,     # unknown during streaming
                "total_tokens": input_tokens + output_tokens,          # unknown during streaming
            },
        }

    else:
        response = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            max_tokens=max_tokens,
            stream=False,
            stop=stop_strings,
        )

        if reset:
            model.reset()

        return response

###
# LLM FUNCTIONS
###

def construct_gemini_generative_model(in_api_key: str, temperature: float, model_choice: str, system_prompt: str, max_tokens: int, random_seed=seed) -> Tuple[object, dict]:
    """
    Constructs a GenerativeModel for Gemini API calls.
    ...
    """
    # Construct a GenerativeModel
    try:
        if in_api_key:
            #print("Getting API key from textbox")
            api_key = in_api_key
            client = ai.Client(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            #print("Searching for API key in environmental variables")
            api_key = os.environ["GOOGLE_API_KEY"]
            client = ai.Client(api_key=api_key)
        else:
            print("No Gemini API key found")
            raise Warning("No Gemini API key found.")
    except Exception as e:
        print("Error constructing Gemini generative model:", e)
        raise Warning("Error constructing Gemini generative model:", e)
        
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens, seed=random_seed)

    return client, config

def construct_azure_client(in_api_key: str, endpoint: str) -> Tuple[object, dict]:
    """
    Constructs an OpenAI client for Azure/OpenAI AI Inference.
    """
    try:
        key = None
        if in_api_key:
            key = in_api_key
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            key = os.environ["AZURE_OPENAI_API_KEY"]
        if not key:
            raise Warning("No Azure/OpenAI API key found.")

        if not endpoint:
            endpoint = os.environ.get("AZURE_OPENAI_INFERENCE_ENDPOINT", "")
            if not endpoint:
                # Assume using OpenAI API
                client = OpenAI(
                api_key=key,
                )
            else:
                # Use the provided endpoint
                client = OpenAI(
                api_key=key,
                base_url=f"{endpoint}",
                )

        

        return client, dict()
    except Exception as e:
        print("Error constructing Azure/OpenAI client:", e)
        raise

def call_aws_claude(prompt: str, system_prompt: str, temperature: float, max_tokens: int, model_choice:str, bedrock_runtime:boto3.Session.client, assistant_prefill:str="") -> ResponseObject:
    """
    This function sends a request to AWS Claude with the following parameters:
    - prompt: The user's input prompt to be processed by the model.
    - system_prompt: A system-defined prompt that provides context or instructions for the model.
    - temperature: A value that controls the randomness of the model's output, with higher values resulting in more diverse responses.
    - max_tokens: The maximum number of tokens (words or characters) in the model's response.
    - model_choice: The specific model to use for processing the request.
    - bedrock_runtime: The client object for boto3 Bedrock runtime
    - assistant_prefill: A string indicating the text that the response should start with.
    
    The function constructs the request configuration, invokes the model, extracts the response text, and returns a ResponseObject containing the text and metadata.
    """

    inference_config = {
        "maxTokens": max_tokens,
        "topP": 0.999,
        "temperature":temperature,
    }

    if not assistant_prefill:
        messages =  [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                    ],
                }
            ]
    else:
        messages =  [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    # Pre-filling with '|'
                    "content": [{"text": assistant_prefill}]
                }
            ]
    
    system_prompt_list = [
        {
            'text': system_prompt
        }
    ]

    # The converse API call itself. Note I've renamed the response variable for clarity.
    api_response = bedrock_runtime.converse(
        modelId=model_choice,
        messages=messages,
        system=system_prompt_list,
        inferenceConfig=inference_config
    )

    output_message = api_response['output']['message']

    if 'reasoningContent' in output_message['content'][0]:
        # Extract the reasoning text
        reasoning_text = output_message['content'][0]['reasoningContent']['reasoningText']['text']

        # Extract the output text
        text = assistant_prefill + output_message['content'][1]['text']
    else:
        text = assistant_prefill + output_message['content'][0]['text']

    # The usage statistics are neatly provided in the 'usage' key.
    usage = api_response['usage']
    
    # The full API response metadata is in 'ResponseMetadata' if you still need it.
    metadata = api_response['ResponseMetadata']

    # Create ResponseObject with the cleanly extracted data.
    response = ResponseObject(
        text=text,
        usage_metadata=usage
    )
    
    return response

def call_transformers_model(prompt: str, system_prompt: str, gen_config: LlamaCPPGenerationConfig, model=None, tokenizer=None, assistant_model=None, speculative_decoding=speculative_decoding, progress=Progress(track_tqdm=False)):
    """
    This function sends a request to a transformers model (through Unsloth) with the given prompt, system prompt, and generation configuration.
    """
    from transformers import TextStreamer

    if model is None:
        model = get_model()
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if assistant_model is None and speculative_decoding:
        assistant_model = get_assistant_model()
    
    if model is None or tokenizer is None:
        raise ValueError("No model or tokenizer available. Either pass them as parameters or ensure LOAD_LOCAL_MODEL_AT_START is True.")
    
    # 1. Define the conversation as a list of dictionaries
    def wrap_text_message(text):
        return [{"type": "text", "text": text}]

    if MULTIMODAL_PROMPT_FORMAT == "True":
        conversation = [
            {"role": "system", "content": wrap_text_message(system_prompt)},
            {"role": "user", "content": wrap_text_message(prompt)}
        ]

    else:
        conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]

    # 2. Apply the chat template
    # This function formats the conversation into the exact string Gemma 3 expects.
    # add_generation_prompt=True adds the special tokens that tell the model it's its turn to speak.

    try:
        input_ids = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt = True, # Must add for generation
                tokenize = True,
                return_tensors = "pt",
            ).to("cuda")
    except Exception as e:
        print("Error applying chat template:", e)
        raise

    # Map LlamaCPP parameters to transformers parameters
    generation_kwargs = {
        'max_new_tokens': gen_config.max_tokens,
        'temperature': gen_config.temperature,
        'top_p': gen_config.top_p,
        'top_k': gen_config.top_k,
        'do_sample': True
        #'pad_token_id': tokenizer.eos_token_id
    }

    if gen_config.stream:
        streamer = TextStreamer(tokenizer, skip_prompt = True)
    else:
        streamer = None
    
    # Remove parameters that don't exist in transformers
    if hasattr(gen_config, 'repeat_penalty'):
        generation_kwargs['repetition_penalty'] = gen_config.repeat_penalty

    # --- Timed Inference Test ---
    print("\nStarting model inference...")
    start_time = time.time()

    # Use speculative decoding if assistant model is available
    if speculative_decoding and assistant_model is not None:
        #print("Using speculative decoding with assistant model")
        outputs = model.generate(
            input_ids,
            assistant_model=assistant_model,
            **generation_kwargs,
        streamer = streamer
        )
    else:
        #print("Generating without speculative decoding")
        outputs = model.generate(
            input_ids,
            **generation_kwargs,
        streamer = streamer
        )

    end_time = time.time()

    # --- Decode and Display Results ---
    new_tokens = outputs[0][input_ids.shape[-1]:]
    assistant_reply = tokenizer.decode(new_tokens, skip_special_tokens=True)


    num_input_tokens = input_ids.shape[-1]  # This gets the sequence length (number of tokens)
    num_generated_tokens = len(new_tokens)
    duration = end_time - start_time
    tokens_per_second = num_generated_tokens / duration

    print("\n--- Performance ---")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return assistant_reply, num_input_tokens, num_generated_tokens

# Function to send a request and update history
def send_request(prompt: str, conversation_history: List[dict], client: ai.Client | OpenAI, config: types.GenerateContentConfig, model_choice: str, system_prompt: str, temperature: float, bedrock_runtime:boto3.Session.client, model_source:str, local_model= list(), tokenizer=None, assistant_model=None, assistant_prefill = "", progress=Progress(track_tqdm=True)) -> Tuple[str, List[dict]]:
    """Sends a request to a language model and manages the conversation history.

    This function constructs the full prompt by appending the new user prompt to the conversation history,
    generates a response from the model, and updates the conversation history with the new prompt and response.
    It handles different model sources (Gemini, AWS, Local) and includes retry logic for API calls.

    Args:
        prompt (str): The user's input prompt to be sent to the model.
        conversation_history (List[dict]): A list of dictionaries representing the ongoing conversation.
                                           Each dictionary should have 'role' and 'parts' keys.
        client (ai.Client): The API client object for the chosen model (e.g., Gemini `ai.Client`, or Azure/OpenAI `OpenAI`).
        config (types.GenerateContentConfig): Configuration settings for content generation (e.g., Gemini `types.GenerateContentConfig`).
        model_choice (str): The specific model identifier to use (e.g., "gemini-pro", "claude-v2").
        system_prompt (str): An optional system-level instruction or context for the model.
        temperature (float): Controls the randomness of the model's output, with higher values leading to more diverse responses.
        bedrock_runtime (boto3.Session.client): The boto3 Bedrock runtime client object for AWS models.
        model_source (str): Indicates the source/provider of the model (e.g., "Gemini", "AWS", "Local").
        local_model (list, optional): A list containing the local model and its tokenizer (if `model_source` is "Local"). Defaults to [].
        tokenizer (object, optional): The tokenizer object for local models. Defaults to None.
        assistant_model (object, optional): An optional assistant model used for speculative decoding with local models. Defaults to None.
        assistant_prefill (str, optional): A string to pre-fill the assistant's response, useful for certain models like Claude. Defaults to "".
        progress (Progress, optional): A progress object for tracking the operation, typically from `tqdm`. Defaults to Progress(track_tqdm=True).

    Returns:
        Tuple[str, List[dict]]: A tuple containing the model's response text and the updated conversation history.
    """
    # Constructing the full prompt from the conversation history
    full_prompt = "Conversation history:\n"
    num_transformer_input_tokens = 0
    num_transformer_generated_tokens = 0
    response_text = ""
    
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
    if "Gemini" in model_source:

        for i in progress_bar:
            try:
                print("Calling Gemini model, attempt", i + 1)

                response = client.models.generate_content(model=model_choice, contents=full_prompt, config=config)

                #print("Successful call to Gemini model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to Gemini model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")               

                time.sleep(timeout_wait)
            
            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens
                
    elif "AWS" in model_source:
        for i in progress_bar:
            try:
                print("Calling AWS Claude model, attempt", i + 1)
                response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice, bedrock_runtime=bedrock_runtime, assistant_prefill=assistant_prefill)

                #print("Successful call to Claude model.")
                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to Claude model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")
                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens
    elif "Azure/OpenAI" in model_source:
        for i in progress_bar:
            try:
                print("Calling Azure/OpenAI inference model, attempt", i + 1)
                
                messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ]

                response_raw = client.chat.completions.create(
                messages=messages,
                model=model_choice,
                temperature=temperature,
                max_completion_tokens=max_tokens
                )

                response_text = response_raw.choices[0].message.content
                usage = getattr(response_raw, "usage", None)
                input_tokens = 0
                output_tokens = 0
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
                    output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
                response = ResponseObject(
                    text=response_text,
                    usage_metadata={'inputTokens': input_tokens, 'outputTokens': output_tokens}
                )
                break
            except Exception as e:
                print("Call to Azure/OpenAI model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")
                time.sleep(timeout_wait)
            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens
    elif "Local" in model_source:
        # This is the local model
        for i in progress_bar:
            try:
                print("Calling local model, attempt", i + 1)

                gen_config = LlamaCPPGenerationConfig()
                gen_config.update_temp(temperature)

                if USE_LLAMA_CPP == "True":
                    response = call_llama_cpp_chatmodel(prompt, system_prompt, gen_config, model=local_model)

                else:
                    response, num_transformer_input_tokens, num_transformer_generated_tokens = call_transformers_model(prompt, system_prompt, gen_config, model=local_model, tokenizer=tokenizer, assistant_model=assistant_model)
                    response_text = response

                break
            except Exception as e:
                # If fails, try again after X seconds in case there is a throttle limit
                print("Call to local model failed:", e, " Waiting for ", str(timeout_wait), "seconds and trying again.")         

                time.sleep(timeout_wait)

            if i == number_of_api_retry_attempts:
                return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens
    else:
        print("Model source not recognised")
        return ResponseObject(text="", usage_metadata={'RequestId':"FAILED"}), conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens

    # Update the conversation history with the new prompt and response
    conversation_history.append({'role': 'user', 'parts': [prompt]})

    # Check if is a LLama.cpp model response
    if isinstance(response, ResponseObject):
        response_text = response.text
    elif 'choices' in response: # LLama.cpp model response
        if "gpt-oss" in model_choice: response_text = response['choices'][0]['message']['content'].split('<|start|>assistant<|channel|>final<|message|>')[1]
        else: response_text = response['choices'][0]['message']['content']
    elif model_source == "Gemini":
        response_text = response.text
    else: # Assume transformers model response
        if "gpt-oss" in model_choice: response_text = response.split('<|start|>assistant<|channel|>final<|message|>')[1]
        else: response_text = response
            
    # Replace multiple spaces with single space
    response_text = re.sub(r' {2,}', ' ', response_text)  
    response_text = response_text.strip()
    
    conversation_history.append({'role': 'assistant', 'parts': [response_text]})
    
    return response, conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens

def process_requests(prompts: List[str],
system_prompt: str,
conversation_history: List[dict],
whole_conversation: List[str],
whole_conversation_metadata: List[str],
client: ai.Client | OpenAI,
config: types.GenerateContentConfig,
model_choice: str,
temperature: float,
bedrock_runtime:boto3.Session.client,
model_source:str,
batch_no:int = 1,
local_model = list(),
tokenizer=None,
assistant_model=None,
master:bool = False,
assistant_prefill="") -> Tuple[List[ResponseObject], List[dict], List[str], List[str]]:
    """
    Processes a list of prompts by sending them to the model, appending the responses to the conversation history, and updating the whole conversation and metadata.

    Args:
        prompts (List[str]): A list of prompts to be processed.
        system_prompt (str): The system prompt.
        conversation_history (List[dict]): The history of the conversation.
        whole_conversation (List[str]): The complete conversation including prompts and responses.
        whole_conversation_metadata (List[str]): Metadata about the whole conversation.
        client (object): The client to use for processing the prompts, from either Gemini or OpenAI client.
        config (dict): Configuration for the model.
        model_choice (str): The choice of model to use.        
        temperature (float): The temperature parameter for the model.
        model_source (str): Source of the model, whether local, AWS, or Gemini
        batch_no (int): Batch number of the large language model request.
        local_model: Local gguf model (if loaded)
        master (bool): Is this request for the master table.
        assistant_prefill (str, optional): Is there a prefill for the assistant response. Currently only working for AWS model calls
        bedrock_runtime: The client object for boto3 Bedrock runtime

    Returns:
        Tuple[List[ResponseObject], List[dict], List[str], List[str]]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, and the updated whole conversation metadata.
    """
    responses = list()

    # Clear any existing progress bars
    tqdm._instances.clear()

    for prompt in prompts:

        response, conversation_history, response_text, num_transformer_input_tokens, num_transformer_generated_tokens = send_request(prompt, conversation_history, client=client, config=config, model_choice=model_choice, system_prompt=system_prompt, temperature=temperature, local_model=local_model, tokenizer=tokenizer, assistant_model=assistant_model, assistant_prefill=assistant_prefill, bedrock_runtime=bedrock_runtime, model_source=model_source)

        responses.append(response)
        whole_conversation.append(system_prompt)
        whole_conversation.append(prompt)
        whole_conversation.append(response_text)

        whole_conversation_metadata.append(f"Batch {batch_no}:")

        try:
            if "AWS" in model_source:
                output_tokens = response.usage_metadata.get('outputTokens', 0)
                input_tokens = response.usage_metadata.get('inputTokens', 0)
   
            elif "Gemini" in model_source:
                output_tokens = response.usage_metadata.candidates_token_count
                input_tokens = response.usage_metadata.prompt_token_count

            elif "Azure/OpenAI" in model_source:
                input_tokens = response.usage_metadata.get('inputTokens', 0)
                output_tokens = response.usage_metadata.get('outputTokens', 0)

            elif "Local" in model_source:
                if USE_LLAMA_CPP == "True":
                    output_tokens = response['usage'].get('completion_tokens', 0)
                    input_tokens = response['usage'].get('prompt_tokens', 0)

                if USE_LLAMA_CPP == "False":
                    input_tokens = num_transformer_input_tokens
                    output_tokens = num_transformer_generated_tokens

            else:
                input_tokens = 0
                output_tokens = 0

            whole_conversation_metadata.append("input_tokens: " + str(input_tokens) + " output_tokens: " + str(output_tokens))

        except KeyError as e:
            print(f"Key error: {e} - Check the structure of response.usage_metadata")

    return responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text

def call_llm_with_markdown_table_checks(batch_prompts: List[str],
                                        system_prompt: str,
                                        conversation_history: List[dict],
                                        whole_conversation: List[str], 
                                        whole_conversation_metadata: List[str],
                                        client: ai.Client | OpenAI,
                                        client_config: types.GenerateContentConfig,
                                        model_choice: str, 
                                        temperature: float,
                                        reported_batch_no: int,
                                        local_model: object,
                                        tokenizer:object,
                                        bedrock_runtime:boto3.Session.client,
                                        model_source:str,
                                        MAX_OUTPUT_VALIDATION_ATTEMPTS: int,
                                        assistant_prefill:str = "",                                       
                                        master:bool=False,
                                        CHOSEN_LOCAL_MODEL_TYPE:str=CHOSEN_LOCAL_MODEL_TYPE,
                                        random_seed:int=seed) -> Tuple[List[ResponseObject], List[dict], List[str], List[str], str]:
    """
    Call the large language model with checks for a valid markdown table.

    Parameters:
    - batch_prompts (List[str]): A list of prompts to be processed.
    - system_prompt (str): The system prompt.
    - conversation_history (List[dict]): The history of the conversation.
    - whole_conversation (List[str]): The complete conversation including prompts and responses.
    - whole_conversation_metadata (List[str]): Metadata about the whole conversation.
    - client (ai.Client | OpenAI): The client object for running Gemini or Azure/OpenAI API calls.
    - client_config (types.GenerateContentConfig): Configuration for the model.
    - model_choice (str): The choice of model to use.        
    - temperature (float): The temperature parameter for the model.
    - reported_batch_no (int): The reported batch number.
    - local_model (object): The local model to use.
    - tokenizer (object): The tokenizer to use.
    - bedrock_runtime (boto3.Session.client): The client object for boto3 Bedrock runtime.
    - model_source (str): The source of the model, whether in AWS, Gemini, or local.
    - MAX_OUTPUT_VALIDATION_ATTEMPTS (int): The maximum number of attempts to validate the output.
    - assistant_prefill (str, optional): The text to prefill the LLM response. Currently only working with AWS Claude calls.
    - master (bool, optional): Boolean to determine whether this call is for the master output table.
    - CHOSEN_LOCAL_MODEL_TYPE (str, optional): String to determine model type loaded.
    - random_seed (int, optional): The random seed used for LLM generation.

    Returns:
    - Tuple[List[ResponseObject], List[dict], List[str], List[str], str]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, the updated whole conversation metadata, and the response text.
    """

    call_temperature = temperature  # This is correct now with the fixed parameter name

    # Update Gemini config with the new temperature settings
    client_config = types.GenerateContentConfig(temperature=call_temperature, max_output_tokens=max_tokens, seed=random_seed)

    for attempt in range(MAX_OUTPUT_VALIDATION_ATTEMPTS):
        # Process requests to large language model
        responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(
            batch_prompts, system_prompt, conversation_history, whole_conversation, 
            whole_conversation_metadata, client, client_config, model_choice, 
            call_temperature, bedrock_runtime, model_source, reported_batch_no, local_model, tokenizer=tokenizer, master=master, assistant_prefill=assistant_prefill
        )

        stripped_response = response_text.strip()

        # Check if response meets our criteria (length and contains table) OR is "No change"
        if (len(stripped_response) > 120 and '|' in stripped_response) or stripped_response.lower().startswith("no change"):
            if stripped_response.lower().startswith("no change"):
                print(f"Attempt {attempt + 1} produced 'No change' response.")
            else:
                print(f"Attempt {attempt + 1} produced response with markdown table.")
            break  # Success - exit loop

        # Increase temperature for next attempt
        call_temperature = temperature + (0.1 * (attempt + 1))
        print(f"Attempt {attempt + 1} resulted in invalid table: {stripped_response}. "
            f"Trying again with temperature: {call_temperature}")

    else:  # This runs if no break occurred (all attempts failed)
        print(f"Failed to get valid response after {MAX_OUTPUT_VALIDATION_ATTEMPTS} attempts")

    return responses, conversation_history, whole_conversation, whole_conversation_metadata, stripped_response

def create_missing_references_df(basic_response_df: pd.DataFrame, existing_reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies references in basic_response_df that are not present in existing_reference_df.
    Returns a DataFrame with the missing references and the character count of their responses.

    Args:
        basic_response_df (pd.DataFrame): DataFrame containing 'Reference' and 'Response' columns.
        existing_reference_df (pd.DataFrame): DataFrame containing 'Response References' column.

    Returns:
        pd.DataFrame: A DataFrame with 'Missing Reference' and 'Response Character Count' columns.
                      'Response Character Count' will be 0 for empty strings and NaN for actual missing data.
    """
    # Ensure columns are treated as strings for robust comparison
    existing_references_unique = existing_reference_df['Response References'].astype(str).unique()

    # Step 1: Identify all rows from basic_response_df that correspond to missing references
    # We want the entire row to access the 'Response' column later
    missing_data_rows = basic_response_df[
        ~basic_response_df['Reference'].astype(str).isin(existing_references_unique)
    ].copy() # .copy() to avoid SettingWithCopyWarning

    # Step 2: Create the new DataFrame
    # Populate the 'Missing Reference' column directly
    missing_df = pd.DataFrame({
        'Missing Reference': missing_data_rows['Reference']
    })

    # Step 3: Calculate and add 'Response Character Count'
    # .str.len() works on Series of strings, handling empty strings (0) and NaN (NaN)
    missing_df['Response Character Count'] = missing_data_rows['Response'].str.len()

    # Optional: Add the actual response text for easier debugging/inspection if needed
    # missing_df['Response Text'] = missing_data_rows['Response']

    # Reset index to have a clean, sequential index for the new DataFrame
    missing_df = missing_df.reset_index(drop=True)

    return missing_df

def calculate_tokens_from_metadata(metadata_string:str, model_choice:str, model_name_map:dict):
    '''
    Calculate the number of input and output tokens for given queries based on metadata strings.

    Args:
        metadata_string (str): A string containing all relevant metadata from the string.
        model_choice (str): A string describing the model name
        model_name_map (dict): A dictionary mapping model name to source
    '''

    model_source = model_name_map[model_choice]["source"]

    # Regex to find the numbers following the keys in the "Query summary metadata" section
    # This ensures we get the final, aggregated totals for the whole query.
    input_regex = r"input_tokens: (\d+)"
    output_regex = r"output_tokens: (\d+)"

    # re.findall returns a list of all matching strings (the captured groups).
    input_token_strings = re.findall(input_regex, metadata_string)
    output_token_strings = re.findall(output_regex, metadata_string)

    # Convert the lists of strings to lists of integers and sum them up
    total_input_tokens = sum([int(token) for token in input_token_strings])
    total_output_tokens = sum([int(token) for token in output_token_strings])

    number_of_calls = len(input_token_strings)

    print(f"Found {number_of_calls} LLM call entries in metadata.")
    print("-" * 20)
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")

    return total_input_tokens, total_output_tokens, number_of_calls