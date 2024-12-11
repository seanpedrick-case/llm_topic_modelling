from typing import TypeVar
import torch.cuda
import os
import time
import spaces
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from tools.helper_functions import RUN_LOCAL_MODEL

torch.cuda.empty_cache()

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

model_type = None # global variable setup

full_text = "" # Define dummy source text (full text) just to enable highlight function to load

model = [] # Define empty list for model functions to run
tokenizer = [] #[] # Define empty list for model functions to run

local_model_type = "Gemma 2b"

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
threads = torch.get_num_threads() # 8
print("CPU threads:", threads)

temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repetition_penalty: float = 1.2 # Mild repetition penalty to prevent repeating table rows
last_n_tokens: int = 512
max_new_tokens: int = 4096 # 200
seed: int = 42
reset: bool = True
stream: bool = False
threads: int = threads
batch_size:int = 256
context_length:int = 12288
sample = True


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

###
# Load local model
###

@spaces.GPU
def load_model(local_model_type:str, gpu_layers:int, max_context_length:int, gpu_config:llama_cpp_init_config_gpu=gpu_config, cpu_config:llama_cpp_init_config_cpu=cpu_config, torch_device:str=torch_device):
    '''
    Load in a model from Hugging Face hub via the transformers package, or using llama_cpp_python by downloading a GGUF file from Huggingface Hub. 
    '''
    print("Loading model ", local_model_type)

    if local_model_type == "Gemma 2b":
        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
            gpu_config.update_context(max_context_length)
            print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU. And a maximum context length of ", gpu_config.n_ctx)
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

            # Update context length according to slider
            gpu_config.update_context(max_context_length)
            cpu_config.update_context(max_context_length)

            print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU. And a maximum context length of ", gpu_config.n_ctx)

        #print(vars(gpu_config))
        #print(vars(cpu_config))

        def get_model_path():
            repo_id = os.environ.get("REPO_ID", "lmstudio-community/gemma-2-2b-it-GGUF")# "bartowski/Llama-3.2-3B-Instruct-GGUF") # "lmstudio-community/gemma-2-2b-it-GGUF")#"QuantFactory/Phi-3-mini-128k-instruct-GGUF")
            filename = os.environ.get("MODEL_FILE", "gemma-2-2b-it-Q8_0.gguf") # )"Llama-3.2-3B-Instruct-Q5_K_M.gguf") #"gemma-2-2b-it-Q8_0.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")
            model_dir = "model/gemma" #"model/phi"  # Assuming this is your intended directory

            # Construct the expected local path
            local_path = os.path.join(model_dir, filename)

            if os.path.exists(local_path):
                print(f"Model already exists at: {local_path}")
                return local_path
            else:
                print(f"Checking default Hugging Face folder. Downloading model from Hugging Face Hub if not found")
                return hf_hub_download(repo_id=repo_id, filename=filename)
            
        model_path = get_model_path()        

        try:
            print(vars(gpu_config))
            llama_model = Llama(model_path=model_path, **vars(gpu_config)) #  type_k=8, type_v = 8, flash_attn=True, 
        
        except Exception as e:
            print("GPU load failed")
            print(e)
            llama_model = Llama(model_path=model_path, type_k=8, **vars(cpu_config)) # type_v = 8, flash_attn=True, 

        tokenizer = []

    model = llama_model
    tokenizer = tokenizer
    local_model_type = local_model_type

    load_confirmation = "Finished loading model: " + local_model_type

    print(load_confirmation)
    return local_model_type, load_confirmation, local_model_type, model, tokenizer

###
# Load local model
###
if RUN_LOCAL_MODEL == "1":
    print("Loading model")
    local_model_type, load_confirmation, local_model_type, model, tokenizer = load_model(local_model_type, gpu_layers, context_length, gpu_config, cpu_config, torch_device)
    print("model loaded:", model)


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

@spaces.GPU
def call_llama_cpp_model(formatted_string:str, gen_config:str):
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