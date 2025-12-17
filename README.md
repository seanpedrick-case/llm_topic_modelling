---
title: Large language model topic modelling
emoji: 📚
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: true
license: agpl-3.0
short_description: Create thematic summaries for open text data with LLMs
---

# Large language model topic modelling

Version: 0.7.0

Extract topics and summarise outputs using Large Language Models (LLMs, Gemma 3 4b/GPT-OSS 20b if local (see tools/config.py to modify), Gemini, Azure, or AWS Bedrock models (e.g. Claude, Nova models). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and a topic summary. Instructions on use can be found in the README.md file. You can try out examples by clicking on one of the example datasets on the main app page, which will show you example outputs from a local model run. API keys for AWS, Azure, and Gemini services can be entered on the settings page (note that Gemini has a free public API).

NOTE: Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.

Basic use:
1. On the front page, choose your model for inference. Gemma 3/GPT-OSS will use 'on-device' inference. Calls to Gemini or AWS will require an API key that can be input on the 'LLM and topic extraction' page.
1. Upload a csv/xlsx/parquet file containing at least one open text column.
2. Select the relevant open text column from the dropdown.
3. If you have your own suggested (zero shot) topics, upload this (see examples folder for an example file)
4. Write a one sentence description of the consultation/context of the open text.
5. Click 'Extract topics, deduplicate, and summarise'. This will run through the whole analysis process from topic extraction, to topic deduplication, to topic-level and overall summaries.
6. A summary xlsx file workbook will be created on the front page in the box 'Overall summary xlsx file'. This will combine all the results from the different processes into one workbook.

# Installation guide

Here is a step-by-step guide to clone the repository, create a virtual environment, and install dependencies from the relevant `requirements` file. This guide assumes you have **Git** and **Python 3.11** installed.

-----

### Step 1: Clone the Git Repository

First, you need to copy the project files to your local machine. Navigate to the directory where you want to store the project using the `cd` (change directory) command. Then, use `git clone` with the repository's URL.

1.  **Clone the repo:**

    ```bash
    git clone https://github.com/seanpedrick-case/llm_topic_modelling.git
    ```

2.  **Navigate into the new project folder:**

    ```bash
    cd llm_topic_modelling
    ```
-----

### Step 2: Create and Activate a Virtual Environment

A virtual environment is a self-contained directory that holds a specific Python interpreter and its own set of installed packages. This is crucial for isolating your project's dependencies.

NOTE: Alternatively you could also create and activate a Conda environment instead of using venv below.

1.  **Create the virtual environment:** We'll use Python's built-in `venv` module. It's common practice to name the environment folder `.venv`.

    ```bash
    python -m venv .venv
    ```

    *This command tells Python to create a new virtual environment in a folder named `.venv`.*

2.  **Activate the environment:** You must "activate" the environment to start using it. The command differs based on your operating system and shell.

      * **On macOS / Linux (bash/zsh):**

        ```bash
        source .venv/bin/activate
        ```

      * **On Windows (Command Prompt):**

        ```bash
        .\.venv\Scripts\activate
        ```

      * **On Windows (PowerShell):**

        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```

    You'll know it's active because your command prompt will be prefixed with `(.venv)`.

-----

### Step 3: Install Dependencies

Now that your virtual environment is active, you can install all the required packages. Here you have two options, install from the pyproject.toml file (recommended), or install from requirements files.

1. **Install from pyproject.toml (recommended)**

You can install the 'lightweight' version of the app to access all available cloud provider or local inference (e.g. llama server, vLLM server) APIs. This version will not allow you to run local models such as Gemma 12b or GPT-OSS-20b 'in-app', i.e. accessible from the GUI interface directly. However, you will have access to AWS, Gemma, or Azure/OpenAI models with appropriate API keys. Use the following command in your environment to install the relevant packages:

```bash
pip install .
```

#### Install torch (optional)

If you want to run inference with transformers with full/quantised models, and the associated Unsloth package, you can run the following command for CPU inference. For GPU inference, please refer to the requirements_gpu.txt guide, and the 'Install from a requirements file' section below:

```bash
pip install .[torch]
```

#### Install llama-cpp-python (optional)

You can run quantised GGUF models in-app using llama-cpp-python. However, installation of this package is not always straightforward, particularly considering that wheels are not available for the latest version apart from for linux. This package is not being updated regularly, and so support may be removed for this package in future. Long term I would advise instead looking into running GGUF models using llama-server and calling the API from this app using the lightweight version (details here: https://github.com/ggml-org/llama.cpp).

If you do want to install llama-cpp-python in app, first try the following command:

```bash
pip install .[llamacpp]
```

This will install the CPU version of llama-cpp-python. If you want GPU support, first I would try using pip install with specific wheels for your system, e.g. for Linux: See files in https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.16-cu124 . If you are still struggling, see here for more details on installation here: https://llama-cpp-python.readthedocs.io/en/latest

**NOTE:** A sister repository contains [llama-cpp-python 3.16 wheels for Python version 3.11/10](https://github.com/seanpedrick-case/llama-cpp-python-whl-builder/releases/tag/v0.1.0) so that users can avoid having to build the package from source. I also have a guide to building the package on a Windows system [here](https://github.com/seanpedrick-case/llm_topic_modelling/blob/main/windows_install_llama-cpp-python.txt).

#### Install mcp version of gradio

You can install an mcp-compatible version of gradio for this app with the following command:

```bash
pip install .[mcp]
```

2. **Install from a requirements file (not recommended)**

The repo provides several requirements files that are relevant for different situations. To start, I advise installing using the **requirements_lightweight.txt** file, which installs the app with access to all cloud provider or local inference (e.g. llama server, vLLM server) APIs. This approach is much simpler as a first step, and avoids issues with potentially complicated llama-cpp-python installation and GPU management described below.

If you want to run models locally 'in app', then you have two further requirements files to choose from:

- **requirements_cpu.txt**: Used for Python 3.11 CPU-only environments. Uncomment the requirements under 'Windows' for Windows compatibility. Make sure you have [Openblas](https://github.com/OpenMathLib/OpenBLAS) installed!
- **requirements_gpu.txt**: Used for Python 3.11 GPU-enabled environments. Uncomment the requirements under 'Windows' for Windows compatibility (CUDA 12.4).

Example The below instructions will guide you in how to install the GPU-enabled version of the app for local inference.

**Install packages for local model 'in-app' inference from the requirements file:**
    ```bash
    pip install -r requirements_gpu.txt
    ```
    *This command reads every package name listed in the file and installs it into your `.venv` environment.*

NOTE: If default llama-cpp-python installation does not work when installing from the above, go into the requirements_gpu.txt file and uncomment the lines to install a wheel for llama-cpp-python 0.3.16 relevant to your system.

### Step 4: Verify CUDA compatibility (if using a GPU environment)

Install the relevant toolkit for CUDA 12.4 from here: https://developer.nvidia.com/cuda-12-4-0-download-archive

Restart your computer

Ensure you have the latest drivers for your NVIDIA GPU. Check your current version and memory availability by running nvidia-smi

In command line, CUDA compatibility can be checked by running nvcc --version


### Step 5: Ensure you have compatible NVIDIA drivers

Make sure you have the latest NVIDIA drivers installed on your system for your GPU (be careful in particular if using WSL that you have drivers compatible with this). Official drivers can be found here: https://www.nvidia.com/en-us/drivers

Current drivers can be found by running nvidia-smi in command line

### Step 6: Run the app

Go to the app project directory. Run python app.py

### Step 7: (optional) change default configuration

A number of configuration options can be seen the tools/config.py file. You can either pass in these variables as environment variables, or you can create a file in config/app_config.env to read this into the app on initialisation.
