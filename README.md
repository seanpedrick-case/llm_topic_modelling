---
title: Large language model topic modelling
emoji: 📝
colorFrom: purple
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: true
license: agpl-3.0
---

# Large language model topic modelling

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
    cd example-repo
    ```
-----

### Step 2: Create and Activate a Virtual Environment

A virtual environment is a self-contained directory that holds a specific Python interpreter and its own set of installed packages. This is crucial for isolating your project's dependencies.

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

Now that your virtual environment is active, you can install all the required packages listed in the relevant project `requirements.txt` file using `pip`.

1. **Choose the relevant requirements file**

****NOTE:** To start, I advise installing using the **requirements_no_local.txt** file, which installs the app without local model inference. This approach is much simpler as a first step, and avoids issues with potentially complicated llama-cpp-python installation and GPU management described below.

Llama-cpp-python version 3.16 is compatible with Gemma 3 and GPT-OSS models, but does not at the time of writing have relevant wheels for CPU inference or for Windows. A sister repository contains [llama-cpp-python 3.16 wheels for Python version 3.11/10](https://github.com/seanpedrick-case/llama-cpp-python-whl-builder/releases/tag/v0.1.0) so that users can avoid having to build the package from source. If you prefer to build from source, then please refer to the llama-cpp-python documentation [here](https://github.com/abetlen/llama-cpp-python). I also have a guide to building the package on a Windows system [here](https://github.com/seanpedrick-case/llm_topic_modelling/blob/main/windows_install_llama-cpp-python.txt).

The repo provides several requirements files that are relevant for different situations. I would advise using requirements_gpu.txt for GPU environments, and requirements_cpu.txt for CPU environments:

- **requirements_no_local.txt**: Can be used to install the app without local model inference for a more lightweight installation.
- **requirements_gpu.txt**: Used for Python 3.11 GPU-enabled environments. Uncomment the requirements under 'Windows' for Windows compatibility (CUDA 12.4).
- **requirements_cpu.txt**: Used for Python 3.11 CPU-only environments. Uncomment the requirements under 'Windows' for Windows compatibility. Make sure you have [Openblas](https://github.com/OpenMathLib/OpenBLAS) installed!
- **requirements.txt**: Used for the Python 3.10 GPU-enabled environment on Hugging Face spaces (CUDA 12.4).

The below instructions will guide you in how to install the GPU-enabled version of the app for local inference.

2.  **Install packages from the requirements file:**
    ```bash
    pip install -r requirements_gpu.txt
    ```
    *This command reads every package name listed in the file and installs it into your `.venv` environment.*

NOTE: If default llama-cpp-python installation does not work when installing from the above, go into the requirements_gpu.txt file and uncomment the lines to install a wheel for llama-cpp-python 0.3.16 relevant to your system.

You're all set\! ✅ Your project is cloned, and all dependencies are installed in an isolated environment.

When you are finished working, you can leave the virtual environment by simply typing:

```bash
deactivate
```

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
