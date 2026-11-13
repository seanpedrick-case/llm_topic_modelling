---
title: Large language model topic modelling
emoji: 📚
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: true
license: mit
short_description: Extract topics from consultations/other open text with LLMs
---

# Table of contents

- [Introduction](#introduction)
- [Using the app quickstart](#using-the-gui-interface-quickstart---test-the-app-with-built-in-examples)
- [Installation guide (local GUI version)](#installation-guide)
- [Using the GUI interface](#using-the-gui-interface)
- [Reading the output results file](#using-the-output-results-xlsx-file)


# Introduction

The Large language model topic modelling app is designed to help you analyse open text responses from consultations, surveys, or other sources of feedback. It uses advanced natural language processing techniques to identify key topics and themes within your data that can inform decision-making and reporting. Used correctly, the outputs can save you time in analysing large volumes of text data, and help you identify key themes and topics that may not be immediately obvious from a manual review. However, it is not a tool that can be used without human oversight. 

Here's how it can be used:

- Without a user-suggested list of topics, the tool can generate an exploratory AI-suggested list of topics/themes from your data that can be used to guide your analysis.
- If you have an approved list of topics/themes, you can use this app to assign your data to these topics/themes and generate a summary of the results that should then be verified before further use. **Important: Your topic names should be as clear as possible, and contain all the information necessary for someone unfamiliar with your consultation to fully undertand the meaning. Also, a maximum of 35 topics is highly recommended.**
- You can also create analyses based on groups of data (e.g., by location, department, or other categories) to see how topics/themes vary across different segments of your data.
- You can analyse the sentiment of responses (e.g., positive, negative, neutral) to understand how people feel about different topics/themes.

**Note that the AI may not always agree with a human assessment of the topics/themes, so you should check the outputs carefully before using them in any reports or decision-making. Also, never copy and paste the outputs directly into reports without checking them first, as the AI may hallucinate or create biased outputs.**

When accounting for ambiguous topics and analyst errors, the system has been shown to agree ~80% of the time with human analysts for real surveys where there are a moderate number of topics (~35 or less) to choose from. With this in mind, if providing a list of topics, it is important that their meaning is as clear as possible, and that their meaning does not overlap with other topics in the list.

Below is an overview of how to access and use the app.

NOTE: Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.

## Using the GUI interface quickstart - Test the app with built-in examples

The LLM topic modelling app comes with built-in example datasets to help you get started quickly. To test the application:

1. Try the example version of the app on Hugging Face [here](https://huggingface.co/spaces/seanpedrickcase/llm_topic_modelling), or [follow the installation instructions below to install locally](#installation-guide)
2. Open the **'All in one topic extraction and summarisation'** tab
3. Scroll down to the **'Try topic extraction and summarisation with an example dataset'** section
4. Select one of the provided examples:
   - **Main Street construction consultation** - A simple consultation dataset
   - **Case notes for young people** - Social care case notes with grouped data
   - **Main Street construction consultation with suggested topics** - Example showing how to provide candidate topics
   - **Case notes grouped by person with suggested topics** - Example using grouped data with suggested topics
   - **Case notes structured summary with suggested topics** - Example of structured summaries

5. Click the example to load the data and populate the form fields
6. Click the **'Extract topics...'** button to run the analysis
7. Download the Excel output file to see the results (outputs explained [here](#using-the-output-results-xlsx-file))

Further details on use are below in the [Using the GUI interface guide](#using-the-gui-interface)

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

This will install the CPU version of llama-cpp-python. If you want GPU support, first I would try using pip install with specific wheels for your system: See files in https://github.com/abetlen/llama-cpp-python/releases . If you are still struggling, see here for more details on installation here: https://llama-cpp-python.readthedocs.io/en/latest

#### Install mcp version of gradio

You can install an mcp-compatible version of gradio for this app with the following command:

```bash
pip install .[mcp]
```

2. **Install from a requirements file (not recommended)**

The repo provides several requirements files that are relevant for different situations. To start, I advise installing using the **requirements_lightweight.txt** file, which installs the app with access to all cloud provider or local inference (e.g. llama server, vLLM server) APIs. This approach is much simpler as a first step, and avoids issues with potentially complicated llama-cpp-python installation and GPU management described below.

If you want to run models locally 'in app', then you have two further requirements files to choose from:

- **requirements_cpu.txt**: Used for Python 3.1x CPU-only environments.
- **requirements_gpu.txt**: Used for Python 3.1x GPU-enabled environments.

Example The below instructions will guide you in how to install the GPU-enabled version of the app for local inference.

**Install packages for local model 'in-app' inference from the requirements file:**
    ```bash
    pip install -r requirements_gpu.txt
    ```
    *This command reads every package name listed in the file and installs it into your `.venv` environment.*

NOTE: If default llama-cpp-python installation does not work when installing from the above, go into the requirements_gpu.txt file and uncomment the lines to install a wheel for llama-cpp-python 0.3.16 relevant to your system.

### Step 4: Verify CUDA compatibility (if using a GPU environment)

Install the relevant toolkit for CUDA from here: https://developer.nvidia.com/

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

This is a great way to understand the output format and what the analysis can do before running with your own data.

# Using the GUI interface

The **'All in one topic extraction and summarisation'** tab is the main interface for analyzing your data. Follow these steps:

### Step 1: Upload your data file

- Click on **'Choose Excel or csv files'** to select your input file
- Supported formats: `.xlsx`, `.xls`, `.csv`, or `.parquet`
- You can upload single or multiple files at once
- The file(s) will be processed together

### Step 2: Select the text column to analyse

- Use the **'Select the open text column of interest'** dropdown menu
- This dropdown automatically populates with all available columns from your uploaded file(s)
- Select the column containing the text you want to analyse for topics
- The column should contain open-ended text responses or comments

### Step 3: (Optional) Provide context to the LLM

- In the **'Write context to the large language model'** textbox, enter up to one sentence describing your task
- Example: *"Consultation for the construction of flats on Main Street"* or *"Social Care case notes for young people"*
- This helps the language model understand the domain and provide more relevant topic analysis
- If left blank, the model will still analyse the text but may use more generic topic categories

### Step 4: (Optional) Group analysis by another column

- If you want separate analysis for different groups within your data, click the **'Group analysis by values in another column'** accordion
- Select the grouping column from the dropdown (e.g., by "Client", "Location", "Department")
- The analysis will be performed separately for each unique value in that column
- Results will show topic summaries broken down by group

### Step 5: (Optional) Provide suggested topics

- Click the **'Provide list of suggested topics'** accordion to expand this section
- Upload a CSV file containing your candidate topics
- **File format requirements:**
  - Must have at least one column with a header
  - Topic names should be listed below the header
  - Optional column headers:
    - `General topic` - used for top-level topic categories
    - `Subtopic` - used for more specific sub-categories
    - A third column can be included for topic descriptions
- Using suggested topics helps guide the model toward relevant categories for your specific use case
- **Note:** If using grouped analysis and suggested topics together, use the headers 'General topic' and/or 'Subtopic' for best results

### Step 6: (Optional) Specify sentiment analysis categories

- Click the **'Response sentiment analysis'** accordion
- By default, responses are categorised as "Negative" or "Positive"
- You can customise these categories by specifying different sentiment types
- This will add an additional dimension to your topic analysis

### Step 7: Set batch size (optional)

- Adjust the **'Number of responses to submit in a single LLM query'** slider
- The default batch size is already optimised for most use cases
- Larger batch sizes may be faster but require more processing
- Smaller batch sizes may be more reliable for very sensitive or nuanced content
- Range: 1-50 responses per batch

### Step 8: Run the analysis

- Click the **'Extract topics...'** button to start the analysis
- A progress indicator will show the number of files prepared
- Processing time depends on file size, batch size, and your selected LLM model
- Output messages will display any warnings or important information
- Once complete, the Excel output file(s) will be available for download

# Using the output results (XLSX) file

The analysis results are provided as a multi-sheet Excel file (.xlsx). Each sheet contains different aspects of the analysis to help you explore and understand the topics identified in your data.

### Sheet: Cover Sheet

- **Overview**: Provides a summary of the analysis parameters and data coverage
- **Contains**:
  - The name of the column analyzed
  - Number of response rows that contained text
  - Number of responses that had at least one topic assigned
  - This helps you understand data quality and completeness

### Sheet: Overall summary

- **Overview**: A high-level text summary of the entire analysis
- **Contains**: A narrative description of the main findings, key themes, and significant patterns across all responses
- **Use this to**: Get a quick overview of the key insights without reviewing individual topics

### Sheet: Topic summary

- **Overview**: A structured summary of all identified topics
- **Contains**:
  - General topic and subtopic categories (if applicable)
  - Number of responses relevant to each topic (Column E)
  - Summary statistics about topic distribution
- **Use this to**: Understand which topics were most prevalent and how topics are distributed across your dataset
- **Key column**: **Column E** shows the count of responses assigned to each topic, helping you prioritise which topics have the most feedback

### Sheet: Response level data

- **Overview**: Detailed response-by-response breakdown
- **Contains**:
  - Each topic assigned to a response on a separate row
  - Original response text
  - Topic assignments and any associated metadata
- **Use this to**: Review individual responses and see exactly which topics were assigned to specific comments
- **Useful for**: Validation and quality checking

### Sheet: Topic response pivot table

- **Overview**: A cross-tabulation view of responses and topics
- **Contains**:
  - Full text of original responses, each on a separate row
  - Topic names across the columns
  - Values of `1` where a response is relevant to a topic, empty cells where not relevant
- **Format**: This is essentially a matrix with responses as rows and topics as columns
- **Use this to**: Easily see which responses match which topics and identify response sets matching specific topic combinations

### Sheet: General topic response pivot table

- **Overview**: A cross-tabulation view of responses and general topics
- **Contains**:
  - Full text of original responses, each on a separate row
  - Topic names across the columns
  - Values of `1` (or more) where a response is relevant to a topic, empty cells where not relevant
- **Format**: This is essentially a matrix with responses as rows and topics as columns
- **Use this to**: Easily see which responses match which topics and identify response sets matching specific topic combinations

### Sheet: Missing responses

- **Overview**: Responses that did not receive any topic assignment
- **Contains**:
  - All comments without assigned topics
  - Character length of each response
- **Use this to**: 
  - Identify comments that should have been categorised but weren't
  - Spot very short responses that may not have had enough text for analysis
  - Validate model performance and identify potential edge cases
  - Potentially refine your suggested topics or context to improve coverage

### Sheet: Original data

- **Overview**: Your original input data for reference
- **Contains**:
  - All data from your input file
  - A **Reference column (Column A)** that provides unique identifiers for each row
- **Use this to**: Cross-reference analysis results back to your original data
- **Reference column**: Use the values in Column A to link findings from other sheets back to the original source data

### Tips for interpreting results

- **Topic Prevalence**: Look at Column E in the "Topic summary" sheet to prioritise which topics to focus on - topics with higher counts represent broader concerns
- **Completeness**: The "Cover Sheet" and "Missing responses" sheets help you assess whether the model successfully analysed your data
- **Validation**: Spot-check responses in the "Response level data" sheet to ensure topics are being assigned appropriately
- **Pivot Table**: The "Topic response pivot table" is useful for exporting to other tools or creating visualisations of topic coverage
- **Group Analysis**: If you grouped your analysis by another variable, separate Excel files or sheets will be created for each group
