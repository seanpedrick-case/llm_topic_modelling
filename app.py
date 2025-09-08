import spaces
import os
import gradio as gr
import pandas as pd
from datetime import datetime
from tools.helper_functions import put_columns_in_df, get_connection_params, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, view_table, empty_output_vars_extract_topics, empty_output_vars_summarise, load_in_previous_reference_file, join_cols_onto_reference_df, load_in_previous_data_files, load_in_data_file, load_in_default_cost_codes, reset_base_dataframe, update_cost_code_dataframe_from_dropdown_select, df_select_callback_cost, enforce_cost_codes, _get_env_list, move_overall_summary_output_files_to_front_page
from tools.aws_functions import upload_file_to_s3, download_file_from_s3
from tools.llm_api_call import modify_existing_output_tables, wrapper_extract_topics_per_column_value
from tools.dedup_summaries import sample_reference_table_summaries, summarise_output_topics, deduplicate_topics, overall_summary
from tools.combine_sheets_into_xlsx import collect_output_csvs_and_create_excel_output
from tools.custom_csvlogger import CSVLogger_custom
from tools.auth import authenticate_user
from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt, verify_titles_prompt, verify_titles_system_prompt, two_para_summary_format_prompt, single_para_summary_format_prompt
from tools.verify_titles import verify_titles
from tools.config import RUN_AWS_FUNCTIONS, HOST_NAME, ACCESS_LOGS_FOLDER, FEEDBACK_LOGS_FOLDER, USAGE_LOGS_FOLDER, RUN_LOCAL_MODEL,  FILE_INPUT_HEIGHT, GEMINI_API_KEY, model_full_names, BATCH_SIZE_DEFAULT, CHOSEN_LOCAL_MODEL_TYPE, LLM_SEED, COGNITO_AUTH, MAX_QUEUE_SIZE, MAX_FILE_SIZE, GRADIO_SERVER_PORT, ROOT_PATH, INPUT_FOLDER, OUTPUT_FOLDER, S3_LOG_BUCKET, CONFIG_FOLDER, GRADIO_TEMP_DIR, MPLCONFIGDIR, model_name_map, GET_COST_CODES, ENFORCE_COST_CODES, DEFAULT_COST_CODE, COST_CODES_PATH, S3_COST_CODES_PATH, OUTPUT_COST_CODES_PATH, SHOW_COSTS, SAVE_LOGS_TO_CSV, SAVE_LOGS_TO_DYNAMODB, ACCESS_LOG_DYNAMODB_TABLE_NAME, USAGE_LOG_DYNAMODB_TABLE_NAME, FEEDBACK_LOG_DYNAMODB_TABLE_NAME, LOG_FILE_NAME, FEEDBACK_LOG_FILE_NAME, USAGE_LOG_FILE_NAME, CSV_ACCESS_LOG_HEADERS, CSV_FEEDBACK_LOG_HEADERS, CSV_USAGE_LOG_HEADERS, DYNAMODB_ACCESS_LOG_HEADERS, DYNAMODB_FEEDBACK_LOG_HEADERS, DYNAMODB_USAGE_LOG_HEADERS, S3_ACCESS_LOGS_FOLDER, S3_FEEDBACK_LOGS_FOLDER, S3_USAGE_LOGS_FOLDER, AWS_ACCESS_KEY, AWS_SECRET_KEY, SHOW_EXAMPLES, HF_TOKEN

def ensure_folder_exists(output_folder:str):
    """Checks if the specified folder exists, creates it if not."""   

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        pass
            #print(f"The {output_folder} folder already exists.")

ensure_folder_exists(CONFIG_FOLDER)
ensure_folder_exists(OUTPUT_FOLDER)
ensure_folder_exists(INPUT_FOLDER)
ensure_folder_exists(GRADIO_TEMP_DIR)
ensure_folder_exists(MPLCONFIGDIR)
ensure_folder_exists(FEEDBACK_LOGS_FOLDER)
ensure_folder_exists(ACCESS_LOGS_FOLDER)
ensure_folder_exists(USAGE_LOGS_FOLDER)

# Convert string environment variables to string or list
if SAVE_LOGS_TO_CSV == "True": SAVE_LOGS_TO_CSV = True 
else: SAVE_LOGS_TO_CSV = False
if SAVE_LOGS_TO_DYNAMODB == "True": SAVE_LOGS_TO_DYNAMODB = True 
else: SAVE_LOGS_TO_DYNAMODB = False

if CSV_ACCESS_LOG_HEADERS: CSV_ACCESS_LOG_HEADERS = _get_env_list(CSV_ACCESS_LOG_HEADERS)
if CSV_FEEDBACK_LOG_HEADERS: CSV_FEEDBACK_LOG_HEADERS = _get_env_list(CSV_FEEDBACK_LOG_HEADERS)
if CSV_USAGE_LOG_HEADERS: CSV_USAGE_LOG_HEADERS = _get_env_list(CSV_USAGE_LOG_HEADERS)

if DYNAMODB_ACCESS_LOG_HEADERS: DYNAMODB_ACCESS_LOG_HEADERS = _get_env_list(DYNAMODB_ACCESS_LOG_HEADERS)
if DYNAMODB_FEEDBACK_LOG_HEADERS: DYNAMODB_FEEDBACK_LOG_HEADERS = _get_env_list(DYNAMODB_FEEDBACK_LOG_HEADERS)
if DYNAMODB_USAGE_LOG_HEADERS: DYNAMODB_USAGE_LOG_HEADERS = _get_env_list(DYNAMODB_USAGE_LOG_HEADERS)

today_rev = datetime.now().strftime("%Y%m%d")

if RUN_LOCAL_MODEL == "1":
    default_model_choice = CHOSEN_LOCAL_MODEL_TYPE
elif RUN_AWS_FUNCTIONS == "1":
    default_model_choice = "anthropic.claude-3-haiku-20240307-v1:0"
else:
    default_model_choice = "gemini-2.5-flash"

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Default(primary_hue="blue"), fill_width=True)

with app:

    ###
    # STATE VARIABLES
    ###

    text_output_file_list_state = gr.Dropdown([], allow_custom_value=True, visible=False, label="text_output_file_list_state")
    text_output_modify_file_list_state = gr.Dropdown([], allow_custom_value=True, visible=False, label="text_output_modify_file_list_state")
    log_files_output_list_state = gr.Dropdown([], allow_custom_value=True, visible=False, label="log_files_output_list_state")
    first_loop_state = gr.Checkbox(True, visible=False)
    second_loop_state = gr.Checkbox(False, visible=False)
    modified_unique_table_change_bool = gr.Checkbox(True, visible=False) # This boolean is used to flag whether a file upload should change just the modified unique table object on the second tab

    file_data_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="file_data_state", visible=False, type="pandas")
    master_topic_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_topic_df_state", visible=False, type="pandas")
    master_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_unique_topics_df_state", visible=False, type="pandas")
    master_reference_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_reference_df_state", visible=False, type="pandas")
    missing_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="missing_df_state", visible=False, type="pandas")

    master_modify_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_modify_unique_topics_df_state", visible=False, type="pandas")
    master_modify_reference_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_modify_reference_df_state", visible=False, type="pandas")   

    # Blank placeholder for conversation metadata textbox, as logging file output can get too long for large amounts of calls
    conversation_metadata_textbox_placeholder = gr.Textbox(value="", label="Query metadata - usage counts and other parameters", lines=8, visible=False) 
 
    session_hash_state = gr.Textbox(visible=False, value=HOST_NAME)
    output_folder_state = gr.Textbox(visible=False, value=OUTPUT_FOLDER)
    input_folder_state = gr.Textbox(visible=False, value=INPUT_FOLDER)

    # s3 bucket name
    s3_default_bucket = gr.Textbox(label = "Default S3 bucket", value=S3_LOG_BUCKET, visible=False)
    s3_log_bucket_name = gr.Textbox(visible=False, value=S3_LOG_BUCKET)

    # Logging variables
    access_logs_state = gr.Textbox(label= "access_logs_state", value=ACCESS_LOGS_FOLDER + LOG_FILE_NAME, visible=False)
    access_s3_logs_loc_state = gr.Textbox(label= "access_s3_logs_loc_state", value=S3_ACCESS_LOGS_FOLDER, visible=False)
    feedback_logs_state = gr.Textbox(label= "feedback_logs_state", value=FEEDBACK_LOGS_FOLDER + FEEDBACK_LOG_FILE_NAME, visible=False)
    feedback_s3_logs_loc_state = gr.Textbox(label= "feedback_s3_logs_loc_state", value=S3_FEEDBACK_LOGS_FOLDER, visible=False)
    usage_logs_state = gr.Textbox(label= "usage_logs_state", value=USAGE_LOGS_FOLDER + USAGE_LOG_FILE_NAME, visible=False)
    usage_s3_logs_loc_state = gr.Textbox(label= "usage_s3_logs_loc_state", value=S3_USAGE_LOGS_FOLDER, visible=False)

    # Logging for input / output tokens
    input_tokens_num = gr.Textbox('0', visible=False, label="Total input tokens")
    output_tokens_num = gr.Textbox('0', visible=False, label="Total output tokens")
    number_of_calls_num = gr.Textbox('0', visible=False, label="Total LLM calls")

    # Summary state objects
    summary_reference_table_sample_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="summary_reference_table_sample_state", visible=False, type="pandas")
    master_reference_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_reference_df_revised_summaries_state", visible=False, type="pandas")
    master_unique_topics_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_unique_topics_df_revised_summaries_state", visible=False, type="pandas")
    summarised_output_df = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="summarised_output_df", visible=False, type="pandas")
    summarised_references_markdown = gr.Markdown("", visible=False)
    summarised_outputs_list = gr.Dropdown(value= list(), choices= list(), visible=False, label="List of summarised outputs", allow_custom_value=True)
    latest_summary_completed_num = gr.Number(0, visible=False)

    summary_xlsx_output_files_list = gr.Dropdown(value= list(), choices= list(), visible=False, label="List of xlsx summary output files", allow_custom_value=True)

    original_data_file_name_textbox = gr.Textbox(label = "Reference data file name", value="", visible=False)
    working_data_file_name_textbox = gr.Textbox(label = "Working data file name", value="", visible=False)
    unique_topics_table_file_name_textbox = gr.Textbox(label="Unique topics data file name textbox", visible=False)

    model_name_map_state = gr.JSON(model_name_map, visible=False, label="model_name_map_state")

    # Cost code elements
    s3_default_cost_codes_file = gr.Textbox(label = "Default cost centre file", value=S3_COST_CODES_PATH, visible=False)
    default_cost_codes_output_folder_location = gr.Textbox(label = "Output default cost centre location", value=OUTPUT_COST_CODES_PATH, visible=False)
    enforce_cost_code_textbox = gr.Textbox(label = "Enforce cost code textbox", value=ENFORCE_COST_CODES, visible=False)
    default_cost_code_textbox = gr.Textbox(label = "Default cost code textbox", value=DEFAULT_COST_CODE, visible=False)

    # Placeholders for elements that may be made visible later below depending on environment variables
    cost_code_dataframe_base = gr.Dataframe(value=pd.DataFrame(), row_count = (0, "dynamic"), label="Cost codes", type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, show_search='filter', wrap=True, max_height=200, visible=False)
    cost_code_dataframe = gr.Dataframe(value=pd.DataFrame(), type="pandas", visible=False, wrap=True)
    cost_code_choice_drop = gr.Dropdown(value=DEFAULT_COST_CODE, label="Choose cost code for analysis. Please contact Finance if you can't find your cost code in the given list.", choices=[DEFAULT_COST_CODE], allow_custom_value=False, visible=False)

    latest_batch_completed = gr.Number(value=0, label="Number of files prepared", interactive=False, visible=False)
    # Duplicate version of the above variable for when you don't want to initiate the summarisation loop
    latest_batch_completed_no_loop = gr.Number(value=0, label="Number of files prepared", interactive=False, visible=False)

    ###
    # UI LAYOUT
    ###

    gr.Markdown("""# Large language model topic modelling

    Extract topics and summarise outputs using Large Language Models (LLMs, Gemma 3 4b/GPT-OSS 20b if local (see tools/config.py to modify), Gemini 2.5, or Bedrock models (e.g. Claude 3 Haiku, Claude Sonnet 3.7, Nova models). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and relevant text rows related to them. The prompts are designed for topic modelling public consultations, but they can be adapted to different contexts (see the LLM settings tab to modify).
    
    Instructions on use can be found in the README.md file. Try it out with this [dummy development consultation dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/tree/main), which you can also try with [zero-shot topics](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/tree/main). Try also this [dummy case notes dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_case_notes/tree/main).

    You can use an AWS Bedrock model (paid), or Gemini (a free API for Flash). The use of Gemini requires an API key. To set up your own Gemini API key, go [here](https://aistudio.google.com/app/u/1/plan_information).

    NOTE: Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.""")
    
    with gr.Tab(label="1. Extract topics"):
        gr.Markdown("""### Choose a tabular data file (xlsx, csv, parquet) of open text to extract topics from.""")
        with gr.Row():
            model_choice = gr.Dropdown(value = default_model_choice, choices = model_full_names, label="LLM model", multiselect=False)        

        with gr.Accordion("Upload xlsx or csv file", open = True):
            in_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])
        
        in_excel_sheets = gr.Dropdown(choices=[""], multiselect = False, label="Select the Excel sheet of interest.", visible=False, allow_custom_value=True)
        in_colnames = gr.Dropdown(choices=[""], multiselect = False, label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)

        with gr.Accordion("Group analysis by unique values in a specific column", open=False):
            in_group_col = gr.Dropdown(multiselect = False, label="Select the open text column to group by", allow_custom_value=True, interactive=True)
        
        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(height=FILE_INPUT_HEIGHT, label="Input topics from file (csv). File should have at least one column with a header, and all topic names below this. Using the headers 'General topic' and/or 'Subtopic' will allow for these columns to be suggested to the model. If a third column is present, it will be assumed to be a topic description.")
            with gr.Row(equal_height=True):
                force_zero_shot_radio = gr.Radio(label="Force responses into zero shot topics", value="No", choices=["Yes", "No"])
                force_single_topic_radio = gr.Radio(label="Ask the model to assign responses to only a single topic", value="No", choices=["Yes", "No"])
                produce_structures_summary_radio = gr.Radio(label="Ask the model to produce structured summaries using the zero shot topics as headers rather than extract topics", value="No", choices=["Yes", "No"])

        context_textbox = gr.Textbox(label="Write up to one sentence giving context to the large language model for your task (e.g. 'Consultation for the construction of flats on Main Street')")

        sentiment_checkbox = gr.Radio(label="Choose sentiment categories to split responses", value="Negative or Positive", choices=["Negative or Positive", "Negative, Neutral, or Positive", "Do not assess sentiment"])

        if SHOW_EXAMPLES == "True":
            examples = gr.Examples(examples=[[["example_data/dummy_consultation_response.csv"], "Response text", "Consultation for the construction of flats on Main Street", "dummy_consultation_response.csv"], [["example_data/combined_case_notes.csv",], "Case Note",  "Social Care case notes for young people",  "combined_case_notes.csv"]], inputs=[in_data_files, in_colnames, context_textbox, original_data_file_name_textbox], example_labels=["Consultation for the construction of flats on Main Street", "Social Care case notes for young people"], label="Test with an example dataset")

        if GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True":
            with gr.Accordion("Assign task to cost code", open = True, visible=True):
                gr.Markdown("Please ensure that you have approval from your budget holder before using this app for redaction tasks that incur a cost.")
                with gr.Row():
                    cost_code_dataframe = gr.Dataframe(value=pd.DataFrame(), row_count = (0, "dynamic"), label="Existing cost codes", type="pandas", interactive=True, show_fullscreen_button=True, show_copy_button=True, show_search='filter', visible=True, wrap=True, max_height=200)
                    with gr.Column():
                        reset_cost_code_dataframe_button = gr.Button(value="Reset code code table filter")
                        cost_code_choice_drop = gr.Dropdown(value=DEFAULT_COST_CODE, label="Choose cost code for analysis", choices=[DEFAULT_COST_CODE], allow_custom_value=False, visible=True)

        all_in_one_btn = gr.Button("All in one - Extract topics, deduplicate, and summarise", variant="primary")
        extract_topics_btn = gr.Button("1. Extract topics", variant="secondary")
        
        with gr.Row(equal_height=True):
            output_messages_textbox = gr.Textbox(value="", label="Output messages", scale=1, interactive=False)            
            topic_extraction_output_files = gr.File(label="Extract topics output files", scale=1, interactive=False)
            topic_extraction_output_files_xlsx = gr.File(label="Overall summary xlsx file", scale=1, interactive=False)            

        display_topic_table_markdown = gr.Markdown(value="### Language model response will appear here", show_copy_button=True)        

        data_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        data_feedback_radio = gr.Radio(label="Please give some feedback about the results of the topic extraction.",
                choices=["The results were good", "The results were not good"], visible=False)
        data_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        data_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

        with gr.Row():
            s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)

    with gr.Tab(label="2. Modify, deduplicate, and summarise topic outputs"):
        gr.Markdown("""Load in previously completed Extract Topics output files ('reference_table', and 'unique_topics' files) to modify topics, deduplicate topics, or summarise the outputs. If you want pivot table outputs, please load in the original data file along with the selected open text column on the first tab before deduplicating or summarising.""")

        with gr.Accordion("Modify existing topics", open = False):
            modification_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to modify topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])

            modifiable_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=(4, "fixed"), row_count = (1, "fixed"), visible=True, type="pandas")

            save_modified_files_button = gr.Button(value="Save modified topic names")

        with gr.Accordion("Deduplicate topics - upload reference data file and unique data files", open = True):            
            ### DEDUPLICATION
            deduplication_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to deduplicate topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])
            deduplication_input_files_status = gr.Textbox(value = "", label="Previous file input", visible=False)

            with gr.Row():
                merge_general_topics_drop = gr.Dropdown(label="Merge general topic values together for duplicate subtopics.", value="Yes", choices=["Yes", "No"])
                merge_sentiment_drop = gr.Dropdown(label="Merge sentiment values together for duplicate subtopics.", value="No", choices=["Yes", "No"])                
                deduplicate_score_threshold = gr.Number(label="Similarity threshold with which to determine duplicates.", value = 90, minimum=5, maximum=100, precision=0)

            deduplicate_previous_data_btn = gr.Button("2.a. Deduplicate topics", variant="primary")

            ### SUMMARISATION
            summarisation_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to summarise", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])

            summarise_format_radio = gr.Radio(label="Choose summary type", value=two_para_summary_format_prompt, choices=[two_para_summary_format_prompt, single_para_summary_format_prompt])
            
            summarise_previous_data_btn = gr.Button("2.b. Summarise topics", variant="primary")
            with gr.Row():
                summary_output_files = gr.File(height=FILE_INPUT_HEIGHT, label="Summarised output files", interactive=False, scale=3)
                summary_output_files_xlsx = gr.File(height=FILE_INPUT_HEIGHT, label="xlsx file summary", interactive=False, scale=1)

            summarised_output_markdown = gr.Markdown(value="### Summarised table will appear here", show_copy_button=True)

    with gr.Tab(label="3. Create overall summary"):
        gr.Markdown("""### Create an overall summary from an existing topic summary table.""")

        ### SUMMARISATION
        overall_summarisation_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload a '...unique_topic' file to summarise", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])

        overall_summarise_format_radio = gr.Radio(label="Choose summary type", value=two_para_summary_format_prompt, choices=[two_para_summary_format_prompt, single_para_summary_format_prompt], visible=False) # This is currently an invisible placeholder in case in future I want to add in overall summarisation customisation
        
        overall_summarise_previous_data_btn = gr.Button("3. Summarise table", variant="primary")

        with gr.Row():
            overall_summary_output_files = gr.File(height=FILE_INPUT_HEIGHT, label="Summarised output files", interactive=False, scale=3)
            overall_summary_output_files_xlsx = gr.File(height=FILE_INPUT_HEIGHT, label="xlsx file summary", interactive=False, scale=1)
        
        overall_summarised_output_markdown = gr.HTML(value="### Overall summary will appear here")    
    
    with gr.Tab(label="Topic table viewer", visible=False):
        gr.Markdown("""### View a 'unique_topic_table' csv file in markdown format.""")
    
        in_view_table = gr.File(height=FILE_INPUT_HEIGHT, label="Choose unique topic csv files", file_count= "single", file_types=['.csv', '.parquet'])
        view_table_markdown = gr.Markdown(value = "", label="View table", show_copy_button=True)

    with gr.Tab(label="Continue unfinished topic extraction", visible=False):
        gr.Markdown("""### Load in output files from a previous topic extraction process and continue topic extraction with new data.""")

        with gr.Accordion("Upload reference data file and unique data files", open = True):
            in_previous_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose output csv files", file_count= "multiple", file_types=['.csv'])
            in_previous_data_files_status = gr.Textbox(value = "", label="Previous file input")
            continue_previous_data_files_btn = gr.Button(value="Continue previous topic extraction", variant="primary")

    with gr.Tab(label="Verify descriptions", visible=False):
        gr.Markdown("""### Choose a tabular data file (xlsx or csv) with titles and original text to verify descriptions for.""")
        with gr.Row():
            verify_model_choice = gr.Dropdown(value = default_model_choice, choices = model_full_names, label="LLM model", multiselect=False)
            verify_in_api_key = gr.Textbox(value = "", label="Enter Gemini API key (only if using Google API models)", lines=1, type="password")

        with gr.Accordion("Upload xlsx or csv file", open = True):
            verify_in_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet'])
        
        verify_in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheet"], multiselect = False, label="Select the Excel sheet.", visible=False, allow_custom_value=True)
        verify_in_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = True, label="Select the open text columns that have a response and a title/description. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
        #verify_title_colnames = gr.Dropdown(choices=["Choose column with titles"], multiselect = False, label="Select the open text columns that have a title. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
        
        verify_titles_btn = gr.Button("Verify descriptions", variant="primary")
        verify_titles_file_output = gr.File(height=FILE_INPUT_HEIGHT, label="Descriptions verification output files")
        verify_display_topic_table_markdown = gr.Markdown(value="### Language model response will appear here", show_copy_button=True)  

        verify_modification_input_files_placeholder = gr.File(height=FILE_INPUT_HEIGHT, label="Placeholder for files to avoid errors", visible=False)

    with gr.Tab(label="LLM and topic extraction settings"):
        gr.Markdown("""Define settings that affect large language model output.""")
        with gr.Accordion("Settings for LLM generation", open = True):
            temperature_slide = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Choose LLM temperature setting", precision=1)
            batch_size_number = gr.Number(label = "Number of responses to submit in a single LLM query", value = BATCH_SIZE_DEFAULT, precision=0, minimum=1, maximum=100)
            random_seed = gr.Number(value=LLM_SEED, label="Random seed for LLM generation", visible=False)

        with gr.Accordion("AWS API keys", open = False):
            gr.Markdown("""Querying Bedrock models with API keys requires a role with IAM permissions for the bedrock:InvokeModel action.""")
            with gr.Row():
                aws_access_key_textbox = gr.Textbox(value=AWS_ACCESS_KEY, label="AWS access key", lines=1, type="password")
                aws_secret_key_textbox = gr.Textbox(value=AWS_SECRET_KEY, label="AWS secret key", lines=1, type="password")

        with gr.Accordion("Gemini API keys", open = False):
            google_api_key_textbox = gr.Textbox(value = GEMINI_API_KEY, label="Enter Gemini API key (only if using Google API models)", lines=1, type="password")

        with gr.Accordion("Hugging Face API keys", open = False):
            hf_api_key_textbox = gr.Textbox(value = HF_TOKEN, label="Enter Hugging Face API key (only if using Hugging Face models)", lines=1, type="password")

        with gr.Accordion("Log outputs", open = False):
            log_files_output = gr.File(height=FILE_INPUT_HEIGHT, label="Log file output", interactive=False)
            conversation_metadata_textbox = gr.Textbox(value="", label="Query metadata - usage counts and other parameters", lines=8)

        with gr.Accordion("Prompt settings", open = False, visible=False):
            number_of_prompts = gr.Number(value=1, label="Number of prompts to send to LLM in sequence", minimum=1, maximum=3, visible=False)
            system_prompt_textbox = gr.Textbox(label="Initial system prompt", lines = 4, value = system_prompt)
            initial_table_prompt_textbox = gr.Textbox(label = "Initial topics prompt", lines = 8, value = initial_table_prompt)
            prompt_2_textbox = gr.Textbox(label = "Prompt 2", lines = 8, value = prompt2, visible=False)
            prompt_3_textbox = gr.Textbox(label = "Prompt 3", lines = 8, value = prompt3, visible=False)
            add_to_existing_topics_system_prompt_textbox = gr.Textbox(label="Additional topics system prompt", lines = 4, value = add_existing_topics_system_prompt)
            add_to_existing_topics_prompt_textbox = gr.Textbox(label = "Additional topics prompt", lines = 8, value = add_existing_topics_prompt)
            verify_titles_system_prompt_textbox = gr.Textbox(label="Verify descriptions system prompt", lines = 4, value = verify_titles_system_prompt, visible=False)
            verify_titles_prompt_textbox = gr.Textbox(label = "Verify descriptions prompt", lines = 8, value = verify_titles_prompt, visible=False)

        with gr.Accordion("Join additional columns to reference file outputs", open = False):
            join_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = True, label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
            with gr.Row():
                in_join_files = gr.File(height=FILE_INPUT_HEIGHT, label="Reference file should go here. Original data file should be loaded on the first tab.")
                join_cols_btn = gr.Button("Join columns to reference output", variant="primary")
            out_join_files = gr.File(height=FILE_INPUT_HEIGHT, label="Output joined reference files will go here.")

        with gr.Accordion("Export output files to xlsx format", open = False, visible=False):
            export_xlsx_btn = gr.Button("Export output files to xlsx format", variant="primary")
            out_xlsx_files = gr.File(height=FILE_INPUT_HEIGHT, label="Output xlsx files will go here.")        

        # Invisible text box to hold the session hash/username just for logging purposes
        session_hash_textbox = gr.Textbox(label = "Session hash", value="", visible=False) 
        
        estimated_time_taken_number = gr.Number(label= "Estimated time taken (seconds)", value=0.0, precision=1, visible=False) # This keeps track of the time taken to redact files for logging purposes.
        total_number_of_batches = gr.Number(label = "Current batch number", value = 1, precision=0, visible=False)
        
        text_output_logs = gr.Textbox(label = "Output summary logs", visible=False)
   
    ###
    # INTERACTIVE ELEMENT FUNCTIONS
    ###

    ###
    # INITIAL TOPIC EXTRACTION
    ###

    # Tabular data upload
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets, original_data_file_name_textbox, join_colnames, in_group_col])

    # Click on cost code dataframe/dropdown fills in cost code textbox
    # Allow user to select items from cost code dataframe for cost code
    if SHOW_COSTS=="True" and (GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True"):
        cost_code_dataframe.select(df_select_callback_cost, inputs=[cost_code_dataframe], outputs=[cost_code_choice_drop])
        reset_cost_code_dataframe_button.click(reset_base_dataframe, inputs=[cost_code_dataframe_base], outputs=[cost_code_dataframe])

        cost_code_choice_drop.select(update_cost_code_dataframe_from_dropdown_select, inputs=[cost_code_choice_drop, cost_code_dataframe_base], outputs=[cost_code_dataframe])

    
    # Extract topics
    extract_topics_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, working_data_file_name_textbox, display_topic_table_markdown, summary_output_files, summarisation_input_files, overall_summarisation_input_files, overall_summary_output_files]).\
    success(fn= enforce_cost_codes, inputs=[enforce_cost_code_textbox, cost_code_choice_drop, cost_code_dataframe_base]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, working_data_file_name_textbox, total_number_of_batches], api_name="load_data").\
    success(fn=wrapper_extract_topics_per_column_value,                           
        inputs=[in_group_col,
                in_data_files,
                file_data_state,              
                master_topic_df_state,
                master_reference_df_state,
                master_unique_topics_df_state,
                display_topic_table_markdown,
                original_data_file_name_textbox,
                total_number_of_batches,
                google_api_key_textbox,
                temperature_slide,
                in_colnames,
                model_choice,
                candidate_topics,
                first_loop_state,
                conversation_metadata_textbox,
                latest_batch_completed,
                estimated_time_taken_number,                
                initial_table_prompt_textbox,
                prompt_2_textbox,
                prompt_3_textbox,
                system_prompt_textbox,
                add_to_existing_topics_system_prompt_textbox,
                add_to_existing_topics_prompt_textbox,
                number_of_prompts,
                batch_size_number,
                context_textbox,
                sentiment_checkbox,
                force_zero_shot_radio,
                in_excel_sheets,
                force_single_topic_radio,
                produce_structures_summary_radio,
                aws_access_key_textbox,
                aws_secret_key_textbox,
                hf_api_key_textbox,
                output_folder_state],
        outputs=[display_topic_table_markdown,
                master_topic_df_state,
                master_unique_topics_df_state,
                master_reference_df_state,
                topic_extraction_output_files,
                text_output_file_list_state,
                latest_batch_completed,
                log_files_output,
                log_files_output_list_state,
                conversation_metadata_textbox,
                estimated_time_taken_number,
                deduplication_input_files,
                summarisation_input_files,
                modifiable_unique_topics_df_state,
                modification_input_files,
                in_join_files,
                missing_df_state,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                output_messages_textbox],
                api_name="extract_topics", show_progress_on=output_messages_textbox).\
                success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False, api_name="usage_logs").\
                success(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_state, master_unique_topics_df_state, summarised_output_df, missing_df_state, in_excel_sheets, usage_logs_state, model_name_map_state, output_folder_state], outputs=[topic_extraction_output_files_xlsx, summary_xlsx_output_files_list])

    ###
    # DEDUPLICATION AND SUMMARISATION FUNCTIONS
    ###
    # If you upload data into the deduplication input box, the modifiable topic dataframe box is updated
    modification_input_files.upload(fn=load_in_previous_data_files, inputs=[modification_input_files, modified_unique_table_change_bool], outputs=[modifiable_unique_topics_df_state, master_modify_reference_df_state, master_modify_unique_topics_df_state, working_data_file_name_textbox, unique_topics_table_file_name_textbox, text_output_modify_file_list_state])

    # Modify output table with custom topic names
    save_modified_files_button.click(fn=modify_existing_output_tables, inputs=[master_modify_unique_topics_df_state, modifiable_unique_topics_df_state, master_modify_reference_df_state, text_output_modify_file_list_state, output_folder_state], outputs=[master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, deduplication_input_files, summarisation_input_files, working_data_file_name_textbox, unique_topics_table_file_name_textbox, summarised_output_markdown])
    
    # When button pressed, deduplicate data
    deduplicate_previous_data_btn.click(load_in_previous_data_files, inputs=[deduplication_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
        success(deduplicate_topics, inputs=[master_reference_df_state, master_unique_topics_df_state, working_data_file_name_textbox, unique_topics_table_file_name_textbox, in_excel_sheets, merge_sentiment_drop, merge_general_topics_drop, deduplicate_score_threshold, in_data_files, in_colnames, output_folder_state], outputs=[master_reference_df_state, master_unique_topics_df_state, summarisation_input_files, log_files_output, summarised_output_markdown], scroll_to_output=True, api_name="deduplicate_topics")
    
    # When button pressed, summarise previous data
    summarise_previous_data_btn.click(empty_output_vars_summarise, inputs=None, outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, overall_summarisation_input_files]).\
    success(fn= enforce_cost_codes, inputs=[enforce_cost_code_textbox, cost_code_choice_drop, cost_code_dataframe_base]).\
        success(load_in_previous_data_files, inputs=[summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
            success(sample_reference_table_summaries, inputs=[master_reference_df_state, random_seed], outputs=[summary_reference_table_sample_state, summarised_references_markdown], api_name="sample_summaries").\
                success(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, google_api_key_textbox, temperature_slide, working_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio, output_folder_state, context_textbox, aws_access_key_textbox, aws_secret_key_textbox, model_name_map_state], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown, log_files_output, overall_summarisation_input_files, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, output_messages_textbox], api_name="summarise_topics", show_progress_on=[output_messages_textbox, summary_output_files]).\
                success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False).\
                success(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_revised_summaries_state, master_unique_topics_df_revised_summaries_state, summarised_output_df, missing_df_state, in_excel_sheets, usage_logs_state, model_name_map_state, output_folder_state], outputs=[summary_output_files_xlsx, summary_xlsx_output_files_list])

    # SUMMARISE WHOLE TABLE PAGE
    overall_summarise_previous_data_btn.click(fn= enforce_cost_codes, inputs=[enforce_cost_code_textbox, cost_code_choice_drop, cost_code_dataframe_base]).\
            success(load_in_previous_data_files, inputs=[overall_summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
            success(overall_summary, inputs=[master_unique_topics_df_state, model_choice, google_api_key_textbox, temperature_slide, working_data_file_name_textbox, output_folder_state, in_colnames, context_textbox, aws_access_key_textbox, aws_secret_key_textbox, model_name_map_state], outputs=[overall_summary_output_files, overall_summarised_output_markdown, summarised_output_df, conversation_metadata_textbox, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, output_messages_textbox], scroll_to_output=True, api_name="overall_summary", show_progress_on=[output_messages_textbox, overall_summary_output_files]).\
            success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False).\
            success(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_state, master_unique_topics_df_state, summarised_output_df, missing_df_state, in_excel_sheets, usage_logs_state, model_name_map_state, output_folder_state], outputs=[overall_summary_output_files_xlsx, summary_xlsx_output_files_list])

    
    # All in one button
    # Extract topics - deduplicate and summarise using default settings
    all_in_one_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, working_data_file_name_textbox, display_topic_table_markdown, summary_output_files, summarisation_input_files, overall_summarisation_input_files, overall_summary_output_files]).\
    success(fn= enforce_cost_codes, inputs=[enforce_cost_code_textbox, cost_code_choice_drop, cost_code_dataframe_base]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, working_data_file_name_textbox, total_number_of_batches], api_name="load_data").\
    success(fn=wrapper_extract_topics_per_column_value,                           
        inputs=[in_group_col,
                in_data_files,
                file_data_state,              
                master_topic_df_state,
                master_reference_df_state,
                master_unique_topics_df_state,
                display_topic_table_markdown,
                original_data_file_name_textbox,
                total_number_of_batches,
                google_api_key_textbox,
                temperature_slide,
                in_colnames,
                model_choice,
                candidate_topics,
                first_loop_state,
                conversation_metadata_textbox,
                latest_batch_completed,
                estimated_time_taken_number,                
                initial_table_prompt_textbox,
                prompt_2_textbox,
                prompt_3_textbox,
                system_prompt_textbox,
                add_to_existing_topics_system_prompt_textbox,
                add_to_existing_topics_prompt_textbox,
                number_of_prompts,
                batch_size_number,
                context_textbox,
                sentiment_checkbox,
                force_zero_shot_radio,
                in_excel_sheets,
                force_single_topic_radio,
                produce_structures_summary_radio,
                aws_access_key_textbox,
                aws_secret_key_textbox,
                hf_api_key_textbox,
                output_folder_state],
        outputs=[display_topic_table_markdown,
                master_topic_df_state,
                master_unique_topics_df_state,
                master_reference_df_state,
                topic_extraction_output_files,
                text_output_file_list_state,
                latest_batch_completed,
                log_files_output,
                log_files_output_list_state,
                conversation_metadata_textbox,
                estimated_time_taken_number,
                deduplication_input_files,
                summarisation_input_files,
                modifiable_unique_topics_df_state,
                modification_input_files,
                in_join_files,
                missing_df_state,
                input_tokens_num,
                output_tokens_num,
                number_of_calls_num,
                output_messages_textbox], show_progress_on=output_messages_textbox).\
                success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False).\
                success(load_in_previous_data_files, inputs=[deduplication_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
                success(deduplicate_topics, inputs=[master_reference_df_state, master_unique_topics_df_state, working_data_file_name_textbox, unique_topics_table_file_name_textbox, in_excel_sheets, merge_sentiment_drop, merge_general_topics_drop, deduplicate_score_threshold, in_data_files, in_colnames, output_folder_state], outputs=[master_reference_df_state, master_unique_topics_df_state, summarisation_input_files, log_files_output, summarised_output_markdown]).\
                success(load_in_previous_data_files, inputs=[summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
                success(sample_reference_table_summaries, inputs=[master_reference_df_state, random_seed], outputs=[summary_reference_table_sample_state, summarised_references_markdown]).\
                success(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, google_api_key_textbox, temperature_slide, working_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio, output_folder_state, context_textbox, aws_access_key_textbox, aws_secret_key_textbox, model_name_map_state], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, display_topic_table_markdown, log_files_output, overall_summarisation_input_files, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, output_messages_textbox], show_progress_on=[output_messages_textbox, summary_output_files]).\
                success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False).\
                success(load_in_previous_data_files, inputs=[overall_summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
                success(overall_summary, inputs=[master_unique_topics_df_state, model_choice, google_api_key_textbox, temperature_slide, working_data_file_name_textbox, output_folder_state, in_colnames, context_textbox, aws_access_key_textbox, aws_secret_key_textbox, model_name_map_state], outputs=[overall_summary_output_files, overall_summarised_output_markdown, summarised_output_df, conversation_metadata_textbox, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, output_messages_textbox], show_progress_on=[output_messages_textbox, overall_summary_output_files]).\
                success(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False).\
                success(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_state, master_unique_topics_df_state, summarised_output_df, missing_df_state, in_excel_sheets, usage_logs_state, model_name_map_state, output_folder_state], outputs=[overall_summary_output_files_xlsx, summary_xlsx_output_files_list]).\
                success(move_overall_summary_output_files_to_front_page, inputs=[summary_xlsx_output_files_list], outputs=[topic_extraction_output_files_xlsx])

    ###
    # CONTINUE PREVIOUS TOPIC EXTRACTION PAGE
    ###

    # If uploaded partially completed consultation files do this. This should then start up the 'latest_batch_completed' change action above to continue extracting topics.
    continue_previous_data_files_btn.click(
        load_in_data_file, inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, working_data_file_name_textbox, total_number_of_batches]).\
        success(load_in_previous_data_files, inputs=[in_previous_data_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed, in_previous_data_files_status, working_data_file_name_textbox, unique_topics_table_file_name_textbox])
    
    ###
    # VERIFY TEXT TITLES/DESCRIPTIONS
    ###

    # Tabular data upload
    verify_in_data_files.upload(fn=put_columns_in_df, inputs=[verify_in_data_files], outputs=[verify_in_colnames, verify_in_excel_sheets, original_data_file_name_textbox, join_colnames])

    verify_titles_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, working_data_file_name_textbox, display_topic_table_markdown]).\
    success(load_in_data_file,
        inputs = [verify_in_data_files, verify_in_colnames, batch_size_number, verify_in_excel_sheets], outputs = [file_data_state, working_data_file_name_textbox, total_number_of_batches], api_name="verify_load_data").\
    success(fn=verify_titles,
        inputs=[verify_in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, display_topic_table_markdown, original_data_file_name_textbox, total_number_of_batches, verify_in_api_key, temperature_slide, verify_in_colnames, verify_model_choice, candidate_topics, latest_batch_completed, display_topic_table_markdown, text_output_file_list_state, log_files_output_list_state, first_loop_state, conversation_metadata_textbox, verify_titles_prompt_textbox, prompt_2_textbox, prompt_3_textbox, verify_titles_system_prompt_textbox, verify_titles_system_prompt_textbox, verify_titles_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number, sentiment_checkbox, force_zero_shot_radio, produce_structures_summary_radio, aws_access_key_textbox, aws_secret_key_textbox, in_excel_sheets, output_folder_state],
        outputs=[verify_display_topic_table_markdown, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, verify_titles_file_output, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, deduplication_input_files, summarisation_input_files, modifiable_unique_topics_df_state, verify_modification_input_files_placeholder], api_name="verify_descriptions")
    
    ###
    # VIEW TABLE PAGE
    ###

    in_view_table.upload(view_table, inputs=[in_view_table], outputs=[view_table_markdown])
    
    ###
    # LLM SETTINGS PAGE
    ###

    reference_df_data_file_name_textbox = gr.Textbox(label="reference_df_data_file_name_textbox", visible=False)
    master_reference_df_state_joined = gr.Dataframe(visible=False)

    join_cols_btn.click(fn=load_in_previous_reference_file, inputs=[in_join_files], outputs=[master_reference_df_state, reference_df_data_file_name_textbox]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, working_data_file_name_textbox, total_number_of_batches]).\
    success(fn=join_cols_onto_reference_df, inputs=[master_reference_df_state, file_data_state, join_colnames, reference_df_data_file_name_textbox], outputs=[master_reference_df_state_joined, out_join_files])

    # Export to xlsx file
    export_xlsx_btn.click(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_state, master_unique_topics_df_state, summarised_output_df, missing_df_state, in_excel_sheets, usage_logs_state, model_name_map_state, output_folder_state], outputs=[out_xlsx_files], api_name="export_xlsx")

    # If relevant environment variable is set, load in the default cost code file from S3 or locally
    if GET_COST_CODES == "True" and (COST_CODES_PATH or S3_COST_CODES_PATH):
        if not os.path.exists(COST_CODES_PATH) and S3_COST_CODES_PATH and RUN_AWS_FUNCTIONS == "1":
            print("Downloading cost codes from S3")
            app.load(download_file_from_s3, inputs=[s3_default_bucket, s3_default_cost_codes_file, default_cost_codes_output_folder_location]).\
            success(load_in_default_cost_codes, inputs = [default_cost_codes_output_folder_location, default_cost_code_textbox], outputs=[cost_code_dataframe, cost_code_dataframe_base, cost_code_choice_drop])
            print("Successfully loaded cost codes from S3")
        elif os.path.exists(COST_CODES_PATH):
            print("Loading cost codes from default cost codes path location:", COST_CODES_PATH)
            app.load(load_in_default_cost_codes, inputs = [default_cost_codes_output_folder_location, default_cost_code_textbox], outputs=[cost_code_dataframe, cost_code_dataframe_base, cost_code_choice_drop])
        else: print("Could not load in cost code data")

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, output_folder_state, session_hash_textbox, input_folder_state])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)
    access_callback.setup([session_hash_textbox], ACCESS_LOGS_FOLDER)
    
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS, replacement_headers=CSV_ACCESS_LOG_HEADERS), [session_hash_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    # Log usage when making a query
    usage_callback = CSVLogger_custom(dataset_file_name=USAGE_LOG_FILE_NAME)
    usage_callback.setup([session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox_placeholder, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], USAGE_LOGS_FOLDER)

    # See extract topics and summarise calls to see the calls to usage logs

    # number_of_calls_num.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False, api_name="usage_logs").\
    #     success(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    number_of_calls_num.change(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    # User submitted feedback
    feedback_callback = CSVLogger_custom(dataset_file_name=FEEDBACK_LOG_FILE_NAME)
    feedback_callback.setup([data_feedback_radio, data_further_details_text, original_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], FEEDBACK_LOGS_FOLDER)

    data_submit_feedback_btn.click(lambda *args: feedback_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS, replacement_headers=CSV_FEEDBACK_LOG_HEADERS), [data_feedback_radio, data_further_details_text, original_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[data_further_details_text])

###
# APP RUN
###

if __name__ == "__main__":
    if COGNITO_AUTH == "1":
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, inbrowser=True, auth=authenticate_user, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
    else:
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, inbrowser=True, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)