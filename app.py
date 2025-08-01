import spaces
import os
import gradio as gr
import pandas as pd
from datetime import datetime
from tools.helper_functions import put_columns_in_df, get_connection_params, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, view_table, empty_output_vars_extract_topics, empty_output_vars_summarise, load_in_previous_reference_file, join_cols_onto_reference_df, load_in_previous_data_files, load_in_data_file
from tools.aws_functions import upload_file_to_s3
from tools.llm_api_call import modify_existing_output_tables, wrapper_extract_topics_per_column_value
from tools.dedup_summaries import sample_reference_table_summaries, summarise_output_topics, deduplicate_topics, overall_summary
from tools.combine_sheets_into_xlsx import collect_output_csvs_and_create_excel_output
from tools.auth import authenticate_user
from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt, verify_titles_prompt, verify_titles_system_prompt, two_para_summary_format_prompt, single_para_summary_format_prompt
from tools.verify_titles import verify_titles
from tools.config import RUN_AWS_FUNCTIONS, HOST_NAME, ACCESS_LOGS_FOLDER, FEEDBACK_LOGS_FOLDER, USAGE_LOGS_FOLDER, RUN_LOCAL_MODEL,  FILE_INPUT_HEIGHT, GEMINI_API_KEY, model_full_names, BATCH_SIZE_DEFAULT, CHOSEN_LOCAL_MODEL_TYPE, LLM_SEED, COGNITO_AUTH, MAX_QUEUE_SIZE, MAX_FILE_SIZE, GRADIO_SERVER_PORT, ROOT_PATH, INPUT_FOLDER, OUTPUT_FOLDER, S3_LOG_BUCKET

today_rev = datetime.now().strftime("%Y%m%d")

if RUN_LOCAL_MODEL == "1":
    default_model_choice = CHOSEN_LOCAL_MODEL_TYPE
elif RUN_AWS_FUNCTIONS == "1":
    default_model_choice = "anthropic.claude-3-haiku-20240307-v1:0"
else:
    default_model_choice = "gemini-2.0-flash-001"

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
 
    session_hash_state = gr.Textbox(visible=False, value=HOST_NAME)
    output_folder_state = gr.Textbox(visible=False, value=OUTPUT_FOLDER)
    input_folder_state = gr.Textbox(visible=False, value=INPUT_FOLDER)

    # s3 bucket name
    s3_log_bucket_name = gr.Textbox(visible=False, value=S3_LOG_BUCKET)

    # Logging state
    log_file_name = 'log.csv'

    access_logs_state = gr.Textbox(ACCESS_LOGS_FOLDER + log_file_name, visible=False)
    access_s3_logs_loc_state = gr.Textbox(ACCESS_LOGS_FOLDER, visible=False)
    usage_logs_state = gr.Textbox(USAGE_LOGS_FOLDER + log_file_name, visible=False)
    usage_s3_logs_loc_state = gr.Textbox(USAGE_LOGS_FOLDER, visible=False)
    feedback_logs_state = gr.Textbox(FEEDBACK_LOGS_FOLDER + log_file_name, visible=False)
    feedback_s3_logs_loc_state = gr.Textbox(FEEDBACK_LOGS_FOLDER, visible=False)

    # Summary state objects
    summary_reference_table_sample_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="summary_reference_table_sample_state", visible=False, type="pandas")
    master_reference_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_reference_df_revised_summaries_state", visible=False, type="pandas")
    master_unique_topics_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_unique_topics_df_revised_summaries_state", visible=False, type="pandas")
    summarised_output_df = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="summarised_output_df", visible=False, type="pandas")
    summarised_references_markdown = gr.Markdown("", visible=False)
    summarised_outputs_list = gr.Dropdown(value=[], choices=[], visible=False, label="List of summarised outputs", allow_custom_value=True)
    latest_summary_completed_num = gr.Number(0, visible=False)

    original_data_file_name_textbox = gr.Textbox(label = "Reference data file name", value="", visible=False)
    unique_topics_table_file_name_textbox = gr.Textbox(label="Unique topics data file name textbox", visible=False)

    ###
    # UI LAYOUT
    ###

    gr.Markdown("""# Large language model topic modelling

    Extract topics and summarise outputs using Large Language Models (LLMs, a Gemma model if local, Gemini Flash/Pro, or Claude 3 through AWS Bedrock if running on AWS). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and relevant text rows related to them. The prompts are designed for topic modelling public consultations, but they can be adapted to different contexts (see the LLM settings tab to modify). 
    
    Instructions on use can be found in the README.md file. Try it out with this [dummy development consultation dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation), which you can also try with [zero-shot topics](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/blob/main/example_zero_shot.csv), or this [dummy case notes dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_case_notes).

    You can use an AWS Bedrock model (Claude 3, paid), or Gemini (a free API, but with strict limits for the Pro model). The use of Gemini models requires an API key. To set up your own Gemini API key, go [here](https://aistudio.google.com/app/u/1/plan_information).

    NOTE: that **API calls to Gemini are not considered secure**, so please only submit redacted, non-sensitive tabular files to this source. Also, large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.""")
    
    with gr.Tab(label="Extract topics"):
        gr.Markdown("""### Choose a tabular data file (xlsx or csv) of open text to extract topics from.""")
        with gr.Row():
            model_choice = gr.Dropdown(value = default_model_choice, choices = model_full_names, label="LLM model to use", multiselect=False)
            in_api_key = gr.Textbox(value = GEMINI_API_KEY, label="Enter Gemini API key (only if using Google API models)", lines=1, type="password")

        with gr.Accordion("Upload xlsx or csv file", open = True):
            in_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
        
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

        extract_topics_btn = gr.Button("Extract topics", variant="primary")
        
        topic_extraction_output_files = gr.File(height=FILE_INPUT_HEIGHT, label="Output files")
        display_topic_table_markdown = gr.Markdown(value="### Language model response will appear here", show_copy_button=True)        
        latest_batch_completed = gr.Number(value=0, label="Number of files prepared", interactive=False, visible=False)
        # Duplicate version of the above variable for when you don't want to initiate the summarisation loop
        latest_batch_completed_no_loop = gr.Number(value=0, label="Number of files prepared", interactive=False, visible=False)

        data_feedback_title = gr.Markdown(value="## Please give feedback", visible=False)
        data_feedback_radio = gr.Radio(label="Please give some feedback about the results of the topic extraction.",
                choices=["The results were good", "The results were not good"], visible=False)
        data_further_details_text = gr.Textbox(label="Please give more detailed feedback about the results:", visible=False)
        data_submit_feedback_btn = gr.Button(value="Submit feedback", visible=False)

        with gr.Row():
            s3_logs_output_textbox = gr.Textbox(label="Feedback submission logs", visible=False)

    with gr.Tab(label="Modify, deduplicate, and summarise topic outputs"):
        gr.Markdown("""Load in previously completed Extract Topics output files ('reference_table', and 'unique_topics' files) to modify topics, deduplicate topics, or summarise the outputs. If you want pivot table outputs, please load in the original data file along with the selected open text column on the first tab before deduplicating or summarising.""")

        with gr.Accordion("Modify existing topics", open = False):
            modification_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to modify topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])

            modifiable_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=(4, "fixed"), row_count = (1, "fixed"), visible=True, type="pandas")

            save_modified_files_button = gr.Button(value="Save modified topic names")

        with gr.Accordion("Deduplicate topics - upload reference data file and unique data files", open = True):            
            ### DEDUPLICATION
            deduplication_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to deduplicate topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            deduplication_input_files_status = gr.Textbox(value = "", label="Previous file input", visible=False)

            with gr.Row():
                merge_general_topics_drop = gr.Dropdown(label="Merge general topic values together for duplicate subtopics.", value="Yes", choices=["Yes", "No"])
                merge_sentiment_drop = gr.Dropdown(label="Merge sentiment values together for duplicate subtopics.", value="No", choices=["Yes", "No"])                
                deduplicate_score_threshold = gr.Number(label="Similarity threshold with which to determine duplicates.", value = 90, minimum=5, maximum=100, precision=0)

            deduplicate_previous_data_btn = gr.Button("Deduplicate topics", variant="primary")

            ### SUMMARISATION
            summarisation_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload files to summarise", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])

            summarise_format_radio = gr.Radio(label="Choose summary type", value=two_para_summary_format_prompt, choices=[two_para_summary_format_prompt, single_para_summary_format_prompt])
            
            summarise_previous_data_btn = gr.Button("Summarise topics", variant="primary")
            summary_output_files = gr.File(height=FILE_INPUT_HEIGHT, label="Summarised output files", interactive=False)
            summarised_output_markdown = gr.Markdown(value="### Summarised table will appear here", show_copy_button=True)

    with gr.Tab(label="Create overall summary"):
        gr.Markdown("""### Create an overall summary from an existing topic summary table.""")

        ### SUMMARISATION
        overall_summarisation_input_files = gr.File(height=FILE_INPUT_HEIGHT, label="Upload a '...unique_topic' file to summarise", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])

        overall_summarise_format_radio = gr.Radio(label="Choose summary type", value=two_para_summary_format_prompt, choices=[two_para_summary_format_prompt, single_para_summary_format_prompt], visible=False) # This is currently an invisible placeholder in case in future I want to add in overall summarisation customisation
        
        overall_summarise_previous_data_btn = gr.Button("Summarise table", variant="primary")
        overall_summary_output_files = gr.File(height=FILE_INPUT_HEIGHT, label="Summarised output files", interactive=False)
        overall_summarised_output_markdown = gr.HTML(value="### Overall summary will appear here")    
    
    with gr.Tab(label="Topic table viewer"):
        gr.Markdown("""### View a 'unique_topic_table' csv file in markdown format.""")
    
        in_view_table = gr.File(height=FILE_INPUT_HEIGHT, label="Choose unique topic csv files", file_count= "single", file_types=['.csv', '.parquet', '.csv.gz'])
        view_table_markdown = gr.Markdown(value = "", label="View table", show_copy_button=True)

    with gr.Tab(label="Continue unfinished topic extraction"):
        gr.Markdown("""### Load in output files from a previous topic extraction process and continue topic extraction with new data.""")

        with gr.Accordion("Upload reference data file and unique data files", open = True):
            in_previous_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose output csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            in_previous_data_files_status = gr.Textbox(value = "", label="Previous file input")
            continue_previous_data_files_btn = gr.Button(value="Continue previous topic extraction", variant="primary")

    with gr.Tab(label="Verify descriptions"):
        gr.Markdown("""### Choose a tabular data file (xlsx or csv) with titles and original text to verify descriptions for.""")
        with gr.Row():
            verify_model_choice = gr.Dropdown(value = default_model_choice, choices = model_full_names, label="LLM model to use", multiselect=False)
            verify_in_api_key = gr.Textbox(value = "", label="Enter Gemini API key (only if using Google API models)", lines=1, type="password")

        with gr.Accordion("Upload xlsx or csv file", open = True):
            verify_in_data_files = gr.File(height=FILE_INPUT_HEIGHT, label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
        
        verify_in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheet"], multiselect = False, label="Select the Excel sheet.", visible=False, allow_custom_value=True)
        verify_in_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = True, label="Select the open text columns that have a response and a title/description. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
        #verify_title_colnames = gr.Dropdown(choices=["Choose column with titles"], multiselect = False, label="Select the open text columns that have a title. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
        
        verify_titles_btn = gr.Button("Verify descriptions", variant="primary")
        verify_titles_file_output = gr.File(height=FILE_INPUT_HEIGHT, label="Descriptions verification output files")
        verify_display_topic_table_markdown = gr.Markdown(value="### Language model response will appear here", show_copy_button=True)  

        verify_modification_input_files_placeholder = gr.File(height=FILE_INPUT_HEIGHT, label="Placeholder for files to avoid errors", visible=False)

    with gr.Tab(label="Topic extraction settings"):
        gr.Markdown("""Define settings that affect large language model output.""")
        with gr.Accordion("Settings for LLM generation", open = True):
            temperature_slide = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Choose LLM temperature setting")
            batch_size_number = gr.Number(label = "Number of responses to submit in a single LLM query", value = BATCH_SIZE_DEFAULT, precision=0, minimum=1, maximum=100)
            random_seed = gr.Number(value=LLM_SEED, label="Random seed for LLM generation", visible=False)            

        with gr.Accordion("Prompt settings", open = False):
            number_of_prompts = gr.Number(value=1, label="Number of prompts to send to LLM in sequence", minimum=1, maximum=3, visible=False)
            system_prompt_textbox = gr.Textbox(label="Initial system prompt", lines = 4, value = system_prompt)
            initial_table_prompt_textbox = gr.Textbox(label = "Initial topics prompt", lines = 8, value = initial_table_prompt)
            prompt_2_textbox = gr.Textbox(label = "Prompt 2", lines = 8, value = prompt2, visible=False)
            prompt_3_textbox = gr.Textbox(label = "Prompt 3", lines = 8, value = prompt3, visible=False)
            add_to_existing_topics_system_prompt_textbox = gr.Textbox(label="Additional topics system prompt", lines = 4, value = add_existing_topics_system_prompt)
            add_to_existing_topics_prompt_textbox = gr.Textbox(label = "Additional topics prompt", lines = 8, value = add_existing_topics_prompt)
            verify_titles_system_prompt_textbox = gr.Textbox(label="Additional topics system prompt", lines = 4, value = verify_titles_system_prompt)
            verify_titles_prompt_textbox = gr.Textbox(label = "Additional topics prompt", lines = 8, value = verify_titles_prompt)

        with gr.Accordion("Join additional columns to reference file outputs", open = False):
            join_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = True, label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
            with gr.Row():
                in_join_files = gr.File(height=FILE_INPUT_HEIGHT, label="Reference file should go here. Original data file should be loaded on the first tab.")
                join_cols_btn = gr.Button("Join columns to reference output", variant="primary")
            out_join_files = gr.File(height=FILE_INPUT_HEIGHT, label="Output joined reference files will go here.")

        with gr.Accordion("Export output files to xlsx format", open = False):
            export_xlsx_btn = gr.Button("Export output files to xlsx format", variant="primary")
            out_xlsx_files = gr.File(height=FILE_INPUT_HEIGHT, label="Output xlsx files will go here.")

        with gr.Accordion("Logging outputs", open = False):
            log_files_output = gr.File(height=FILE_INPUT_HEIGHT, label="Log file output", interactive=False)
            conversation_metadata_textbox = gr.Textbox(label="Query metadata - usage counts and other parameters", interactive=False, lines=8)

        with gr.Accordion("Enter AWS API keys", open = False):
            aws_access_key_textbox = gr.Textbox(label="AWS access key", interactive=False, lines=1, type="password")
            aws_secret_key_textbox = gr.Textbox(label="AWS secret key", interactive=False, lines=1, type="password")

        # Invisible text box to hold the session hash/username just for logging purposes
        session_hash_textbox = gr.Textbox(label = "Session hash", value="", visible=False) 
        
        estimated_time_taken_number = gr.Number(label= "Estimated time taken (seconds)", value=0.0, precision=1, visible=False) # This keeps track of the time taken to redact files for logging purposes.
        total_number_of_batches = gr.Number(label = "Current batch number", value = 1, precision=0, visible=False)
        
        text_output_logs = gr.Textbox(label = "Output summary logs", visible=False)
            
    # AWS options - not yet implemented
    # with gr.Tab(label="Advanced options"):
    #     with gr.Accordion(label = "AWS data access", open = True):
    #         aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
    #         with gr.Row():
    #             in_aws_file = gr.Dropdown(label="Choose file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth borough plan"])
    #             load_aws_data_button = gr.Button(value="Load data from AWS", variant="secondary")
                
    #         aws_log_box = gr.Textbox(label="AWS data load status")
    
    # ### Loading AWS data ###
    # load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_file, aws_log_box])
   
    ###
    # INTERACTIVE ELEMENT FUNCTIONS
    ###

    ###
    # INITIAL TOPIC EXTRACTION
    ###

    # Tabular data upload
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets, original_data_file_name_textbox, join_colnames, in_group_col])

    extract_topics_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, original_data_file_name_textbox, display_topic_table_markdown, summary_output_files, summarisation_input_files, overall_summarisation_input_files, overall_summary_output_files]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, original_data_file_name_textbox, total_number_of_batches], api_name="load_data").\
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
                in_api_key,
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
                 missing_df_state],
                 api_name="extract_topics")
    
    # extract_topics_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, original_data_file_name_textbox, display_topic_table_markdown]).\
    # success(load_in_data_file,                           
    #     inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, original_data_file_name_textbox, total_number_of_batches], api_name="load_data").\
    # success(fn=extract_topics,                           
    #     inputs=[in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, display_topic_table_markdown, original_data_file_name_textbox, total_number_of_batches, in_api_key, temperature_slide, in_colnames, model_choice, candidate_topics, latest_batch_completed, display_topic_table_markdown, text_output_file_list_state, log_files_output_list_state, first_loop_state, conversation_metadata_textbox, initial_table_prompt_textbox, prompt_2_textbox, prompt_3_textbox, system_prompt_textbox, add_to_existing_topics_system_prompt_textbox, add_to_existing_topics_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number, sentiment_checkbox, force_zero_shot_radio, in_excel_sheets, force_single_topic_radio, output_folder_state],        
    #     outputs=[display_topic_table_markdown, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, deduplication_input_files, summarisation_input_files, modifiable_unique_topics_df_state, modification_input_files, in_join_files], api_name="extract_topics")

    ###
    # DEDUPLICATION AND SUMMARISATION FUNCTIONS
    ###
    # If you upload data into the deduplication input box, the modifiable topic dataframe box is updated
    modification_input_files.change(fn=load_in_previous_data_files, inputs=[modification_input_files, modified_unique_table_change_bool], outputs=[modifiable_unique_topics_df_state, master_modify_reference_df_state, master_modify_unique_topics_df_state, original_data_file_name_textbox, unique_topics_table_file_name_textbox, text_output_modify_file_list_state])

    # Modify output table with custom topic names
    save_modified_files_button.click(fn=modify_existing_output_tables, inputs=[master_modify_unique_topics_df_state, modifiable_unique_topics_df_state, master_modify_reference_df_state, text_output_modify_file_list_state, output_folder_state], outputs=[master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, deduplication_input_files, summarisation_input_files, original_data_file_name_textbox, unique_topics_table_file_name_textbox, summarised_output_markdown])
    
    # When button pressed, deduplicate data
    deduplicate_previous_data_btn.click(load_in_previous_data_files, inputs=[deduplication_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, original_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
        success(deduplicate_topics, inputs=[master_reference_df_state, master_unique_topics_df_state, original_data_file_name_textbox, unique_topics_table_file_name_textbox, in_excel_sheets, merge_sentiment_drop, merge_general_topics_drop, deduplicate_score_threshold, in_data_files, in_colnames, output_folder_state], outputs=[master_reference_df_state, master_unique_topics_df_state, summarisation_input_files, log_files_output, summarised_output_markdown], scroll_to_output=True, api_name="deduplicate_topics")
    
    # When button pressed, summarise previous data
    summarise_previous_data_btn.click(empty_output_vars_summarise, inputs=None, outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, overall_summarisation_input_files]).\
        success(load_in_previous_data_files, inputs=[summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, original_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
            success(sample_reference_table_summaries, inputs=[master_reference_df_state, random_seed], outputs=[summary_reference_table_sample_state, summarised_references_markdown], api_name="sample_summaries").\
                success(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, temperature_slide, original_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio, output_folder_state, context_textbox, aws_access_key_textbox, aws_secret_key_textbox], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown, log_files_output, overall_summarisation_input_files], api_name="summarise_topics")

    latest_summary_completed_num.change(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, temperature_slide, original_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio, output_folder_state, context_textbox], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown, log_files_output, overall_summarisation_input_files], scroll_to_output=True)

    # SUMMARISE WHOLE TABLE PAGE
    overall_summarise_previous_data_btn.click(load_in_previous_data_files, inputs=[overall_summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, original_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
            success(overall_summary, inputs=[master_unique_topics_df_state, model_choice, in_api_key, temperature_slide, unique_topics_table_file_name_textbox, output_folder_state, in_colnames, context_textbox, aws_access_key_textbox, aws_secret_key_textbox], outputs=[overall_summary_output_files, overall_summarised_output_markdown, summarised_output_df], scroll_to_output=True, api_name="overall_summary")

    ###
    # CONTINUE PREVIOUS TOPIC EXTRACTION PAGE
    ###

    # If uploaded partially completed consultation files do this. This should then start up the 'latest_batch_completed' change action above to continue extracting topics.
    continue_previous_data_files_btn.click(
        load_in_data_file, inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, original_data_file_name_textbox, total_number_of_batches]).\
        success(load_in_previous_data_files, inputs=[in_previous_data_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed, in_previous_data_files_status, original_data_file_name_textbox, unique_topics_table_file_name_textbox])
    
    ###
    # VERIFY TEXT TITLES/DESCRIPTIONS
    ###

    # Tabular data upload
    verify_in_data_files.upload(fn=put_columns_in_df, inputs=[verify_in_data_files], outputs=[verify_in_colnames, verify_in_excel_sheets, original_data_file_name_textbox, join_colnames])

    verify_titles_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, original_data_file_name_textbox, display_topic_table_markdown]).\
    success(load_in_data_file,
        inputs = [verify_in_data_files, verify_in_colnames, batch_size_number, verify_in_excel_sheets], outputs = [file_data_state, original_data_file_name_textbox, total_number_of_batches], api_name="verify_load_data").\
    success(fn=verify_titles,
        inputs=[verify_in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, display_topic_table_markdown, original_data_file_name_textbox, total_number_of_batches, verify_in_api_key, temperature_slide, verify_in_colnames, verify_model_choice, candidate_topics, latest_batch_completed, display_topic_table_markdown, text_output_file_list_state, log_files_output_list_state, first_loop_state, conversation_metadata_textbox, verify_titles_prompt_textbox, prompt_2_textbox, prompt_3_textbox, verify_titles_system_prompt_textbox, verify_titles_system_prompt_textbox, verify_titles_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number, sentiment_checkbox, force_zero_shot_radio, produce_structures_summary_radio, aws_access_key_textbox, aws_secret_key_textbox, in_excel_sheets, output_folder_state],        
        outputs=[verify_display_topic_table_markdown, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, verify_titles_file_output, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, deduplication_input_files, summarisation_input_files, modifiable_unique_topics_df_state, verify_modification_input_files_placeholder], api_name="verify_descriptions")
    
    ###
    # LLM SETTINGS PAGE
    ###

    reference_df_data_file_name_textbox = gr.Textbox(label="reference_df_data_file_name_textbox", visible=False)
    master_reference_df_state_joined = gr.Dataframe(visible=False)

    join_cols_btn.click(fn=load_in_previous_reference_file, inputs=[in_join_files], outputs=[master_reference_df_state, reference_df_data_file_name_textbox]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, original_data_file_name_textbox, total_number_of_batches]).\
    success(fn=join_cols_onto_reference_df, inputs=[master_reference_df_state, file_data_state, join_colnames, reference_df_data_file_name_textbox], outputs=[master_reference_df_state_joined, out_join_files])

    # Export to xlsx file
    export_xlsx_btn.click(collect_output_csvs_and_create_excel_output, inputs=[in_data_files, in_colnames, original_data_file_name_textbox, in_group_col, model_choice, master_reference_df_state, master_unique_topics_df_state, summarised_output_df, missing_df_state, output_folder_state], outputs=[out_xlsx_files], api_name="export_xlsx")

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, output_folder_state, session_hash_textbox, input_folder_state])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    access_callback.setup([session_hash_textbox], ACCESS_LOGS_FOLDER)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    # Log usage when making a query
    usage_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    usage_callback.setup([session_hash_textbox, original_data_file_name_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], USAGE_LOGS_FOLDER)

    conversation_metadata_textbox.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, original_data_file_name_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    # User submitted feedback
    feedback_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    feedback_callback.setup([data_feedback_radio, data_further_details_text, original_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], FEEDBACK_LOGS_FOLDER)

    data_submit_feedback_btn.click(lambda *args: feedback_callback.flag(list(args)), [data_feedback_radio, data_further_details_text, original_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[data_further_details_text])

    in_view_table.upload(view_table, inputs=[in_view_table], outputs=[view_table_markdown])

###
# APP RUN
###

if __name__ == "__main__":
    if COGNITO_AUTH == "1":
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, auth=authenticate_user, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
    else:
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, inbrowser=True, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)