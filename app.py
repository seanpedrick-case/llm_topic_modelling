import os
import socket
import spaces
from tools.helper_functions import ensure_output_folder_exists, add_folder_to_path, put_columns_in_df, get_connection_params, output_folder, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, model_full_names, view_table, empty_output_vars_extract_topics, empty_output_vars_summarise, RUN_LOCAL_MODEL
from tools.aws_functions import upload_file_to_s3, RUN_AWS_FUNCTIONS
from tools.llm_api_call import extract_topics, load_in_data_file, load_in_previous_data_files, sample_reference_table_summaries, summarise_output_topics, batch_size_default, deduplicate_topics, modify_existing_output_tables
from tools.auth import authenticate_user
from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt
#from tools.aws_functions import load_data_from_aws
import gradio as gr
import pandas as pd
from datetime import datetime

today_rev = datetime.now().strftime("%Y%m%d")

ensure_output_folder_exists()

host_name = socket.gethostname()
# print("host_name is:", host_name)

access_logs_data_folder = 'logs/' + today_rev + '/' + host_name + '/'
feedback_data_folder = 'feedback/' + today_rev + '/' + host_name + '/'
usage_data_folder = 'usage/' + today_rev + '/' + host_name + '/'
file_input_height = 150

print("RUN_LOCAL_MODEL is:", RUN_LOCAL_MODEL)

if RUN_LOCAL_MODEL == "1":
    default_model_choice = "gemma_2b_it_local"

elif RUN_AWS_FUNCTIONS == "1":
    default_model_choice = "anthropic.claude-3-haiku-20240307-v1:0"

else:
    default_model_choice = "gemini-2.0-flash"

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Base())

with app:

    ###
    # STATE VARIABLES
    ###

    text_output_file_list_state = gr.State([])
    text_output_modify_file_list_state = gr.State([])
    log_files_output_list_state = gr.State([]) 
    first_loop_state = gr.State(True)
    second_loop_state = gr.State(False)
    modified_unique_table_change_bool = gr.State(True) # This boolean is used to flag whether a file upload should change just the modified unique table object on the second tab

    file_data_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="file_data_state", visible=False, type="pandas")
    master_topic_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_topic_df_state", visible=False, type="pandas")
    master_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_unique_topics_df_state", visible=False, type="pandas")
    master_reference_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_reference_df_state", visible=False, type="pandas")

    master_modify_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_modify_unique_topics_df_state", visible=False, type="pandas")
    master_modify_reference_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_modify_reference_df_state", visible=False, type="pandas")
    
 
    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    # Logging state
    log_file_name = 'log.csv'

    access_logs_state = gr.State(access_logs_data_folder + log_file_name)
    access_s3_logs_loc_state = gr.State(access_logs_data_folder)
    usage_logs_state = gr.State(usage_data_folder + log_file_name)
    usage_s3_logs_loc_state = gr.State(usage_data_folder)
    feedback_logs_state = gr.State(feedback_data_folder + log_file_name)
    feedback_s3_logs_loc_state = gr.State(feedback_data_folder)

    # Summary state objects
    summary_reference_table_sample_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="summary_reference_table_sample_state", visible=False, type="pandas")
    master_reference_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_reference_df_revised_summaries_state", visible=False, type="pandas")
    master_unique_topics_df_revised_summaries_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=0, row_count = (0, "dynamic"), label="master_unique_topics_df_revised_summaries_state", visible=False, type="pandas")
    summarised_references_markdown = gr.Markdown("", visible=False)
    summarised_outputs_list = gr.Dropdown(value=[], choices=[], visible=False, label="List of summarised outputs", allow_custom_value=True)
    latest_summary_completed_num = gr.Number(0, visible=False)

    reference_data_file_name_textbox = gr.Textbox(label = "Reference data file name", value="", visible=False)
    unique_topics_table_file_name_textbox = gr.Textbox(label="Unique topics data file name textbox", visible=False)

    ###
    # UI LAYOUT
    ###

    gr.Markdown(
    """# Large language model topic modelling

    Extract topics and summarise outputs using Large Language Models (LLMs, Gemma 2B instruct if local, Gemini Flash/Pro, or Claude 3 through AWS Bedrock if running on AWS). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and relevant text rows related to them. The prompts are designed for topic modelling public consultations, but they can be adapted to different contexts (see the LLM settings tab to modify). 
    
    Instructions on use can be found in the README.md file. Try it out with this [dummy development consultation dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation), which you can also try with [zero-shot topics](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/blob/main/example_zero_shot.csv), or this [dummy case notes dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_case_notes).

    You can use an AWS Bedrock model (Claude 3, paid), or Gemini (a free API, but with strict limits for the Pro model). Due to the strict API limits for the best model (Pro 1.5), the use of Gemini requires an API key. To set up your own Gemini API key, go [here](https://aistudio.google.com/app/u/1/plan_information). 

    NOTE: that **API calls to Gemini are not considered secure**, so please only submit redacted, non-sensitive tabular files to this source. Also, large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.""")
    
    with gr.Tab(label="Extract topics"):
        gr.Markdown(
        """
        ### Choose a tabular data file (xlsx or csv) of open text to extract topics from.
        """
        )
        with gr.Row():
            model_choice = gr.Dropdown(value = default_model_choice, choices = model_full_names, label="LLM model to use", multiselect=False)
            in_api_key = gr.Textbox(value = "", label="Enter Gemini API key (only if using Google API models)", lines=1, type="password")

        with gr.Accordion("Upload xlsx or csv file", open = True):
            in_data_files = gr.File(height=file_input_height, label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
        
        in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheet"], multiselect = False, label="Select the Excel sheet.", visible=False, allow_custom_value=True)
        in_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = False, label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.", allow_custom_value=True, interactive=True)
        
        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(height=file_input_height, label="Input topics from file (csv). File should have at least one column with a header, and all topic names below this. Using the headers 'General Topic' and/or 'Subtopic' will allow for these columns to be suggested to the model.")
            force_zero_shot_radio = gr.Radio(label="Force responses into zero shot topics", value="No", choices=["Yes", "No"])

        context_textbox = gr.Textbox(label="Write up to one sentence giving context to the large language model for your task (e.g. 'Consultation for the construction of flats on Main Street')")

        sentiment_checkbox = gr.Radio(label="Choose sentiment categories to split responses", value="Negative, Neutral, or Positive", choices=["Negative, Neutral, or Positive", "Negative or Positive", "Do not assess sentiment"])

        extract_topics_btn = gr.Button("Extract topics", variant="primary")
        
        topic_extraction_output_files = gr.File(height=file_input_height, label="Output files")
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
        gr.Markdown(
        """
        Load in previously completed Extract Topics output files ('reference_table', and 'unique_topics' files) to modify topics, deduplicate topics, or summarise the outputs. If you want pivot table outputs, please load in the original data file along with the selected open text column on the first tab before deduplicating or summarising.
        """)



        with gr.Accordion("Modify existing topics", open = False):
            modification_input_files = gr.File(height=file_input_height, label="Upload files to modify topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])

            modifiable_unique_topics_df_state = gr.Dataframe(value=pd.DataFrame(), headers=None, col_count=(4, "fixed"), row_count = (1, "fixed"), visible=True, type="pandas")

            save_modified_files_button = gr.Button(value="Save modified topic names")


        with gr.Accordion("Upload reference data file and unique data files", open = True):

            
            ### DEDUPLICATION
            deduplication_input_files = gr.File(height=file_input_height, label="Upload files to deduplicate topics", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            deduplication_input_files_status = gr.Textbox(value = "", label="Previous file input", visible=False)

            with gr.Row():
                merge_general_topics_drop = gr.Dropdown(label="Merge general topic values together for duplicate subtopics.", value="No", choices=["Yes", "No"])
                merge_sentiment_drop = gr.Dropdown(label="Merge sentiment values together for duplicate subtopics.", value="No", choices=["Yes", "No"])                
                deduplicate_score_threshold = gr.Number(label="Similarity threshold with which to determine duplicates.", value = 90, minimum=5, maximum=100, precision=0)

            deduplicate_previous_data_btn = gr.Button("Deduplicate topics", variant="primary")


            ### SUMMARISATION            
            summarisation_input_files = gr.File(height=file_input_height, label="Upload files to summarise", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])

            summarise_format_radio = gr.Radio(label="Choose summary type", value="Return a summary up to two paragraphs long that includes as much detail as possible from the original text", choices=["Return a summary up to two paragraphs long that includes as much detail as possible from the original text", "Return a concise summary up to one paragraph long that summarises only the most important themes from the original text"])
            
            summarise_previous_data_btn = gr.Button("Summarise topics", variant="primary")
            summary_output_files = gr.File(height=file_input_height, label="Summarised output files", interactive=False)
            summarised_output_markdown = gr.Markdown(value="### Summarised table will appear here", show_copy_button=True)

    with gr.Tab(label="Continue unfinished topic extraction"):
        gr.Markdown(
        """
        ### Load in output files from a previous topic extraction process and continue topic extraction with new data.
        """)

        with gr.Accordion("Upload reference data file and unique data files", open = True):
            in_previous_data_files = gr.File(height=file_input_height, label="Choose output csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            in_previous_data_files_status = gr.Textbox(value = "", label="Previous file input")
            continue_previous_data_files_btn = gr.Button(value="Continue previous topic extraction", variant="primary")

    
    with gr.Tab(label="Topic table viewer"):
        gr.Markdown(
        """
        ### View a 'unique_topic_table' csv file in markdown format.
        """)
    
        in_view_table = gr.File(height=file_input_height, label="Choose unique topic csv files", file_count= "single", file_types=['.csv', '.parquet', '.csv.gz'])
        view_table_markdown = gr.Markdown(value = "", label="View table", show_copy_button=True)

    with gr.Tab(label="Topic extraction settings"):
        gr.Markdown(
        """
        Define settings that affect large language model output.
        """)
        with gr.Accordion("Settings for LLM generation", open = True):
            temperature_slide = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Choose LLM temperature setting")
            batch_size_number = gr.Number(label = "Number of responses to submit in a single LLM query", value = batch_size_default, precision=0, minimum=1, maximum=100)
            random_seed = gr.Number(value=42, label="Random seed for LLM generation", visible=False)            

        with gr.Accordion("Prompt settings", open = True):
            number_of_prompts = gr.Number(value=1, label="Number of prompts to send to LLM in sequence", minimum=1, maximum=3, visible=False)
            system_prompt_textbox = gr.Textbox(label="Initial system prompt", lines = 4, value = system_prompt)
            initial_table_prompt_textbox = gr.Textbox(label = "Initial topics prompt", lines = 8, value = initial_table_prompt)
            prompt_2_textbox = gr.Textbox(label = "Prompt 2", lines = 8, value = prompt2, visible=False)
            prompt_3_textbox = gr.Textbox(label = "Prompt 3", lines = 8, value = prompt3, visible=False)
            add_to_existing_topics_system_prompt_textbox = gr.Textbox(label="Additional topics system prompt", lines = 4, value = add_existing_topics_system_prompt)
            add_to_existing_topics_prompt_textbox = gr.Textbox(label = "Additional topics prompt", lines = 8, value = add_existing_topics_prompt)
            
        log_files_output = gr.File(height=file_input_height, label="Log file output", interactive=False)
        conversation_metadata_textbox = gr.Textbox(label="Query metadata - usage counts and other parameters", interactive=False, lines=8)

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

     # Tabular data upload
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets, reference_data_file_name_textbox])

    extract_topics_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, file_data_state, reference_data_file_name_textbox, display_topic_table_markdown]).\
    success(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, reference_data_file_name_textbox, total_number_of_batches], api_name="load_data").\
    success(fn=extract_topics,                           
        inputs=[in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, display_topic_table_markdown, reference_data_file_name_textbox, total_number_of_batches, in_api_key, temperature_slide, in_colnames, model_choice, candidate_topics, latest_batch_completed, display_topic_table_markdown, text_output_file_list_state, log_files_output_list_state, first_loop_state, conversation_metadata_textbox, initial_table_prompt_textbox, prompt_2_textbox, prompt_3_textbox, system_prompt_textbox, add_to_existing_topics_system_prompt_textbox, add_to_existing_topics_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number, sentiment_checkbox, force_zero_shot_radio],        
        outputs=[display_topic_table_markdown, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, deduplication_input_files, summarisation_input_files, modifiable_unique_topics_df_state, modification_input_files], api_name="extract_topics")
    
    
    # If the output file count text box changes, keep going with redacting each data file until done. Then reveal the feedback buttons.
    # latest_batch_completed.change(fn=extract_topics,                                  
    #     inputs=[in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, display_topic_table_markdown, reference_data_file_name_textbox, total_number_of_batches, in_api_key, temperature_slide, in_colnames, model_choice, candidate_topics, latest_batch_completed, display_topic_table_markdown, text_output_file_list_state, log_files_output_list_state, second_loop_state, conversation_metadata_textbox, initial_table_prompt_textbox, prompt_2_textbox, prompt_3_textbox, system_prompt_textbox, add_to_existing_topics_system_prompt_textbox, add_to_existing_topics_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number, sentiment_checkbox, force_zero_shot_radio],
    #     outputs=[display_topic_table_markdown, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, deduplication_input_files, summarisation_input_files, modifiable_unique_topics_df_state, modification_input_files]).\
    #     success(fn = reveal_feedback_buttons,
    #         outputs=[data_feedback_radio, data_further_details_text, data_submit_feedback_btn, data_feedback_title], scroll_to_output=True)
    
    # If you upload data into the deduplication input box, the modifiable topic dataframe box is updated
    modification_input_files.change(fn=load_in_previous_data_files, inputs=[modification_input_files, modified_unique_table_change_bool], outputs=[modifiable_unique_topics_df_state, master_modify_reference_df_state, master_modify_unique_topics_df_state, reference_data_file_name_textbox, unique_topics_table_file_name_textbox, text_output_modify_file_list_state])
   

    # Modify output table with custom topic names
    save_modified_files_button.click(fn=modify_existing_output_tables, inputs=[master_modify_unique_topics_df_state, modifiable_unique_topics_df_state, master_modify_reference_df_state, text_output_modify_file_list_state], outputs=[master_unique_topics_df_state, master_reference_df_state, topic_extraction_output_files, text_output_file_list_state, deduplication_input_files, summarisation_input_files, reference_data_file_name_textbox, unique_topics_table_file_name_textbox, summarised_output_markdown])
    
    # When button pressed, deduplicate data
    deduplicate_previous_data_btn.click(load_in_previous_data_files, inputs=[deduplication_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, reference_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
        success(deduplicate_topics, inputs=[master_reference_df_state, master_unique_topics_df_state, reference_data_file_name_textbox, unique_topics_table_file_name_textbox, in_excel_sheets, merge_sentiment_drop, merge_general_topics_drop, deduplicate_score_threshold, in_data_files, in_colnames], outputs=[master_reference_df_state, master_unique_topics_df_state, summarisation_input_files, log_files_output, summarised_output_markdown], scroll_to_output=True)
    
    # When button pressed, summarise previous data
    summarise_previous_data_btn.click(empty_output_vars_summarise, inputs=None, outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox]).\
        success(load_in_previous_data_files, inputs=[summarisation_input_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, deduplication_input_files_status, reference_data_file_name_textbox, unique_topics_table_file_name_textbox]).\
            success(sample_reference_table_summaries, inputs=[master_reference_df_state, master_unique_topics_df_state, random_seed], outputs=[summary_reference_table_sample_state, summarised_references_markdown, master_reference_df_state, master_unique_topics_df_state]).\
                success(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, summarised_references_markdown, temperature_slide, reference_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown, log_files_output])

    latest_summary_completed_num.change(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, summarised_references_markdown, temperature_slide, reference_data_file_name_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, in_data_files, in_excel_sheets, in_colnames, log_files_output_list_state, summarise_format_radio], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown, log_files_output], scroll_to_output=True)

    # If uploaded partially completed consultation files do this. This should then start up the 'latest_batch_completed' change action above to continue extracting topics.
    continue_previous_data_files_btn.click(
            load_in_data_file, inputs = [in_data_files, in_colnames, batch_size_number, in_excel_sheets], outputs = [file_data_state, reference_data_file_name_textbox, total_number_of_batches]).\
            success(load_in_previous_data_files, inputs=[in_previous_data_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed, in_previous_data_files_status, reference_data_file_name_textbox])

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    access_callback.setup([session_hash_textbox], access_logs_data_folder)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # Log usage usage when making a query
    usage_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    usage_callback.setup([session_hash_textbox, reference_data_file_name_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], usage_data_folder)

    conversation_metadata_textbox.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, reference_data_file_name_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # User submitted feedback
    feedback_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    feedback_callback.setup([data_feedback_radio, data_further_details_text, reference_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], feedback_data_folder)

    data_submit_feedback_btn.click(lambda *args: feedback_callback.flag(list(args)), [data_feedback_radio, data_further_details_text, reference_data_file_name_textbox, model_choice, temperature_slide, display_topic_table_markdown, conversation_metadata_textbox], None, preprocess=False).\
        success(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])

    in_view_table.upload(view_table, inputs=[in_view_table], outputs=[view_table_markdown])

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')

MAX_QUEUE_SIZE = int(get_or_create_env_var('MAX_QUEUE_SIZE', '5'))
print(f'The value of MAX_QUEUE_SIZE is {MAX_QUEUE_SIZE}')

MAX_FILE_SIZE = get_or_create_env_var('MAX_FILE_SIZE', '100mb')
print(f'The value of MAX_FILE_SIZE is {MAX_FILE_SIZE}')

GRADIO_SERVER_PORT = int(get_or_create_env_var('GRADIO_SERVER_PORT', '7861'))
print(f'The value of GRADIO_SERVER_PORT is {GRADIO_SERVER_PORT}')

ROOT_PATH = get_or_create_env_var('ROOT_PATH', '')
print(f'The value of ROOT_PATH is {ROOT_PATH}')

if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, auth=authenticate_user, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
    else:
        app.queue(max_size=MAX_QUEUE_SIZE).launch(show_error=True, inbrowser=True, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)