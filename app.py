import os
import socket
import spaces
from tools.helper_functions import ensure_output_folder_exists, add_folder_to_path, put_columns_in_df, get_connection_params, output_folder, get_or_create_env_var, reveal_feedback_buttons, wipe_logs, model_full_names, view_table, empty_output_vars_extract_topics, empty_output_vars_summarise, RUN_LOCAL_MODEL
from tools.aws_functions import upload_file_to_s3, RUN_AWS_FUNCTIONS
from tools.llm_api_call import extract_topics, load_in_data_file, load_in_previous_data_files, sample_reference_table_summaries, summarise_output_topics, batch_size_default
from tools.auth import authenticate_user
from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt, add_existing_topics_system_prompt, add_existing_topics_prompt
#from tools.aws_functions import load_data_from_aws
import gradio as gr
import pandas as pd
from datetime import datetime

today_rev = datetime.now().strftime("%Y%m%d")

ensure_output_folder_exists()

host_name = socket.gethostname()
print("host_name is:", host_name)

access_logs_data_folder = 'logs/' + today_rev + '/' + host_name + '/'
feedback_data_folder = 'feedback/' + today_rev + '/' + host_name + '/'
usage_data_folder = 'usage/' + today_rev + '/' + host_name + '/'

print("RUN_LOCAL_MODEL is:", RUN_LOCAL_MODEL)

if RUN_LOCAL_MODEL == "1":
    default_model_choice = "gemma_2b_it_local"

elif RUN_AWS_FUNCTIONS == "1":
    default_model_choice = "anthropic.claude-3-haiku-20240307-v1:0"

else:
    default_model_choice = "gemini-1.5-flash-002"

# Create the gradio interface
app = gr.Blocks(theme = gr.themes.Base())

with app:

    ###
    # STATE VARIABLES
    ###

    text_output_file_list_state = gr.State([])
    log_files_output_list_state = gr.State([]) 
    first_loop_state = gr.State(True)
    second_loop_state = gr.State(False)

    file_data_state = gr.State(pd.DataFrame())
    master_topic_df_state = gr.State(pd.DataFrame())
    master_reference_df_state = gr.State(pd.DataFrame())
    master_unique_topics_df_state = gr.State(pd.DataFrame())
 
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
    summary_reference_table_sample_state = gr.State(pd.DataFrame())
    master_reference_df_revised_summaries_state = gr.State(pd.DataFrame())
    master_unique_topics_df_revised_summaries_state = gr.State(pd.DataFrame())
    summarised_references_markdown = gr.Markdown("", visible=False)
    summarised_outputs_list = gr.Dropdown(value=[], choices=[], visible=False, label="List of summarised outputs", allow_custom_value=True)
    latest_summary_completed_num = gr.Number(0, visible=False)

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
            in_data_files = gr.File(label="Choose Excel or csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
        
        in_excel_sheets = gr.Dropdown(choices=["Choose Excel sheet"], multiselect = False, label="Select the Excel sheet.", visible=False, allow_custom_value=True)
        in_colnames = gr.Dropdown(choices=["Choose column with responses"], multiselect = False, label="Select the open text column of interest.", allow_custom_value=True, interactive=True)
        
        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv). File should have a single column with a header, and all topic keywords below.")

        context_textbox = gr.Textbox(label="Write up to one sentence giving context to the large language model for your task (e.g. 'Consultation for the construction of flats on Main Street')")

        extract_topics_btn = gr.Button("Extract topics from open text", variant="primary")
        
        text_output_summary = gr.Markdown(value="### Language model response will appear here")
        text_output_file = gr.File(label="Output files")
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

    with gr.Tab(label="Summarise topic outputs"):
        gr.Markdown(
        """
        ### Load in previously completed Extract Topics output files ('reference_table', and 'unique_topics' files) to summarise the outputs.
        """)
        with gr.Accordion("Upload reference data file and unique data files", open = True):
            summarisation_in_previous_data_files = gr.File(label="Choose output csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            summarisation_in_previous_data_files_status = gr.Textbox(value = "", label="Previous file input", visible=False)
            summarise_previous_data_btn = gr.Button("Summarise existing topics", variant="primary")
            summary_output_files = gr.File(label="Summarised output files", interactive=False)
            summarised_output_markdown = gr.Markdown(value="### Summarised table will appear here")

    with gr.Tab(label="Continue previous topic extraction"):
        gr.Markdown(
        """
        ### Load in data files from a previous attempt at extracting topics to continue it.
        """)

        with gr.Accordion("Upload reference data file and unique data files", open = True):
            in_previous_data_files = gr.File(label="Choose output csv files", file_count= "multiple", file_types=['.xlsx', '.xls', '.csv', '.parquet', '.csv.gz'])
            in_previous_data_files_status = gr.Textbox(value = "", label="Previous file input")
            continue_previous_data_files_btn = gr.Button(value="Continue previous topic extraction", variant="primary")

    
    with gr.Tab(label="View output topics table"):
        gr.Markdown(
        """
        ### View a 'unique_topic_table' csv file in markdown format.
        """)
    
        in_view_table = gr.File(label="Choose unique topic csv files", file_count= "single", file_types=['.csv', '.parquet', '.csv.gz'])
        view_table_markdown = gr.Markdown(value = "", label="View table")

    with gr.Tab(label="LLM settings"):
        gr.Markdown(
        """
        Define settings that affect large language model output.
        """)
        with gr.Accordion("Settings for LLM generation", open = True):
            temperature_slide = gr.Slider(minimum=0.1, maximum=1.0, value=0.1, label="Choose LLM temperature setting")
            batch_size_number = gr.Number(label = "Number of responses to submit in a single LLM query", value = batch_size_default, precision=0)
            random_seed = gr.Number(value=42, label="Random seed for LLM generation", visible=False)            

        with gr.Accordion("Prompt settings", open = True):
            number_of_prompts = gr.Number(value=1, label="Number of prompts to send to LLM in sequence", minimum=1, maximum=3)
            system_prompt_textbox = gr.Textbox(label="System prompt", lines = 4, value = system_prompt)
            initial_table_prompt_textbox = gr.Textbox(label = "Prompt 1", lines = 8, value = initial_table_prompt)
            prompt_2_textbox = gr.Textbox(label = "Prompt 2", lines = 8, value = prompt2, visible=False)
            prompt_3_textbox = gr.Textbox(label = "Prompt 3", lines = 8, value = prompt3, visible=False)
            add_to_existing_topics_system_prompt_textbox = gr.Textbox(label="Summary system prompt", lines = 4, value = add_existing_topics_system_prompt)
            add_to_existing_topics_prompt_textbox = gr.Textbox(label = "Summary prompt", lines = 8, value = add_existing_topics_prompt)
            
        log_files_output = gr.File(label="Log file output", interactive=False)
        conversation_metadata_textbox = gr.Textbox(label="Query metadata - usage counts and other parameters", interactive=False, lines=8)

        # Invisible text box to hold the session hash/username just for logging purposes
        session_hash_textbox = gr.Textbox(label = "Session hash", value="", visible=False) 
        data_file_names_textbox = gr.Textbox(label = "Data file name", value="", visible=False)
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
    in_data_files.upload(fn=put_columns_in_df, inputs=[in_data_files], outputs=[in_colnames, in_excel_sheets, data_file_names_textbox])

    extract_topics_btn.click(fn=empty_output_vars_extract_topics, inputs=None, outputs=[master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, text_output_file, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number]).\
    then(load_in_data_file,                           
        inputs = [in_data_files, in_colnames, batch_size_number], outputs = [file_data_state, data_file_names_textbox, total_number_of_batches], api_name="load_data").then(\
        fn=extract_topics,                           
        inputs=[in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, text_output_summary, data_file_names_textbox, total_number_of_batches, in_api_key, temperature_slide, in_colnames, model_choice, candidate_topics, latest_batch_completed, text_output_summary, text_output_file_list_state, log_files_output_list_state, first_loop_state, conversation_metadata_textbox, initial_table_prompt_textbox, prompt_2_textbox, prompt_3_textbox, system_prompt_textbox, add_to_existing_topics_system_prompt_textbox, add_to_existing_topics_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number],        
        outputs=[text_output_summary, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, text_output_file, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, summarisation_in_previous_data_files], api_name="extract_topics")
    
    # If the output file count text box changes, keep going with redacting each data file until done. Then reveal the feedback buttons.
    latest_batch_completed.change(fn=extract_topics,                                  
        inputs=[in_data_files, file_data_state, master_topic_df_state, master_reference_df_state, master_unique_topics_df_state, text_output_summary, data_file_names_textbox, total_number_of_batches, in_api_key, temperature_slide, in_colnames, model_choice, candidate_topics, latest_batch_completed, text_output_summary, text_output_file_list_state, log_files_output_list_state, second_loop_state, conversation_metadata_textbox, initial_table_prompt_textbox, prompt_2_textbox, prompt_3_textbox, system_prompt_textbox, add_to_existing_topics_system_prompt_textbox, add_to_existing_topics_prompt_textbox, number_of_prompts, batch_size_number, context_textbox, estimated_time_taken_number],
        outputs=[text_output_summary, master_topic_df_state, master_unique_topics_df_state, master_reference_df_state, text_output_file, text_output_file_list_state, latest_batch_completed, log_files_output, log_files_output_list_state, conversation_metadata_textbox, estimated_time_taken_number, summarisation_in_previous_data_files]).\
        then(fn = reveal_feedback_buttons,
        outputs=[data_feedback_radio, data_further_details_text, data_submit_feedback_btn, data_feedback_title], scroll_to_output=True)
    
    # When button pressed, summarise previous data
    summarise_previous_data_btn.click(empty_output_vars_summarise, inputs=None, outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox]).\
    then(load_in_previous_data_files, inputs=[summarisation_in_previous_data_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed_no_loop, summarisation_in_previous_data_files_status, data_file_names_textbox]).\
    then(sample_reference_table_summaries, inputs=[master_reference_df_state, master_unique_topics_df_state, random_seed], outputs=[summary_reference_table_sample_state, summarised_references_markdown, master_reference_df_state, master_unique_topics_df_state]).\
    then(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, summarised_references_markdown, temperature_slide, data_file_names_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown])

    latest_summary_completed_num.change(summarise_output_topics, inputs=[summary_reference_table_sample_state, master_unique_topics_df_state, master_reference_df_state, model_choice, in_api_key, summarised_references_markdown, temperature_slide, data_file_names_textbox, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox], outputs=[summary_reference_table_sample_state, master_unique_topics_df_revised_summaries_state, master_reference_df_revised_summaries_state, summary_output_files, summarised_outputs_list, latest_summary_completed_num, conversation_metadata_textbox, summarised_output_markdown])

    # If uploaded partially completed consultation files do this. This should then start up the 'latest_batch_completed' change action above to continue extracting topics.
    continue_previous_data_files_btn.click(
            load_in_data_file, inputs = [in_data_files, in_colnames, batch_size_number], outputs = [file_data_state, data_file_names_textbox, total_number_of_batches]).\
            then(load_in_previous_data_files, inputs=[in_previous_data_files], outputs=[master_reference_df_state, master_unique_topics_df_state, latest_batch_completed, in_previous_data_files_status, data_file_names_textbox])

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    access_callback.setup([session_hash_textbox], access_logs_data_folder)
    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # Log usage usage when making a query
    usage_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    usage_callback.setup([session_hash_textbox, data_file_names_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], usage_data_folder)

    conversation_metadata_textbox.change(lambda *args: usage_callback.flag(list(args)), [session_hash_textbox, data_file_names_textbox, model_choice, conversation_metadata_textbox, estimated_time_taken_number], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

    # User submitted feedback
    feedback_callback = gr.CSVLogger(dataset_file_name=log_file_name)
    feedback_callback.setup([data_feedback_radio, data_further_details_text, data_file_names_textbox, model_choice, temperature_slide, text_output_summary, conversation_metadata_textbox], feedback_data_folder)

    data_submit_feedback_btn.click(lambda *args: feedback_callback.flag(list(args)), [data_feedback_radio, data_further_details_text, data_file_names_textbox, model_choice, temperature_slide, text_output_summary, conversation_metadata_textbox], None, preprocess=False).\
    then(fn = upload_file_to_s3, inputs=[feedback_logs_state, feedback_s3_logs_loc_state], outputs=[data_further_details_text])

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