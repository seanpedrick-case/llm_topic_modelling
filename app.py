import os
from datetime import datetime

import gradio as gr
import pandas as pd

from tools.auth import authenticate_user
from tools.aws_functions import (
    download_file_from_s3,
    export_outputs_to_s3,
    upload_file_to_s3,
)
from tools.combine_sheets_into_xlsx import collect_output_csvs_and_create_excel_output
from tools.config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    ACCESS_LOGS_FOLDER,
    API_URL,
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    BATCH_SIZE_DEFAULT,
    COGNITO_AUTH,
    CONFIG_FOLDER,
    COST_CODES_PATH,
    CSV_ACCESS_LOG_HEADERS,
    CSV_FEEDBACK_LOG_HEADERS,
    CSV_USAGE_LOG_HEADERS,
    DEFAULT_COST_CODE,
    DIRECT_MODE_ADDITIONAL_SUMMARY_INSTRUCTIONS,
    DIRECT_MODE_ADDITIONAL_VALIDATION_ISSUES,
    DIRECT_MODE_BATCH_SIZE,
    DIRECT_MODE_CANDIDATE_TOPICS,
    DIRECT_MODE_CONTEXT,
    DIRECT_MODE_CREATE_XLSX_OUTPUT,
    DIRECT_MODE_DEDUP_METHOD,
    DIRECT_MODE_EXCEL_SHEETS,
    DIRECT_MODE_FORCE_SINGLE_TOPIC,
    DIRECT_MODE_FORCE_ZERO_SHOT,
    DIRECT_MODE_GROUP_BY,
    DIRECT_MODE_INFERENCE_SERVER_MODEL,
    DIRECT_MODE_INPUT_FILE,
    DIRECT_MODE_MAX_TIME_FOR_LOOP,
    DIRECT_MODE_MAX_TOKENS,
    DIRECT_MODE_MERGE_GENERAL_TOPICS,
    DIRECT_MODE_MERGE_SENTIMENT,
    DIRECT_MODE_MODEL_CHOICE,
    DIRECT_MODE_NO_OF_SAMPLED_SUMMARIES,
    DIRECT_MODE_OUTPUT_DIR,
    DIRECT_MODE_PREVIOUS_OUTPUT_FILES,
    DIRECT_MODE_PRODUCE_STRUCTURED_SUMMARY,
    DIRECT_MODE_RANDOM_SEED,
    DIRECT_MODE_SAMPLE_REFERENCE_TABLE,
    DIRECT_MODE_SENTIMENT,
    DIRECT_MODE_SHOW_PREVIOUS_TABLE,
    DIRECT_MODE_SIMILARITY_THRESHOLD,
    DIRECT_MODE_SUMMARY_FORMAT,
    # Direct mode variables
    DIRECT_MODE_TASK,
    DIRECT_MODE_TEMPERATURE,
    DIRECT_MODE_TEXT_COLUMN,
    DIRECT_MODE_USERNAME,
    DYNAMODB_ACCESS_LOG_HEADERS,
    DYNAMODB_FEEDBACK_LOG_HEADERS,
    DYNAMODB_USAGE_LOG_HEADERS,
    ENFORCE_COST_CODES,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    FEEDBACK_LOG_FILE_NAME,
    FEEDBACK_LOGS_FOLDER,
    FILE_INPUT_HEIGHT,
    GEMINI_API_KEY,
    GET_COST_CODES,
    GRADIO_SERVER_PORT,
    GRADIO_TEMP_DIR,
    HF_TOKEN,
    HOST_NAME,
    INPUT_FOLDER,
    INTRO_TEXT,
    LLM_SEED,
    LLM_TEMPERATURE,
    LOG_FILE_NAME,
    MAX_FILE_SIZE,
    MAX_QUEUE_SIZE,
    MPLCONFIGDIR,
    OUTPUT_COST_CODES_PATH,
    OUTPUT_DEBUG_FILES,
    OUTPUT_FOLDER,
    ROOT_PATH,
    RUN_AWS_FUNCTIONS,
    RUN_DIRECT_MODE,
    RUN_INFERENCE_SERVER,
    RUN_MCP_SERVER,
    S3_ACCESS_LOGS_FOLDER,
    S3_COST_CODES_PATH,
    S3_FEEDBACK_LOGS_FOLDER,
    S3_LOG_BUCKET,
    S3_OUTPUTS_FOLDER,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SAVE_OUTPUTS_TO_S3,
    SESSION_OUTPUT_FOLDER,
    SHOW_ADDITIONAL_INSTRUCTION_TEXTBOXES,
    SHOW_COSTS,
    SHOW_EXAMPLES,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_FILE_NAME,
    USAGE_LOGS_FOLDER,
    convert_string_to_boolean,
    default_model_choice,
    default_model_source,
    default_source_models,
    ensure_folder_exists,
    model_name_map,
    model_sources,
)
from tools.custom_csvlogger import CSVLogger_custom
from tools.dedup_summaries import (
    deduplicate_topics,
    deduplicate_topics_llm,
    overall_summary,
    wrapper_summarise_output_topics_per_group,
)
from tools.example_table_outputs import (
    case_notes_table,
    case_notes_table_grouped,
    case_notes_table_structured_summary,
    dummy_consultation_table,
    dummy_consultation_table_zero_shot,
)
from tools.helper_functions import (
    df_select_callback_cost,
    empty_output_vars_extract_topics,
    empty_output_vars_summarise,
    enforce_cost_codes,
    ensure_model_in_map,
    get_connection_params,
    join_cols_onto_reference_df,
    load_in_data_file,
    load_in_default_cost_codes,
    load_in_previous_data_files,
    load_in_previous_reference_file,
    move_overall_summary_output_files_to_front_page,
    put_columns_in_df,
    reset_base_dataframe,
    update_cost_code_dataframe_from_dropdown_select,
    update_model_choice,
    view_table,
)
from tools.llm_api_call import (
    all_in_one_pipeline,
    modify_existing_output_tables,
    validate_topics_wrapper,
    wrapper_extract_topics_per_column_value,
)
from tools.prompts import (
    add_existing_topics_prompt,
    add_existing_topics_system_prompt,
    initial_table_prompt,
    single_para_summary_format_prompt,
    system_prompt,
    two_para_summary_format_prompt,
)

ensure_folder_exists(CONFIG_FOLDER)
ensure_folder_exists(OUTPUT_FOLDER)
ensure_folder_exists(INPUT_FOLDER)
ensure_folder_exists(GRADIO_TEMP_DIR)
ensure_folder_exists(MPLCONFIGDIR)
ensure_folder_exists(FEEDBACK_LOGS_FOLDER)
ensure_folder_exists(ACCESS_LOGS_FOLDER)
ensure_folder_exists(USAGE_LOGS_FOLDER)

today_rev = datetime.now().strftime("%Y%m%d")

# Placeholders for example variables
in_data_files = gr.File(
    height=FILE_INPUT_HEIGHT,
    label="Choose Excel or csv files",
    file_count="multiple",
    file_types=[".xlsx", ".xls", ".csv", ".parquet"],
)
in_colnames = gr.Dropdown(
    choices=[""],
    multiselect=False,
    label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.",
    allow_custom_value=True,
    interactive=True,
)
if SHOW_ADDITIONAL_INSTRUCTION_TEXTBOXES == "True":
    context_textbox = gr.Textbox(
        label="Write up to one sentence giving context to the large language model for your task (e.g. 'Consultation for the construction of flats on Main Street')",
        visible=True,
    )
else:
    context_textbox = gr.Textbox(
        label="Write up to one sentence giving context to the large language model for your task (e.g. 'Consultation for the construction of flats on Main Street')",
        visible="hidden",
    )
topic_extraction_output_files_xlsx = gr.File(
    label="Overall summary xlsx file. CSV outputs are available on the 'Advanced' tab.",
    scale=1,
    interactive=False,
    file_count="multiple",
)
display_topic_table_markdown = gr.Markdown(value="", buttons=["copy"])
output_messages_textbox = gr.Textbox(
    value="", label="Output messages", scale=1, interactive=False, lines=4
)
candidate_topics = gr.File(
    height=FILE_INPUT_HEIGHT,
    label="Input topics from file (csv). File should have at least one column with a header, and all topic names below this. Using the headers 'General topic' and/or 'Subtopic' will allow for these columns to be suggested to the model. If a third column is present, it will be assumed to be a topic description.",
    file_count="single",
)
produce_structured_summary_radio = gr.Radio(
    label="Ask the model to produce structured summaries using the suggested topics as headers rather than extract topics",
    value="No",
    choices=["Yes", "No"],
)
in_group_col = gr.Dropdown(
    multiselect=False,
    label="Select the column to group results by",
    allow_custom_value=True,
    interactive=True,
)
batch_size_number = gr.Number(
    label="Number of responses to submit in a single LLM query (batch size)",
    value=BATCH_SIZE_DEFAULT,
    precision=0,
    minimum=1,
    maximum=50,
)

css = """
/* Target tab navigation buttons only - not buttons inside tab content */
/* Gradio renders tab buttons with role="tab" in the navigation area */
button[role="tab"] {
    font-size: 1.1em !important;
    padding: 0.75em 1.2em !important;
}

/* Alternative selectors for different Gradio versions */
.tab-nav button,
nav button[role="tab"],
div[class*="tab-nav"] button {
    font-size: 1.1em !important;
    padding: 0.75em 1.2em !important;
}
"""

# Create the gradio interface
app = gr.Blocks(
    fill_width=True,
    analytics_enabled=False,
    title="LLM topic modelling",
    delete_cache=(43200, 43200),
)

with app:

    ###
    # STATE VARIABLES
    ###

    # Workaround for Gradio 6 issue where 'hidden' element are still sometimes visible as a thing line in the UI
    with gr.Accordion(visible="hidden", elem_classes="hidden_component", open=False):
        text_output_file_list_state = gr.Dropdown(
            list(),
            allow_custom_value=True,
            visible="hidden",
            label="text_output_file_list_state",
            elem_classes="hidden_component",
        )
        text_output_modify_file_list_state = gr.Dropdown(
            list(),
            allow_custom_value=True,
            visible="hidden",
            label="text_output_modify_file_list_state",
            elem_classes="hidden_component",
        )
        log_files_output_list_state = gr.Dropdown(
            list(),
            allow_custom_value=True,
            visible="hidden",
            label="log_files_output_list_state",
            elem_classes="hidden_component",
        )
        first_loop_state = gr.Checkbox(
            True, visible="hidden", elem_classes="hidden_component"
        )
        second_loop_state = gr.Checkbox(
            False, visible="hidden", elem_classes="hidden_component"
        )
        modified_unique_table_change_bool = gr.Checkbox(
            True, visible="hidden", elem_classes="hidden_component"
        )  # This boolean is used to flag whether a file upload should change just the modified unique table object on the second tab

        file_data_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="file_data_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )
        master_topic_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="master_topic_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
        )
        master_unique_topics_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="master_unique_topics_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )
        master_reference_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="master_reference_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )
        missing_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="missing_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )

        master_modify_unique_topics_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="master_modify_unique_topics_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )
        master_modify_reference_df_state = gr.Dataframe(
            value=pd.DataFrame(),
            label="master_modify_reference_df_state",
            visible="hidden",
            type="pandas",
            interactive=True,
            elem_classes="hidden_component",
        )

        # Blank placeholder for conversation metadata textbox, as logging file output can get too long for large amounts of calls
        conversation_metadata_textbox_placeholder = gr.Textbox(
            value="",
            label="Query metadata - usage counts and other parameters",
            lines=8,
            visible="hidden",
            elem_classes="hidden_component",
        )

        session_hash_state = gr.Textbox(visible="hidden", value=HOST_NAME)
        output_folder_state = gr.Textbox(
            visible="hidden", value=OUTPUT_FOLDER, elem_classes="hidden_component"
        )
        input_folder_state = gr.Textbox(
            visible="hidden", value=INPUT_FOLDER, elem_classes="hidden_component"
        )

        # s3 bucket name
        s3_default_bucket = gr.Textbox(
            label="Default S3 bucket",
            value=S3_LOG_BUCKET,
            visible="hidden",
            elem_classes="hidden_component",
        )
        s3_log_bucket_name = gr.Textbox(
            visible="hidden", value=S3_LOG_BUCKET, elem_classes="hidden_component"
        )

        # S3 output settings
        s3_output_folder_state = gr.Textbox(
            label="s3_output_folder_state",
            value=S3_OUTPUTS_FOLDER,
            visible="hidden",
            elem_classes="hidden_component",
        )
        save_outputs_to_s3_checkbox = gr.Checkbox(
            label="save_outputs_to_s3_checkbox",
            value=convert_string_to_boolean(SAVE_OUTPUTS_TO_S3),
            visible="hidden",
            elem_classes="hidden_component",
        )

        # Logging variables
        access_logs_state = gr.Textbox(
            label="access_logs_state",
            value=ACCESS_LOGS_FOLDER + LOG_FILE_NAME,
            visible="hidden",
            elem_classes="hidden_component",
        )
        access_s3_logs_loc_state = gr.Textbox(
            label="access_s3_logs_loc_state",
            value=S3_ACCESS_LOGS_FOLDER,
            visible="hidden",
            elem_classes="hidden_component",
        )
        feedback_logs_state = gr.Textbox(
            label="feedback_logs_state",
            value=FEEDBACK_LOGS_FOLDER + FEEDBACK_LOG_FILE_NAME,
            visible="hidden",
            elem_classes="hidden_component",
        )
        feedback_s3_logs_loc_state = gr.Textbox(
            label="feedback_s3_logs_loc_state",
            value=S3_FEEDBACK_LOGS_FOLDER,
            visible="hidden",
            elem_classes="hidden_component",
        )
        usage_logs_state = gr.Textbox(
            label="usage_logs_state",
            value=USAGE_LOGS_FOLDER + USAGE_LOG_FILE_NAME,
            visible="hidden",
            elem_classes="hidden_component",
        )
        usage_s3_logs_loc_state = gr.Textbox(
            label="usage_s3_logs_loc_state",
            value=S3_USAGE_LOGS_FOLDER,
            visible="hidden",
            elem_classes="hidden_component",
        )

        # Logging for logged content
        logged_content_df = gr.Dataframe(
            label="logged_content_df",
            value=pd.DataFrame(),
            visible="hidden",
            type="pandas",
            elem_classes="hidden_component",
        )

        # Logging for input / output tokens
        input_tokens_num = gr.Textbox(
            "0",
            visible="hidden",
            label="Total input tokens",
            elem_classes="hidden_component",
        )
        output_tokens_num = gr.Textbox(
            "0",
            visible="hidden",
            label="Total output tokens",
            elem_classes="hidden_component",
        )
        number_of_calls_num = gr.Textbox(
            "0",
            visible="hidden",
            label="Total LLM calls",
            elem_classes="hidden_component",
        )

        # Additional UI components for validation
        max_tokens_num = gr.Number(
            value=8192,
            visible="hidden",
            label="Max tokens",
            elem_classes="hidden_component",
        )
        reasoning_suffix_textbox = gr.Textbox(
            value="",
            visible="hidden",
            label="Reasoning suffix",
            elem_classes="hidden_component",
        )
        output_debug_files_radio = gr.Radio(
            value="False",
            choices=["True", "False"],
            visible="hidden",
            label="Output debug files",
        )
        max_time_for_loop_num = gr.Number(
            value=99999,
            visible="hidden",
            label="Max time for loop",
            elem_classes="hidden_component",
        )

        # Summary state objects
        summary_reference_table_sample_state = gr.Dataframe(
            value=pd.DataFrame(),
            headers=None,
            column_count=None,
            label="summary_reference_table_sample_state",
            visible="hidden",
            type="pandas",
            elem_classes="hidden_component",
        )
        master_reference_df_revised_summaries_state = gr.Dataframe(
            value=pd.DataFrame(),
            headers=None,
            column_count=None,
            label="master_reference_df_revised_summaries_state",
            visible="hidden",
            type="pandas",
            elem_classes="hidden_component",
        )
        master_unique_topics_df_revised_summaries_state = gr.Dataframe(
            value=pd.DataFrame(),
            headers=None,
            column_count=None,
            label="master_unique_topics_df_revised_summaries_state",
            visible="hidden",
            type="pandas",
            elem_classes="hidden_component",
        )
        summarised_output_df = gr.Dataframe(
            value=pd.DataFrame(),
            headers=None,
            column_count=None,
            label="summarised_output_df",
            visible="hidden",
            type="pandas",
            elem_classes="hidden_component",
        )
        summarised_references_markdown = gr.Markdown(
            "", visible="hidden", elem_classes="hidden_component"
        )
        summarised_outputs_list = gr.Dropdown(
            value=list(),
            choices=list(),
            visible="hidden",
            label="List of summarised outputs",
            allow_custom_value=True,
            elem_classes="hidden_component",
        )
        latest_summary_completed_num = gr.Number(
            0, visible="hidden", elem_classes="hidden_component"
        )

        summary_xlsx_output_files_list = gr.Dropdown(
            value=list(),
            choices=list(),
            visible="hidden",
            label="List of xlsx summary output files",
            allow_custom_value=True,
            elem_classes="hidden_component",
        )

        original_data_file_name_textbox = gr.Textbox(
            label="Reference data file name",
            value="",
            visible="hidden",
            elem_classes="hidden_component",
        )
        working_data_file_name_textbox = gr.Textbox(
            label="Working data file name",
            value="",
            visible="hidden",
            elem_classes="hidden_component",
        )
        unique_topics_table_file_name_textbox = gr.Textbox(
            label="Unique topics data file name textbox",
            visible="hidden",
            elem_classes="hidden_component",
        )

        dummy_consultation_table_textbox = gr.Textbox(
            value=dummy_consultation_table,
            visible="hidden",
            label="Dummy consultation table",
            elem_classes="hidden_component",
        )
        case_notes_table_textbox = gr.Textbox(
            value=case_notes_table,
            visible="hidden",
            label="Case notes table",
            elem_classes="hidden_component",
        )

        model_name_map_state = gr.JSON(
            model_name_map,
            visible="hidden",
            label="model_name_map_state",
            elem_classes="hidden_component",
        )

        # Cost code elements
        s3_default_cost_codes_file = gr.Textbox(
            label="Default cost centre file",
            value=S3_COST_CODES_PATH,
            visible="hidden",
            elem_classes="hidden_component",
        )
        default_cost_codes_output_folder_location = gr.Textbox(
            label="Output default cost centre location",
            value=OUTPUT_COST_CODES_PATH,
            visible="hidden",
            elem_classes="hidden_component",
        )
        enforce_cost_code_textbox = gr.Textbox(
            label="Enforce cost code textbox",
            value=ENFORCE_COST_CODES,
            visible="hidden",
            elem_classes="hidden_component",
        )
        default_cost_code_textbox = gr.Textbox(
            label="Default cost code textbox",
            value=DEFAULT_COST_CODE,
            visible="hidden",
            elem_classes="hidden_component",
        )

        # Placeholders for elements that may be made visible later below depending on environment variables
        cost_code_dataframe_base = gr.Dataframe(
            value=pd.DataFrame(columns=["Cost code", "Description"]),
            label="Cost codes",
            type="pandas",
            buttons=["fullscreen", "copy"],
            show_search="filter",
            wrap=True,
            max_height=200,
            visible="hidden",
            interactive=True,
        )
        cost_code_dataframe = gr.Dataframe(
            value=pd.DataFrame(columns=["Cost code", "Description"]),
            type="pandas",
            visible="hidden",
            wrap=True,
            interactive=True,
            elem_classes="hidden_component",
        )
        cost_code_choice_drop = gr.Dropdown(
            value=DEFAULT_COST_CODE,
            label="Choose cost code for analysis. Please contact Finance if you can't find your cost code in the given list.",
            choices=[DEFAULT_COST_CODE],
            allow_custom_value=False,
            visible="hidden",
            elem_classes="hidden_component",
        )

        latest_batch_completed = gr.Number(
            value=0,
            label="Number of files prepared",
            interactive=False,
            visible="hidden",
            elem_classes="hidden_component",
        )
        # Duplicate version of the above variable for when you don't want to initiate the summarisation loop
        latest_batch_completed_no_loop = gr.Number(
            value=0,
            label="Number of files prepared",
            interactive=False,
            visible="hidden",
            elem_classes="hidden_component",
        )

        # Invisible text box to hold the session hash/username just for logging purposes
        session_hash_textbox = gr.Textbox(
            label="Session hash",
            value="",
            visible="hidden",
            elem_classes="hidden_component",
        )

        estimated_time_taken_number = gr.Number(
            label="Estimated time taken (seconds)",
            value=0.0,
            precision=1,
            visible="hidden",
            elem_classes="hidden_component",
        )  # This keeps track of the time taken to redact files for logging purposes.
        total_number_of_batches = gr.Number(
            label="Current batch number",
            value=1,
            precision=0,
            visible="hidden",
            elem_classes="hidden_component",
        )

        text_output_logs = gr.Textbox(
            label="Output summary logs",
            visible="hidden",
            elem_classes="hidden_component",
        )

    ###
    # UI LAYOUT
    ###

    gr.Markdown(INTRO_TEXT)

    if SHOW_EXAMPLES == "True":

        def show_info_box_on_click(
            in_data_files,
            in_colnames,
            context_textbox,
            original_data_file_name_textbox,
            topic_extraction_output_files_xlsx,
            display_topic_table_markdown,
            output_messages_textbox,
            candidate_topics,
            produce_structured_summary_radio,
            in_group_col,
            batch_size_number,
        ):
            gr.Info(
                "Example data loaded. Now click on the 'Extract topics...' button below to run the full suite of topic extraction, deduplication, and summarisation."
            )

        # Check if required example files exist before creating Examples
        # This prevents errors in CI environments where example files may not be present
        required_example_files = [
            "example_data/dummy_consultation_response.csv",
            "example_data/combined_case_notes.csv",
        ]
        example_files_exist = all(os.path.exists(f) for f in required_example_files)

        # Only create Examples if files exist, otherwise create empty Examples to avoid errors
        if example_files_exist:
            try:
                examples = gr.Examples(
                    examples=[
                        [
                            ["example_data/dummy_consultation_response.csv"],
                            "Response text",
                            "Consultation for the construction of flats on Main Street",
                            "dummy_consultation_response.csv",
                            [
                                "example_data/dummy_consultation_r_col_Response_text_Gemma_3_4B_topic_analysis.xlsx"
                            ],
                            dummy_consultation_table,
                            "Example output from the dummy consultation dataset successfully loaded. Download the xlsx outputs to the right to see full outputs.",
                            None,
                            "No",
                            None,
                            5,
                        ],
                        [
                            ["example_data/combined_case_notes.csv"],
                            "Case Note",
                            "Social Care case notes for young people",
                            "combined_case_notes.csv",
                            [
                                "example_data/combined_case_notes_col_Case_Note_Gemma_3_4B_topic_analysis.xlsx"
                            ],
                            case_notes_table,
                            "Example output from the case notes dataset successfully loaded. Download the xlsx outputs to the right to see full outputs.",
                            None,
                            "No",
                            None,
                            5,
                        ],
                        [
                            ["example_data/dummy_consultation_response.csv"],
                            "Response text",
                            "Consultation for the construction of flats on Main Street",
                            "dummy_consultation_response.csv",
                            [
                                "example_data/dummy_consultation_r_col_Response_text_Gemma_3_4B_topic_analysis_zero_shot.xlsx"
                            ],
                            dummy_consultation_table_zero_shot,
                            "Example output from the dummy consultation dataset with suggested topics successfully loaded. Download the xlsx outputs to the right to see full outputs.",
                            "example_data/dummy_consultation_response_themes.csv",
                            "No",
                            None,
                            5,
                        ],
                        [
                            ["example_data/combined_case_notes.csv"],
                            "Case Note",
                            "Social Care case notes for young people",
                            "combined_case_notes.csv",
                            [
                                "example_data/combined_case_notes_col_Case_Note_Gemma_3_4B_topic_analysis_grouped.xlsx"
                            ],
                            case_notes_table_grouped,
                            "Example data from the case notes dataset with groups successfully loaded. Download the xlsx outputs to the right to see full outputs.",
                            "example_data/case_note_headers_specific.csv",
                            "No",
                            "Client",
                            5,
                        ],
                        [
                            ["example_data/combined_case_notes.csv"],
                            "Case Note",
                            "Social Care case notes for young people",
                            "combined_case_notes.csv",
                            [
                                "example_data/combined_case_notes_col_Case_Note_Gemma_3_4B_structured_summaries.xlsx"
                            ],
                            case_notes_table_structured_summary,
                            "Example data from the case notes dataset for structured summaries successfully loaded. Download the xlsx outputs to the right to see full outputs.",
                            "example_data/case_note_headers_specific.csv",
                            "Yes",
                            "Client",
                            50,
                        ],
                    ],
                    inputs=[
                        in_data_files,
                        in_colnames,
                        context_textbox,
                        original_data_file_name_textbox,
                        topic_extraction_output_files_xlsx,
                        display_topic_table_markdown,
                        output_messages_textbox,
                        candidate_topics,
                        produce_structured_summary_radio,
                        in_group_col,
                        batch_size_number,
                    ],
                    example_labels=[
                        "Main Street construction consultation",
                        "Case notes for young people",
                        "Main Street construction consultation with suggested topics",
                        "Case notes grouped by person with suggested topics",
                        "Case notes structured summary with suggested topics",
                    ],
                    label="Try topic extraction and summarisation with an example dataset. Example outputs are displayed. Click the 'Extract topics...' button below to rerun the analysis.",
                    fn=show_info_box_on_click,
                    run_on_click=True,
                )
            except (FileNotFoundError, OSError) as e:
                # If example files don't exist (e.g., in CI environment), create empty Examples
                # This allows the app to load without errors
                print(
                    f"Warning: Example files not found, skipping Examples creation: {e}"
                )
                examples = gr.Examples(
                    examples=[],
                    inputs=[
                        in_data_files,
                        in_colnames,
                        context_textbox,
                        original_data_file_name_textbox,
                        topic_extraction_output_files_xlsx,
                        display_topic_table_markdown,
                        output_messages_textbox,
                        candidate_topics,
                        produce_structured_summary_radio,
                        in_group_col,
                        batch_size_number,
                    ],
                    label="Examples not available (example files not found).",
                )
        else:
            # Example files don't exist, create empty Examples
            print(
                "Warning: Required example files not found, skipping Examples creation."
            )
            examples = gr.Examples(
                examples=[],
                inputs=[
                    in_data_files,
                    in_colnames,
                    context_textbox,
                    original_data_file_name_textbox,
                    topic_extraction_output_files_xlsx,
                    display_topic_table_markdown,
                    output_messages_textbox,
                    candidate_topics,
                    produce_structured_summary_radio,
                    in_group_col,
                    batch_size_number,
                ],
                label="Examples not available (example files not found).",
            )

    with gr.Tab(label="All in one topic extraction and summarisation"):
        with gr.Row():
            model_source = gr.Dropdown(
                value=default_model_source,
                choices=model_sources,
                label="Large language model family",
                multiselect=False,
            )
            model_choice = gr.Dropdown(
                value=default_model_choice,
                choices=default_source_models,
                label="Large language model for topic extraction and summarisation",
                multiselect=False,
                allow_custom_value=True,
            )

            model_source.change(
                fn=update_model_choice, inputs=[model_source], outputs=[model_choice]
            )

        with gr.Accordion("Upload xlsx, csv, or parquet file", open=True):
            in_data_files.render()

            in_excel_sheets = gr.Dropdown(
                multiselect=False,
                label="Select the Excel sheet of interest.",
                visible="hidden",
                allow_custom_value=True,
            )
            in_colnames.render()

        with gr.Accordion("Group analysis by values in another column", open=False):
            in_group_col.render()

        with gr.Accordion("Provide list of suggested topics", open=False):
            candidate_topics.render()
            with gr.Row(equal_height=True):
                force_zero_shot_radio = gr.Radio(
                    label="Force responses into suggested topics",
                    value="No",
                    choices=["Yes", "No"],
                )
                force_single_topic_radio = gr.Radio(
                    label="Ask the model to assign responses to only a single topic",
                    value="No",
                    choices=["Yes", "No"],
                )
                produce_structured_summary_radio.render()

        with gr.Accordion("Response sentiment analysis", open=False):
            sentiment_checkbox = gr.Radio(
                label="Response sentiment analysis",
                value="Negative or Positive",
                choices=[
                    "Negative or Positive",
                    "Negative, Neutral, or Positive",
                    "Do not assess sentiment",
                ],
            )

        if GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True":
            with gr.Accordion("Assign task to cost code", open=True, visible=True):
                gr.Markdown(
                    "Please ensure that you have approval from your budget holder before using this app for redaction tasks that incur a cost."
                )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        with gr.Accordion("Cost code table", open=False, visible=True):
                            cost_code_dataframe = gr.Dataframe(
                                value=pd.DataFrame(),
                                label="Existing cost codes",
                                type="pandas",
                                interactive=True,
                                buttons=["fullscreen", "copy"],
                                show_search="filter",
                                visible=True,
                                wrap=True,
                                max_height=200,
                            )
                            reset_cost_code_dataframe_button = gr.Button(
                                value="Reset code code table filter"
                            )
                    with gr.Column():
                        cost_code_choice_drop = gr.Dropdown(
                            value=DEFAULT_COST_CODE,
                            label="Choose cost code for analysis",
                            choices=[DEFAULT_COST_CODE],
                            allow_custom_value=False,
                            visible=True,
                        )

        all_in_one_btn = gr.Button(
            "Extract topics, deduplicate, and summarise", variant="primary"
        )

        with gr.Row(equal_height=True):
            output_messages_textbox.render()

            topic_extraction_output_files_xlsx.render()

        display_topic_table_markdown.render()

        data_feedback_title = gr.Markdown(
            value="## Please give feedback", visible="hidden"
        )
        data_feedback_radio = gr.Radio(
            label="Please give some feedback about the results of the topic extraction.",
            choices=["The results were good", "The results were not good"],
            visible="hidden",
        )
        data_further_details_text = gr.Textbox(
            label="Please give more detailed feedback about the results:",
            visible="hidden",
        )
        data_submit_feedback_btn = gr.Button(value="Submit feedback", visible="hidden")

        with gr.Row():
            s3_logs_output_textbox = gr.Textbox(
                label="Feedback submission logs", visible="hidden"
            )

    with gr.Tab(label="Advanced - Step by step topic extraction and summarisation"):

        with gr.Accordion(
            "1. Extract topics - go to first tab for file upload, model choice, and other settings before clicking this button",
            open=True,
        ):
            context_textbox.render()
            if SHOW_ADDITIONAL_INSTRUCTION_TEXTBOXES == "True":
                additional_summary_instructions_textbox = gr.Textbox(
                    value="", visible=True, label="Additional summary instructions"
                )
            else:
                additional_summary_instructions_textbox = gr.Textbox(
                    value="", visible="hidden", label="Additional summary instructions"
                )

            extract_topics_btn = gr.Button("1. Extract topics", variant="secondary")
            topic_extraction_output_files = gr.File(
                label="Extract topics output files",
                scale=1,
                interactive=True,
                height=FILE_INPUT_HEIGHT,
                file_count="multiple",
            )

        with gr.Accordion(
            "1b. Validate topics - validate previous results with an LLM", open=False
        ):
            if SHOW_ADDITIONAL_INSTRUCTION_TEXTBOXES == "True":
                with gr.Row():
                    show_previous_table_radio = gr.Radio(
                        label="Provide response data to validation process",
                        value="Yes",
                        choices=["Yes", "No"],
                        visible=True,
                        scale=1,
                    )
                    additional_validation_issues_textbox = gr.Textbox(
                        value="",
                        visible=True,
                        label="Additional validation issues for the model to consider (bullet-point list)",
                        scale=3,
                    )
            else:
                with gr.Row():
                    show_previous_table_radio = gr.Radio(
                        label="Provide response data to validation process",
                        value="Yes",
                        choices=["Yes", "No"],
                        visible="hidden",
                        scale=1,
                    )
                    additional_validation_issues_textbox = gr.Textbox(
                        value="",
                        visible="hidden",
                        label="Additional validation issues for the model to consider (bullet-point list)",
                        scale=3,
                    )
            validate_topics_btn = gr.Button("1b. Validate topics", variant="secondary")
            validation_output_files = gr.File(
                label="Validation output files",
                scale=1,
                interactive=False,
                height=FILE_INPUT_HEIGHT,
            )

        with gr.Accordion("2. Modify topics from topic extraction", open=False):
            gr.Markdown(
                """Load in previously completed Extract Topics output files ('reference_table', and 'unique_topics' files) to modify topics, deduplicate topics, or summarise the outputs. If you want pivot table outputs, please load in the original data file along with the selected open text column on the first tab before deduplicating or summarising."""
            )

            modification_input_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Upload reference and unique topic files to modify topics",
                file_count="multiple",
                file_types=[".xlsx", ".xls", ".csv", ".parquet"],
            )

            modifiable_unique_topics_df_state = gr.Dataframe(
                value=pd.DataFrame(),
                headers=None,
                column_count=(4, "fixed"),
                row_count=(1, "fixed"),
                visible=True,
                type="pandas",
            )

            save_modified_files_button = gr.Button(value="Save modified topic names")

        with gr.Accordion(
            "3. Deduplicate topics using fuzzy matching or LLMs", open=False
        ):
            ### DEDUPLICATION
            deduplication_input_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Upload reference and unique topic files to deduplicate topics. Optionally upload suggested topics on the first tab to match to these where possible with LLM deduplication",
                file_count="multiple",
                file_types=[".xlsx", ".xls", ".csv", ".parquet"],
            )
            deduplication_input_files_status = gr.Textbox(
                value="", label="Previous file input", visible="hidden"
            )

            with gr.Row():
                merge_general_topics_drop = gr.Dropdown(
                    label="Merge general topic values together for duplicate subtopics.",
                    value="Yes",
                    choices=["Yes", "No"],
                )
                merge_sentiment_drop = gr.Dropdown(
                    label="Merge sentiment values together for duplicate subtopics.",
                    value="No",
                    choices=["Yes", "No"],
                )
                deduplicate_score_threshold = gr.Number(
                    label="Similarity threshold with which to determine duplicates.",
                    value=90,
                    minimum=5,
                    maximum=100,
                    precision=0,
                )

            with gr.Row():
                deduplicate_previous_data_btn = gr.Button(
                    "3. Deduplicate topics (Fuzzy matching)", variant="primary"
                )
                deduplicate_llm_previous_data_btn = gr.Button(
                    "3b. Deduplicate topics (LLM semantic)", variant="secondary"
                )

        with gr.Accordion("4. Summarise topics", open=False):
            ### SUMMARISATION
            summarisation_input_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Upload reference and unique topic files to summarise",
                file_count="multiple",
                file_types=[".xlsx", ".xls", ".csv", ".parquet"],
            )

            summarise_format_radio = gr.Radio(
                label="Choose summary type (Note: this will also use the custom summary instructions from step 1 above if provided)",
                value=two_para_summary_format_prompt,
                choices=[
                    two_para_summary_format_prompt,
                    single_para_summary_format_prompt,
                ],
            )

            with gr.Row():
                sample_reference_table_checkbox = gr.Checkbox(
                    value=True,
                    label="Sample reference table (recommended for large datasets)",
                )
                no_of_sampled_summaries_number = gr.Number(
                    value=150,
                    label="Number of summaries per group",
                    precision=0,
                    minimum=10,
                    maximum=500,
                )
                random_seed_number = gr.Number(
                    value=42, label="Random seed", precision=0, minimum=1, maximum=9999
                )

            summarise_previous_data_btn = gr.Button(
                "4. Summarise topics", variant="primary"
            )
            with gr.Row():
                summary_output_files = gr.File(
                    height=FILE_INPUT_HEIGHT,
                    label="Summarised output files",
                    interactive=False,
                    scale=3,
                )
                summary_output_files_xlsx = gr.File(
                    height=FILE_INPUT_HEIGHT,
                    label="xlsx file summary",
                    interactive=False,
                    scale=1,
                )

            summarised_output_markdown = gr.Markdown(
                value="### Summarised table will appear here", buttons=["copy"]
            )

        with gr.Accordion("5. Create overall summary", open=False):
            gr.Markdown(
                """### Create an overall summary from an existing topic summary table."""
            )

            ### SUMMARISATION
            overall_summarisation_input_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Upload a '...unique_topic' file to summarise",
                file_count="multiple",
                file_types=[".xlsx", ".xls", ".csv", ".parquet"],
            )

            overall_summarise_format_radio = gr.Radio(
                label="Choose summary type",
                value=two_para_summary_format_prompt,
                choices=[
                    two_para_summary_format_prompt,
                    single_para_summary_format_prompt,
                ],
                visible="hidden",
            )  # This is currently an invisible placeholder in case in future I want to add in overall summarisation customisation

            overall_summarise_previous_data_btn = gr.Button(
                "5. Create overall summary", variant="primary"
            )

            with gr.Row():
                overall_summary_output_files = gr.File(
                    height=FILE_INPUT_HEIGHT,
                    label="Summarised output files",
                    interactive=False,
                    scale=3,
                )
                overall_summary_output_files_xlsx = gr.File(
                    height=FILE_INPUT_HEIGHT,
                    label="xlsx file summary",
                    interactive=False,
                    scale=1,
                )

            overall_summarised_output_markdown = gr.HTML(
                value="### Overall summary will appear here"
            )

    with gr.Tab(label="Topic table viewer", visible="hidden"):
        gr.Markdown("""### View a 'unique_topic_table' csv file in markdown format.""")

        in_view_table = gr.File(
            height=FILE_INPUT_HEIGHT,
            label="Choose unique topic csv files",
            file_count="single",
            file_types=[".csv", ".parquet"],
        )
        view_table_markdown = gr.Markdown(
            value="", label="View table", buttons=["copy"]
        )

    with gr.Tab(label="Continue unfinished topic extraction", visible="hidden"):
        gr.Markdown(
            """### Load in output files from a previous topic extraction process and continue topic extraction with new data."""
        )

        with gr.Accordion(
            "Upload reference data file and unique data files", open=True
        ):
            in_previous_data_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Choose output csv files",
                file_count="multiple",
                file_types=[".csv"],
            )
            in_previous_data_files_status = gr.Textbox(
                value="", label="Previous file input"
            )
            continue_previous_data_files_btn = gr.Button(
                value="Continue previous topic extraction", variant="primary"
            )

    with gr.Tab(label="LLM and topic extraction settings"):
        gr.Markdown("""Define settings that affect large language model output.""")
        with gr.Accordion("Settings for LLM generation", open=True):
            with gr.Row():
                temperature_slide = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=LLM_TEMPERATURE,
                    label="Choose LLM temperature setting",
                    precision=1,
                    step=0.1,
                )
                batch_size_number.render()
            random_seed = gr.Number(
                value=LLM_SEED, label="Random seed for LLM generation", visible="hidden"
            )

        with gr.Accordion("AWS API keys", open=False):
            gr.Markdown(
                """Querying Bedrock models with API keys requires a role with IAM permissions for the bedrock:InvokeModel action."""
            )
            with gr.Row():
                aws_access_key_textbox = gr.Textbox(
                    value=AWS_ACCESS_KEY,
                    label="AWS access key",
                    lines=1,
                    type="password",
                )
                aws_secret_key_textbox = gr.Textbox(
                    value=AWS_SECRET_KEY,
                    label="AWS secret key",
                    lines=1,
                    type="password",
                )

        with gr.Accordion("Gemini API keys", open=False):
            google_api_key_textbox = gr.Textbox(
                value=GEMINI_API_KEY,
                label="Enter Gemini API key (only if using Google API models)",
                lines=1,
                type="password",
            )

        with gr.Accordion("Azure/OpenAI Inference", open=False):
            with gr.Row():
                azure_api_key_textbox = gr.Textbox(
                    value=AZURE_OPENAI_API_KEY,
                    label="Enter Azure/OpenAI Inference API key (only if using Azure/OpenAI models)",
                    lines=1,
                    type="password",
                )
                azure_endpoint_textbox = gr.Textbox(
                    value=AZURE_OPENAI_INFERENCE_ENDPOINT,
                    label="Enter Azure Inference endpoint URL (only if using Azure models)",
                    lines=1,
                )

        with gr.Accordion(
            "Llama-server API", open=False, visible=RUN_INFERENCE_SERVER == "1"
        ):
            api_url_textbox = gr.Textbox(
                value=API_URL,
                label="Enter inference-server API URL (only if using inference-server models)",
                lines=1,
            )

        with gr.Accordion(
            "Hugging Face token for downloading gated models", open=False
        ):
            hf_api_key_textbox = gr.Textbox(
                value=HF_TOKEN,
                label="Enter Hugging Face API key (only for gated models that need a token to download)",
                lines=1,
                type="password",
            )

        with gr.Accordion("Log outputs", open=False):
            log_files_output = gr.File(
                height=FILE_INPUT_HEIGHT, label="Log file output", interactive=False
            )
            conversation_metadata_textbox = gr.Textbox(
                value="",
                label="Query metadata - usage counts and other parameters",
                lines=8,
            )

        with gr.Accordion("Prompt settings", open=False, visible="hidden"):
            number_of_prompts = gr.Number(
                value=1,
                label="Number of prompts to send to LLM in sequence",
                minimum=1,
                maximum=3,
                visible="hidden",
            )
            system_prompt_textbox = gr.Textbox(
                label="Initial system prompt", lines=4, value=system_prompt
            )
            initial_table_prompt_textbox = gr.Textbox(
                label="Initial topics prompt", lines=8, value=initial_table_prompt
            )
            add_to_existing_topics_system_prompt_textbox = gr.Textbox(
                label="Additional topics system prompt",
                lines=4,
                value=add_existing_topics_system_prompt,
            )
            add_to_existing_topics_prompt_textbox = gr.Textbox(
                label="Additional topics prompt",
                lines=8,
                value=add_existing_topics_prompt,
            )

        with gr.Accordion(
            "Join additional columns to reference file outputs", open=False
        ):
            join_colnames = gr.Dropdown(
                choices=["Choose column with responses"],
                multiselect=True,
                label="Select the open text column of interest. In an Excel file, this shows columns across all sheets.",
                allow_custom_value=True,
                interactive=True,
            )
            with gr.Row():
                in_join_files = gr.File(
                    height=FILE_INPUT_HEIGHT,
                    label="Reference file should go here. Original data file should be loaded on the first tab.",
                )
                join_cols_btn = gr.Button(
                    "Join columns to reference output", variant="primary"
                )
            out_join_files = gr.File(
                height=FILE_INPUT_HEIGHT,
                label="Output joined reference files will go here.",
            )

        with gr.Accordion(
            "Export output files to xlsx format", open=False, visible="hidden"
        ):
            export_xlsx_btn = gr.Button(
                "Export output files to xlsx format", variant="primary"
            )
            out_xlsx_files = gr.File(
                height=FILE_INPUT_HEIGHT, label="Output xlsx files will go here."
            )

    ###
    # INTERACTIVE ELEMENT FUNCTIONS
    ###

    ###
    # INITIAL TOPIC EXTRACTION
    ###

    # Tabular data upload
    in_data_files.upload(
        fn=put_columns_in_df,
        inputs=[in_data_files],
        outputs=[
            in_colnames,
            in_excel_sheets,
            original_data_file_name_textbox,
            join_colnames,
            in_group_col,
        ],
    )

    # Click on cost code dataframe/dropdown fills in cost code textbox
    # Allow user to select items from cost code dataframe for cost code

    if SHOW_COSTS == "True" and (
        GET_COST_CODES == "True" or ENFORCE_COST_CODES == "True"
    ):
        cost_code_dataframe.select(
            df_select_callback_cost,
            inputs=[cost_code_dataframe],
            outputs=[cost_code_choice_drop],
        )
        reset_cost_code_dataframe_button.click(
            reset_base_dataframe,
            inputs=[cost_code_dataframe_base],
            outputs=[cost_code_dataframe],
        )

        cost_code_choice_drop.select(
            update_cost_code_dataframe_from_dropdown_select,
            inputs=[cost_code_choice_drop, cost_code_dataframe_base],
            outputs=[cost_code_dataframe],
        )

    # Extract topics
    extract_topics_btn.click(
        fn=empty_output_vars_extract_topics,
        inputs=None,
        outputs=[
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
            file_data_state,
            working_data_file_name_textbox,
            display_topic_table_markdown,
            summary_output_files,
            summarisation_input_files,
            overall_summarisation_input_files,
            overall_summary_output_files,
        ],
    ).success(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        load_in_data_file,
        inputs=[in_data_files, in_colnames, batch_size_number, in_excel_sheets],
        outputs=[
            file_data_state,
            working_data_file_name_textbox,
            total_number_of_batches,
        ],
        api_name="load_data",
    ).success(
        fn=wrapper_extract_topics_per_column_value,
        inputs=[
            in_group_col,
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
            produce_structured_summary_radio,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            hf_api_key_textbox,
            azure_api_key_textbox,
            azure_endpoint_textbox,
            output_folder_state,
            logged_content_df,
            additional_summary_instructions_textbox,
            additional_validation_issues_textbox,
            show_previous_table_radio,
            api_url_textbox,
        ],
        outputs=[
            display_topic_table_markdown,
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
            output_messages_textbox,
            logged_content_df,
        ],
        api_name="extract_topics",
        show_progress_on=[output_messages_textbox, topic_extraction_output_files],
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
        api_name="usage_logs",
    ).then(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_state,
            master_unique_topics_df_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[topic_extraction_output_files_xlsx, summary_xlsx_output_files_list],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )

    # Validate topics
    validate_topics_btn.click(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        load_in_data_file,
        inputs=[in_data_files, in_colnames, batch_size_number, in_excel_sheets],
        outputs=[
            file_data_state,
            working_data_file_name_textbox,
            total_number_of_batches,
        ],
    ).success(
        load_in_previous_data_files,
        inputs=[topic_extraction_output_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed_no_loop,
            deduplication_input_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    ).success(
        fn=validate_topics_wrapper,
        inputs=[
            file_data_state,
            master_reference_df_state,
            master_unique_topics_df_state,
            working_data_file_name_textbox,
            in_colnames,
            batch_size_number,
            model_choice,
            google_api_key_textbox,
            temperature_slide,
            max_tokens_num,
            azure_api_key_textbox,
            azure_endpoint_textbox,
            reasoning_suffix_textbox,
            in_group_col,
            produce_structured_summary_radio,
            force_zero_shot_radio,
            force_single_topic_radio,
            context_textbox,
            additional_summary_instructions_textbox,
            output_folder_state,
            output_debug_files_radio,
            original_data_file_name_textbox,
            additional_validation_issues_textbox,
            max_time_for_loop_num,
            in_data_files,
            sentiment_checkbox,
            logged_content_df,
            show_previous_table_radio,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            api_url_textbox,
        ],
        outputs=[
            display_topic_table_markdown,
            master_topic_df_state,
            master_unique_topics_df_state,
            master_reference_df_state,
            validation_output_files,
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
            output_messages_textbox,
            logged_content_df,
        ],
        api_name="validate_topics",
        show_progress_on=[output_messages_textbox, validation_output_files],
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
        api_name="usage_logs_validation",
    ).then(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_state,
            master_unique_topics_df_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[topic_extraction_output_files_xlsx, summary_xlsx_output_files_list],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )

    ###
    # DEDUPLICATION AND SUMMARISATION FUNCTIONS
    ###
    # If you upload data into the deduplication input box, the modifiable topic dataframe box is updated
    modification_input_files.upload(
        fn=load_in_previous_data_files,
        inputs=[modification_input_files, modified_unique_table_change_bool],
        outputs=[
            modifiable_unique_topics_df_state,
            master_modify_reference_df_state,
            master_modify_unique_topics_df_state,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
            text_output_modify_file_list_state,
        ],
    )

    # Modify output table with custom topic names
    save_modified_files_button.click(
        fn=modify_existing_output_tables,
        inputs=[
            master_modify_unique_topics_df_state,
            modifiable_unique_topics_df_state,
            master_modify_reference_df_state,
            text_output_modify_file_list_state,
            output_folder_state,
        ],
        outputs=[
            master_unique_topics_df_state,
            master_reference_df_state,
            topic_extraction_output_files,
            text_output_file_list_state,
            deduplication_input_files,
            summarisation_input_files,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
            summarised_output_markdown,
        ],
    )

    # When button pressed, deduplicate data
    deduplicate_previous_data_btn.click(
        load_in_previous_data_files,
        inputs=[deduplication_input_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed_no_loop,
            deduplication_input_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    ).success(
        deduplicate_topics,
        inputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
            in_excel_sheets,
            merge_sentiment_drop,
            merge_general_topics_drop,
            deduplicate_score_threshold,
            in_data_files,
            in_colnames,
            output_folder_state,
        ],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            summarisation_input_files,
            log_files_output,
            summarised_output_markdown,
        ],
        scroll_to_output=True,
        api_name="deduplicate_topics",
    )

    # When LLM deduplication button pressed, deduplicate data using LLM
    def deduplicate_topics_llm_wrapper(
        reference_df,
        topic_summary_df,
        reference_table_file_name,
        unique_topics_table_file_name,
        model_choice,
        in_api_key,
        temperature,
        in_excel_sheets,
        merge_sentiment,
        merge_general_topics,
        in_data_files,
        chosen_cols,
        output_folder,
        candidate_topics=None,
        azure_endpoint="",
        api_url=None,
    ):
        # Ensure custom model_choice is registered in model_name_map
        ensure_model_in_map(model_choice)
        model_source = model_name_map[model_choice]["source"]
        return deduplicate_topics_llm(
            reference_df,
            topic_summary_df,
            reference_table_file_name,
            unique_topics_table_file_name,
            model_choice,
            in_api_key,
            temperature,
            model_source,
            None,
            None,
            None,
            None,
            in_excel_sheets,
            merge_sentiment,
            merge_general_topics,
            in_data_files,
            chosen_cols,
            output_folder,
            candidate_topics,
            azure_endpoint,
            api_url,
        )

    deduplicate_llm_previous_data_btn.click(
        load_in_previous_data_files,
        inputs=[deduplication_input_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed_no_loop,
            deduplication_input_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    ).success(
        deduplicate_topics_llm_wrapper,
        inputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
            model_choice,
            google_api_key_textbox,
            temperature_slide,
            in_excel_sheets,
            merge_sentiment_drop,
            merge_general_topics_drop,
            in_data_files,
            in_colnames,
            output_folder_state,
            candidate_topics,
            azure_endpoint_textbox,
            api_url_textbox,
        ],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            summarisation_input_files,
            log_files_output,
            summarised_output_markdown,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
        ],
        scroll_to_output=True,
        api_name="deduplicate_topics_llm",
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
        api_name="usage_logs_llm_dedup",
    )

    # When button pressed, summarise previous data
    summarise_previous_data_btn.click(
        empty_output_vars_summarise,
        inputs=None,
        outputs=[
            summary_reference_table_sample_state,
            master_unique_topics_df_revised_summaries_state,
            master_reference_df_revised_summaries_state,
            summary_output_files,
            summarised_outputs_list,
            latest_summary_completed_num,
            overall_summarisation_input_files,
        ],
    ).success(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        load_in_previous_data_files,
        inputs=[summarisation_input_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed_no_loop,
            deduplication_input_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    ).success(
        wrapper_summarise_output_topics_per_group,
        inputs=[
            in_group_col,
            summary_reference_table_sample_state,
            master_unique_topics_df_state,
            master_reference_df_state,
            model_choice,
            google_api_key_textbox,
            temperature_slide,
            working_data_file_name_textbox,
            summarised_outputs_list,
            latest_summary_completed_num,
            conversation_metadata_textbox,
            in_data_files,
            in_excel_sheets,
            in_colnames,
            log_files_output_list_state,
            summarise_format_radio,
            output_folder_state,
            context_textbox,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            model_name_map_state,
            hf_api_key_textbox,
            azure_endpoint_textbox,
            logged_content_df,
            sample_reference_table_checkbox,
            no_of_sampled_summaries_number,
            random_seed_number,
            api_url_textbox,
        ],
        outputs=[
            summary_reference_table_sample_state,
            master_unique_topics_df_revised_summaries_state,
            master_reference_df_revised_summaries_state,
            summary_output_files,
            summarised_outputs_list,
            latest_summary_completed_num,
            conversation_metadata_textbox,
            summarised_output_markdown,
            log_files_output,
            overall_summarisation_input_files,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            output_messages_textbox,
            logged_content_df,
        ],
        api_name="summarise_topics",
        show_progress_on=[output_messages_textbox, summary_output_files],
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
    ).then(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_revised_summaries_state,
            master_unique_topics_df_revised_summaries_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[summary_output_files_xlsx, summary_xlsx_output_files_list],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )
    # success(sample_reference_table_summaries, inputs=[master_reference_df_state, random_seed, sample_reference_table_checkbox], outputs=[summary_reference_table_sample_state, summarised_references_markdown], api_name="sample_summaries").\

    # SUMMARISE WHOLE TABLE PAGE
    overall_summarise_previous_data_btn.click(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        load_in_previous_data_files,
        inputs=[overall_summarisation_input_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed_no_loop,
            deduplication_input_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    ).success(
        overall_summary,
        inputs=[
            master_unique_topics_df_state,
            model_choice,
            google_api_key_textbox,
            temperature_slide,
            working_data_file_name_textbox,
            output_folder_state,
            in_colnames,
            context_textbox,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            model_name_map_state,
            hf_api_key_textbox,
            azure_endpoint_textbox,
            logged_content_df,
            api_url_textbox,
        ],
        outputs=[
            overall_summary_output_files,
            overall_summarised_output_markdown,
            summarised_output_df,
            conversation_metadata_textbox,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            output_messages_textbox,
            logged_content_df,
        ],
        scroll_to_output=True,
        api_name="overall_summary",
        show_progress_on=[output_messages_textbox, overall_summary_output_files],
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
    ).then(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_state,
            master_unique_topics_df_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[overall_summary_output_files_xlsx, summary_xlsx_output_files_list],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )

    # All in one button
    # Extract topics - deduplicate and summarise using default settings
    all_in_one_btn.click(
        fn=empty_output_vars_extract_topics,
        inputs=None,
        outputs=[
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
            file_data_state,
            working_data_file_name_textbox,
            display_topic_table_markdown,
            summary_output_files,
            summarisation_input_files,
            overall_summarisation_input_files,
            overall_summary_output_files,
        ],
    ).success(
        fn=enforce_cost_codes,
        inputs=[
            enforce_cost_code_textbox,
            cost_code_choice_drop,
            cost_code_dataframe_base,
        ],
    ).success(
        load_in_data_file,
        inputs=[in_data_files, in_colnames, batch_size_number, in_excel_sheets],
        outputs=[
            file_data_state,
            working_data_file_name_textbox,
            total_number_of_batches,
        ],
        api_name="load_data",
    ).success(
        fn=all_in_one_pipeline,
        inputs=[
            in_group_col,
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
            produce_structured_summary_radio,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            hf_api_key_textbox,
            azure_api_key_textbox,
            azure_endpoint_textbox,
            output_folder_state,
            merge_sentiment_drop,
            merge_general_topics_drop,
            deduplicate_score_threshold,
            summarise_format_radio,
            random_seed,
            log_files_output_list_state,
            model_name_map_state,
            usage_logs_state,
            logged_content_df,
            additional_summary_instructions_textbox,
            additional_validation_issues_textbox,
            show_previous_table_radio,
            sample_reference_table_checkbox,
            api_url_textbox,
        ],
        outputs=[
            display_topic_table_markdown,
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
            output_messages_textbox,
            summary_reference_table_sample_state,
            summarised_references_markdown,
            master_unique_topics_df_revised_summaries_state,
            master_reference_df_revised_summaries_state,
            summary_output_files,
            summarised_outputs_list,
            latest_summary_completed_num,
            overall_summarisation_input_files,
            overall_summary_output_files,
            overall_summarised_output_markdown,
            summarised_output_df,
            logged_content_df,
        ],
        show_progress_on=[output_messages_textbox],
        api_name="all_in_one_pipeline",
    ).success(
        lambda *args: usage_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS,
            replacement_headers=CSV_USAGE_LOG_HEADERS,
        ),
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        None,
        preprocess=False,
    ).then(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_revised_summaries_state,
            master_unique_topics_df_revised_summaries_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[overall_summary_output_files_xlsx, summary_xlsx_output_files_list],
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    ).success(
        move_overall_summary_output_files_to_front_page,
        inputs=[summary_xlsx_output_files_list],
        outputs=[topic_extraction_output_files_xlsx],
    )

    ###
    # CONTINUE PREVIOUS TOPIC EXTRACTION PAGE
    ###

    # If uploaded partially completed consultation files do this. This should then start up the 'latest_batch_completed' change action above to continue extracting topics.
    continue_previous_data_files_btn.click(
        load_in_data_file,
        inputs=[in_data_files, in_colnames, batch_size_number, in_excel_sheets],
        outputs=[
            file_data_state,
            working_data_file_name_textbox,
            total_number_of_batches,
        ],
    ).success(
        load_in_previous_data_files,
        inputs=[in_previous_data_files],
        outputs=[
            master_reference_df_state,
            master_unique_topics_df_state,
            latest_batch_completed,
            in_previous_data_files_status,
            working_data_file_name_textbox,
            unique_topics_table_file_name_textbox,
        ],
    )

    ###
    # VIEW TABLE PAGE
    ###

    in_view_table.upload(
        view_table, inputs=[in_view_table], outputs=[view_table_markdown]
    )

    ###
    # LLM SETTINGS PAGE
    ###

    reference_df_data_file_name_textbox = gr.Textbox(
        label="reference_df_data_file_name_textbox", visible="hidden"
    )
    master_reference_df_state_joined = gr.Dataframe(visible="hidden")

    join_cols_btn.click(
        fn=load_in_previous_reference_file,
        inputs=[in_join_files],
        outputs=[master_reference_df_state, reference_df_data_file_name_textbox],
    ).success(
        load_in_data_file,
        inputs=[in_data_files, in_colnames, batch_size_number, in_excel_sheets],
        outputs=[
            file_data_state,
            working_data_file_name_textbox,
            total_number_of_batches,
        ],
    ).success(
        fn=join_cols_onto_reference_df,
        inputs=[
            master_reference_df_state,
            file_data_state,
            join_colnames,
            reference_df_data_file_name_textbox,
        ],
        outputs=[master_reference_df_state_joined, out_join_files],
    )

    # Export to xlsx file
    export_xlsx_btn.click(
        collect_output_csvs_and_create_excel_output,
        inputs=[
            in_data_files,
            in_colnames,
            original_data_file_name_textbox,
            in_group_col,
            model_choice,
            master_reference_df_state,
            master_unique_topics_df_state,
            summarised_output_df,
            missing_df_state,
            in_excel_sheets,
            usage_logs_state,
            model_name_map_state,
            output_folder_state,
            produce_structured_summary_radio,
        ],
        outputs=[out_xlsx_files, summary_xlsx_output_files_list],
        api_name="export_xlsx",
    ).success(
        fn=export_outputs_to_s3,
        inputs=[
            summary_xlsx_output_files_list,
            s3_output_folder_state,
            save_outputs_to_s3_checkbox,
            in_data_files,
        ],
        outputs=None,
    )

    # If relevant environment variable is set, load in the default cost code file from S3 or locally
    if GET_COST_CODES == "True" and (COST_CODES_PATH or S3_COST_CODES_PATH):
        if (
            not os.path.exists(COST_CODES_PATH)
            and S3_COST_CODES_PATH
            and RUN_AWS_FUNCTIONS == "1"
        ):
            print("Downloading cost codes from S3")
            print(
                f"Attempting to download from bucket: {S3_LOG_BUCKET}, key: {S3_COST_CODES_PATH}"
            )

            # Create a wrapper function with error handling
            def download_cost_codes_with_error_handling(bucket, key, local_path):
                try:
                    download_file_from_s3(bucket, key, local_path)
                    return True
                except Exception as e:
                    print(f"Error downloading cost codes from S3: {e}")
                    print(f"Failed to download s3://{bucket}/{key}")
                    return False

            app.load(
                download_cost_codes_with_error_handling,
                inputs=[
                    s3_default_bucket,
                    s3_default_cost_codes_file,
                    default_cost_codes_output_folder_location,
                ],
            ).success(
                load_in_default_cost_codes,
                inputs=[
                    default_cost_codes_output_folder_location,
                    default_cost_code_textbox,
                ],
                outputs=[
                    cost_code_dataframe,
                    cost_code_dataframe_base,
                    cost_code_choice_drop,
                ],
            )
            print("Successfully loaded cost codes from S3")
        elif os.path.exists(COST_CODES_PATH):
            print(
                "Loading cost codes from default cost codes path location:",
                COST_CODES_PATH,
            )
            app.load(
                load_in_default_cost_codes,
                inputs=[
                    default_cost_codes_output_folder_location,
                    default_cost_code_textbox,
                ],
                outputs=[
                    cost_code_dataframe,
                    cost_code_dataframe_base,
                    cost_code_choice_drop,
                ],
            )
        else:
            print("Could not load in cost code data")

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###

    # Get connection parameters
    app.load(
        get_connection_params,
        inputs=None,
        outputs=[
            session_hash_state,
            output_folder_state,
            session_hash_textbox,
            input_folder_state,
        ],
    )

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = CSVLogger_custom(dataset_file_name=LOG_FILE_NAME)
    access_callback.setup([session_hash_textbox], ACCESS_LOGS_FOLDER)

    session_hash_textbox.change(
        lambda *args: access_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_ACCESS_LOG_HEADERS,
            replacement_headers=CSV_ACCESS_LOG_HEADERS,
        ),
        [session_hash_textbox],
        None,
        preprocess=False,
    ).success(
        fn=upload_file_to_s3,
        inputs=[
            access_logs_state,
            access_s3_logs_loc_state,
            s3_log_bucket_name,
            aws_access_key_textbox,
            aws_secret_key_textbox,
        ],
        outputs=[s3_logs_output_textbox],
    )

    # Log usage when making a query
    usage_callback = CSVLogger_custom(dataset_file_name=USAGE_LOG_FILE_NAME)
    usage_callback.setup(
        [
            session_hash_textbox,
            original_data_file_name_textbox,
            in_colnames,
            model_choice,
            conversation_metadata_textbox_placeholder,
            input_tokens_num,
            output_tokens_num,
            number_of_calls_num,
            estimated_time_taken_number,
            cost_code_choice_drop,
        ],
        USAGE_LOGS_FOLDER,
    )

    # See extract topics and summarise calls to see the calls to usage logs

    # number_of_calls_num.change(lambda *args: usage_callback.flag(list(args), save_to_csv=SAVE_LOGS_TO_CSV, save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,  dynamodb_table_name=USAGE_LOG_DYNAMODB_TABLE_NAME, dynamodb_headers=DYNAMODB_USAGE_LOG_HEADERS, replacement_headers=CSV_USAGE_LOG_HEADERS), [session_hash_textbox, original_data_file_name_textbox, in_colnames, model_choice, conversation_metadata_textbox, input_tokens_num, output_tokens_num, number_of_calls_num, estimated_time_taken_number, cost_code_choice_drop], None, preprocess=False, api_name="usage_logs").\
    #     success(fn = upload_file_to_s3, inputs=[usage_logs_state, usage_s3_logs_loc_state, s3_log_bucket_name, aws_access_key_textbox, aws_secret_key_textbox], outputs=[s3_logs_output_textbox])

    number_of_calls_num.change(
        fn=upload_file_to_s3,
        inputs=[
            usage_logs_state,
            usage_s3_logs_loc_state,
            s3_log_bucket_name,
            aws_access_key_textbox,
            aws_secret_key_textbox,
        ],
        outputs=[s3_logs_output_textbox],
    )

    # User submitted feedback
    feedback_callback = CSVLogger_custom(dataset_file_name=FEEDBACK_LOG_FILE_NAME)
    feedback_callback.setup(
        [
            data_feedback_radio,
            data_further_details_text,
            original_data_file_name_textbox,
            model_choice,
            temperature_slide,
            display_topic_table_markdown,
            conversation_metadata_textbox,
        ],
        FEEDBACK_LOGS_FOLDER,
    )

    data_submit_feedback_btn.click(
        lambda *args: feedback_callback.flag(
            list(args),
            save_to_csv=SAVE_LOGS_TO_CSV,
            save_to_dynamodb=SAVE_LOGS_TO_DYNAMODB,
            dynamodb_table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
            dynamodb_headers=DYNAMODB_FEEDBACK_LOG_HEADERS,
            replacement_headers=CSV_FEEDBACK_LOG_HEADERS,
        ),
        [
            data_feedback_radio,
            data_further_details_text,
            original_data_file_name_textbox,
            model_choice,
            temperature_slide,
            display_topic_table_markdown,
            conversation_metadata_textbox,
        ],
        None,
        preprocess=False,
    ).success(
        fn=upload_file_to_s3,
        inputs=[
            feedback_logs_state,
            feedback_s3_logs_loc_state,
            s3_log_bucket_name,
            aws_access_key_textbox,
            aws_secret_key_textbox,
        ],
        outputs=[data_further_details_text],
    )

###
# APP RUN
###

if __name__ == "__main__":
    if RUN_DIRECT_MODE == "1":
        from cli_topics import main

        # Validate required direct mode configuration
        if not DIRECT_MODE_INPUT_FILE and DIRECT_MODE_TASK in [
            "extract",
            "validate",
            "all_in_one",
        ]:
            print(
                "Error: DIRECT_MODE_INPUT_FILE environment variable must be set for direct mode."
            )
            print("Please set DIRECT_MODE_INPUT_FILE to the path of your input file.")
            exit(1)

        if (
            DIRECT_MODE_TASK in ["extract", "validate", "all_in_one"]
            and not DIRECT_MODE_TEXT_COLUMN
        ):
            print(
                "Error: DIRECT_MODE_TEXT_COLUMN environment variable must be set for direct mode tasks: extract, validate, all_in_one."
            )
            print(
                "Please set DIRECT_MODE_TEXT_COLUMN to the name of the text column to process."
            )
            exit(1)

        if (
            DIRECT_MODE_TASK
            in ["validate", "deduplicate", "summarise", "overall_summary"]
            and not DIRECT_MODE_PREVIOUS_OUTPUT_FILES
        ):
            print(
                "Error: DIRECT_MODE_PREVIOUS_OUTPUT_FILES environment variable must be set for direct mode tasks: validate, deduplicate, summarise, overall_summary."
            )
            print(
                "Please set DIRECT_MODE_PREVIOUS_OUTPUT_FILES to a pipe-separated (|) list of previous output file paths."
            )
            exit(1)

        # Parse previous_output_files if provided (pipe-separated string to handle paths with spaces)
        previous_output_files_list = []
        if DIRECT_MODE_PREVIOUS_OUTPUT_FILES:
            # Use pipe separator to handle file paths with spaces
            previous_output_files_list = [
                f.strip()
                for f in DIRECT_MODE_PREVIOUS_OUTPUT_FILES.split("|")
                if f.strip()
            ]

        # Parse excel_sheets if provided (comma-separated string)
        excel_sheets_list = []
        if DIRECT_MODE_EXCEL_SHEETS:
            excel_sheets_list = [
                s.strip() for s in DIRECT_MODE_EXCEL_SHEETS.split(",") if s.strip()
            ]

        # Parse input_file if provided (pipe-separated string for multiple files to handle paths with spaces)
        input_file_list = []
        if DIRECT_MODE_INPUT_FILE:
            # Use pipe separator to handle file paths with spaces
            # First check if it's a single file (no pipe), then split if multiple files
            if "|" in DIRECT_MODE_INPUT_FILE:
                input_file_list = [
                    f.strip() for f in DIRECT_MODE_INPUT_FILE.split("|") if f.strip()
                ]
            else:
                # Single file - use as-is to preserve paths with spaces
                input_file_list = [DIRECT_MODE_INPUT_FILE.strip()]

        # Prepare direct mode arguments based on environment variables
        direct_mode_args = {
            # Task Selection
            "task": DIRECT_MODE_TASK,
            # General Arguments
            "input_file": input_file_list if input_file_list else None,
            "output_dir": DIRECT_MODE_OUTPUT_DIR,
            "input_dir": INPUT_FOLDER,
            "text_column": DIRECT_MODE_TEXT_COLUMN if DIRECT_MODE_TEXT_COLUMN else None,
            "previous_output_files": (
                previous_output_files_list if previous_output_files_list else None
            ),
            "username": DIRECT_MODE_USERNAME,
            "save_to_user_folders": SESSION_OUTPUT_FOLDER,
            "excel_sheets": excel_sheets_list,
            "group_by": DIRECT_MODE_GROUP_BY if DIRECT_MODE_GROUP_BY else None,
            # Model Configuration
            "model_choice": DIRECT_MODE_MODEL_CHOICE,
            "model_source": default_model_source,
            "temperature": float(DIRECT_MODE_TEMPERATURE),
            "batch_size": int(DIRECT_MODE_BATCH_SIZE),
            "max_tokens": int(DIRECT_MODE_MAX_TOKENS),
            "google_api_key": GEMINI_API_KEY,
            "aws_access_key": AWS_ACCESS_KEY,
            "aws_secret_key": AWS_SECRET_KEY,
            "aws_region": AWS_REGION,
            "hf_token": HF_TOKEN,
            "azure_api_key": AZURE_OPENAI_API_KEY,
            "azure_endpoint": AZURE_OPENAI_INFERENCE_ENDPOINT,
            "api_url": API_URL,
            "inference_server_model": (
                DIRECT_MODE_INFERENCE_SERVER_MODEL
                if DIRECT_MODE_INFERENCE_SERVER_MODEL
                else None
            ),
            # Topic Extraction Arguments
            "context": DIRECT_MODE_CONTEXT if DIRECT_MODE_CONTEXT else "",
            "candidate_topics": (
                DIRECT_MODE_CANDIDATE_TOPICS if DIRECT_MODE_CANDIDATE_TOPICS else None
            ),
            "force_zero_shot": DIRECT_MODE_FORCE_ZERO_SHOT,
            "force_single_topic": DIRECT_MODE_FORCE_SINGLE_TOPIC,
            "produce_structured_summary": DIRECT_MODE_PRODUCE_STRUCTURED_SUMMARY,
            "sentiment": DIRECT_MODE_SENTIMENT,
            "additional_summary_instructions": (
                DIRECT_MODE_ADDITIONAL_SUMMARY_INSTRUCTIONS
                if DIRECT_MODE_ADDITIONAL_SUMMARY_INSTRUCTIONS
                else ""
            ),
            # Validation Arguments
            "additional_validation_issues": (
                DIRECT_MODE_ADDITIONAL_VALIDATION_ISSUES
                if DIRECT_MODE_ADDITIONAL_VALIDATION_ISSUES
                else ""
            ),
            "show_previous_table": DIRECT_MODE_SHOW_PREVIOUS_TABLE,
            "output_debug_files": OUTPUT_DEBUG_FILES,
            "max_time_for_loop": int(DIRECT_MODE_MAX_TIME_FOR_LOOP),
            # Deduplication Arguments
            "method": DIRECT_MODE_DEDUP_METHOD,
            "similarity_threshold": int(DIRECT_MODE_SIMILARITY_THRESHOLD),
            "merge_sentiment": DIRECT_MODE_MERGE_SENTIMENT,
            "merge_general_topics": DIRECT_MODE_MERGE_GENERAL_TOPICS,
            # Summarisation Arguments
            "summary_format": DIRECT_MODE_SUMMARY_FORMAT,
            "sample_reference_table": DIRECT_MODE_SAMPLE_REFERENCE_TABLE,
            "no_of_sampled_summaries": int(DIRECT_MODE_NO_OF_SAMPLED_SUMMARIES),
            "random_seed": int(DIRECT_MODE_RANDOM_SEED),
            # Output Format Arguments
            "create_xlsx_output": DIRECT_MODE_CREATE_XLSX_OUTPUT == "True",
            # Logging Arguments
            "save_logs_to_csv": SAVE_LOGS_TO_CSV,
            "save_logs_to_dynamodb": SAVE_LOGS_TO_DYNAMODB,
            "usage_logs_folder": USAGE_LOGS_FOLDER,
            "cost_code": DEFAULT_COST_CODE,
        }

        print(f"Running in direct mode with task: {DIRECT_MODE_TASK}")
        if input_file_list:
            print(f"Input file(s): {', '.join(input_file_list)}")
        print(f"Output directory: {DIRECT_MODE_OUTPUT_DIR}")
        if DIRECT_MODE_TEXT_COLUMN:
            print(f"Text column: {DIRECT_MODE_TEXT_COLUMN}")
        if previous_output_files_list:
            print(f"Previous output files: {', '.join(previous_output_files_list)}")
        if DIRECT_MODE_GROUP_BY:
            print(f"Group by: {DIRECT_MODE_GROUP_BY}")

        # Run the CLI main function with direct mode arguments
        main(direct_mode_args=direct_mode_args)
    else:
        if COGNITO_AUTH == "1":
            app.queue(max_size=MAX_QUEUE_SIZE).launch(
                show_error=True,
                inbrowser=True,
                auth=authenticate_user,
                max_file_size=MAX_FILE_SIZE,
                server_port=GRADIO_SERVER_PORT,
                root_path=ROOT_PATH,
                mcp_server=RUN_MCP_SERVER,
                theme=gr.themes.Default(primary_hue="blue"),
                css=css,
            )
        else:
            app.queue(max_size=MAX_QUEUE_SIZE).launch(
                show_error=True,
                inbrowser=True,
                max_file_size=MAX_FILE_SIZE,
                server_port=GRADIO_SERVER_PORT,
                root_path=ROOT_PATH,
                mcp_server=RUN_MCP_SERVER,
                theme=gr.themes.Default(primary_hue="blue"),
                css=css,
            )
