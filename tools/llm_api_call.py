import json
import os
import re
import string
import time
from io import StringIO
from typing import Any, List, Tuple

import gradio as gr
import markdown
import pandas as pd
import spaces
from gradio import Progress
from tqdm import tqdm

from tools.aws_functions import connect_to_bedrock_runtime
from tools.config import (
    BATCH_SIZE_DEFAULT,
    CHOSEN_LOCAL_MODEL_TYPE,
    DEDUPLICATION_THRESHOLD,
    ENABLE_VALIDATION,
    LLM_CONTEXT_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LLM_SEED,
    MAX_COMMENT_CHARS,
    MAX_GROUPS,
    MAX_OUTPUT_VALIDATION_ATTEMPTS,
    MAX_ROWS,
    MAX_SPACES_GPU_RUN_TIME,
    MAX_TIME_FOR_LOOP,
    MAXIMUM_ZERO_SHOT_TOPICS,
    NUMBER_OF_RETRY_ATTEMPTS,
    OUTPUT_DEBUG_FILES,
    OUTPUT_FOLDER,
    REASONING_SUFFIX,
    RUN_LOCAL_MODEL,
    TIMEOUT_WAIT,
    model_name_map,
)
from tools.dedup_summaries import (
    deduplicate_topics,
    overall_summary,
    process_debug_output_iteration,
    wrapper_summarise_output_topics_per_group,
)
from tools.helper_functions import (
    clean_column_name,
    convert_reference_table_to_pivot_table,
    create_topic_summary_df_from_reference_table,
    ensure_model_in_map,
    generate_zero_shot_topics_df,
    get_basic_response_data,
    load_in_data_file,
    load_in_previous_data_files,
    put_columns_in_df,
    read_file,
    wrap_text,
)
from tools.llm_funcs import (
    calculate_tokens_from_metadata,
    call_llm_with_markdown_table_checks,
    construct_azure_client,
    construct_gemini_generative_model,
    create_missing_references_df,
    get_assistant_model,
    get_model,
    get_tokenizer,
)
from tools.prompts import (
    add_existing_topics_assistant_prefill,
    add_existing_topics_prompt,
    add_existing_topics_system_prompt,
    allow_new_topics_prompt,
    default_response_reference_format,
    default_sentiment_prompt,
    force_existing_topics_prompt,
    force_single_topic_prompt,
    initial_table_assistant_prefill,
    initial_table_prompt,
    initial_table_system_prompt,
    negative_neutral_positive_sentiment_prompt,
    negative_or_positive_sentiment_prompt,
    previous_table_introduction_default,
    structured_summary_prompt,
    validation_prompt_prefix_default,
    validation_prompt_suffix_default,
    validation_prompt_suffix_struct_summary_default,
    validation_system_prompt,
)

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS
random_seed = LLM_SEED
reasoning_suffix = REASONING_SUFFIX
max_rows = MAX_ROWS
maximum_zero_shot_topics = MAXIMUM_ZERO_SHOT_TOPICS
output_debug_files = OUTPUT_DEBUG_FILES

max_text_length = 500

### HELPER FUNCTIONS


def normalise_string(text: str):
    # Replace two or more dashes with a single dash
    text = re.sub(r"-{2,}", "-", text)

    # Replace two or more spaces with a single space
    text = re.sub(r"\s{2,}", " ", text)

    # Replace multiple newlines with a single newline.
    text = re.sub(r"\n{2,}|\r{2,}", "\n", text)

    return text


def reconstruct_markdown_table_from_reference_df(
    reference_df: pd.DataFrame, start_row: int = None, end_row: int = None
) -> tuple[str, pd.DataFrame]:
    """
    Reconstructs a markdown table from reference_df data when all_responses_content is missing.
    Filters to only include rows from the current batch if start_row and end_row are provided.

    Parameters:
    - reference_df (pd.DataFrame): The reference dataframe containing topic analysis data
    - start_row (int, optional): The starting row number for the current batch
    - end_row (int, optional): The ending row number for the current batch

    Returns:
    - tuple[str, pd.DataFrame]: A tuple containing:
        - str: A markdown table string in the required format
        - pd.DataFrame: A pandas DataFrame with the same data as the markdown table
    """
    if reference_df.empty:
        return "", pd.DataFrame()

    # Filter reference_df to current batch if start_row and end_row are provided
    filtered_df = reference_df.copy()
    if start_row is not None and end_row is not None:
        # Convert Response References to numeric for filtering
        filtered_df["Response References"] = pd.to_numeric(
            filtered_df["Response References"], errors="coerce"
        )
        # Filter to only include rows where Response References fall within the current batch range
        filtered_df = filtered_df[
            (filtered_df["Response References"] >= start_row + 1)
            & (filtered_df["Response References"] <= end_row + 1)
        ]

        if filtered_df.empty:
            return "", pd.DataFrame()

    if (
        "Revised summary" in filtered_df.columns
        and "Summary" not in filtered_df.columns
    ):
        filtered_df = filtered_df.rename(columns={"Revised summary": "Summary"})

    # Group by General topic, Subtopic, and Sentiment to aggregate response references
    grouped_df = (
        filtered_df.groupby(["General topic", "Subtopic", "Sentiment"])
        .agg(
            {
                "Response References": lambda x: ", ".join(
                    map(str, sorted(x.unique()))
                ),
                "Summary": "first",  # Take the first summary for each group
            }
        )
        .reset_index()
    )

    # Adjust response references to be relative to the batch (subtract start_row if provided)
    if start_row is not None:
        # Convert response references to relative numbers by subtracting start_row
        def adjust_references(refs_str):
            if not refs_str or refs_str == "":
                return refs_str
            try:
                # Split by comma, convert to int, subtract start_row, convert back to string
                refs = [
                    str(int(ref.strip()) - start_row)
                    for ref in refs_str.split(",")
                    if ref.strip().isdigit()
                ]
                return ", ".join(refs)
            except (ValueError, TypeError):
                return refs_str

        grouped_df["Response References"] = grouped_df["Response References"].apply(
            adjust_references
        )

    # Clean up the data to handle any NaN values and remove "Rows x to y: " prefix from summary
    cleaned_df = grouped_df.copy()
    for col in [
        "General topic",
        "Subtopic",
        "Sentiment",
        "Response References",
        "Summary",
    ]:
        cleaned_df[col] = cleaned_df[col].fillna("").astype(str)

    # Remove "Rows x to y: " prefix from summary if present
    cleaned_df["Summary"] = cleaned_df["Summary"].apply(
        lambda x: (
            re.sub(r"^Rows\s+\d+\s+to\s+\d+:\s*", "", x) if isinstance(x, str) else x
        )
    )

    cleaned_df.drop_duplicates(
        ["General topic", "Subtopic", "Sentiment", "Response References"], inplace=True
    )

    # Create the markdown table
    markdown_table = (
        "| General topic | Subtopic | Sentiment | Response References | Summary |\n"
    )
    markdown_table += "|---|---|---|---|---|\n"

    for _, row in cleaned_df.iterrows():
        general_topic = row["General topic"]
        subtopic = row["Subtopic"]
        sentiment = row["Sentiment"]
        response_refs = row["Response References"]
        summary = row["Summary"]

        # Add row to markdown table
        markdown_table += f"| {general_topic} | {subtopic} | {sentiment} | {response_refs} | {summary} |\n"

    return markdown_table, cleaned_df


def validate_topics(
    file_data: pd.DataFrame,
    reference_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    file_name: str,
    chosen_cols: List[str],
    batch_size: int,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    max_tokens: int,
    azure_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    reasoning_suffix: str = "",
    group_name: str = "All",
    produce_structured_summary_radio: str = "No",
    force_zero_shot_radio: str = "No",
    force_single_topic_radio: str = "No",
    context_textbox: str = "",
    additional_instructions_summary_format: str = "",
    output_folder: str = OUTPUT_FOLDER,
    output_debug_files: str = "False",
    original_full_file_name: str = "",
    additional_validation_issues_provided: str = "",
    max_time_for_loop: int = MAX_TIME_FOR_LOOP,
    sentiment_checkbox: str = "Negative or Positive",
    logged_content: list = None,
    show_previous_table: str = "Yes",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    api_url: str = None,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, pd.DataFrame, list, str, int, int, int]:
    """
    Validates topics by re-running the topic extraction process on all batches
    using the consolidated topics from the original run.

    Parameters:
    - file_data (pd.DataFrame): The input data to validate
    - reference_df (pd.DataFrame): The reference dataframe from the original run
    - topic_summary_df (pd.DataFrame): The topic summary dataframe from the original run
    - file_name (str): Name of the file being processed
    - chosen_cols (List[str]): Columns to process
    - batch_size (int): Size of each batch
    - model_choice (str): The model to use for validation
    - in_api_key (str): API key for the model
    - temperature (float): Temperature for the model
    - max_tokens (int): Maximum tokens for the model
    - azure_api_key_textbox (str): Azure API key if using Azure
    - azure_endpoint_textbox (str): Azure endpoint if using Azure
    - reasoning_suffix (str): Suffix for reasoning
    - group_name (str): Name of the group
    - produce_structured_summary_radio (str): Whether to produce structured summaries
    - force_zero_shot_radio (str): Whether to force zero-shot
    - force_single_topic_radio (str): Whether to force single topic
    - context_textbox (str): Context for the validation
    - additional_instructions_summary_format (str): Additional instructions
    - output_folder (str): Output folder for files
    - output_debug_files (str): Whether to output debug files
    - original_full_file_name (str): Original file name
    - additional_validation_issues_provided (str): Additional validation issues provided
    - max_time_for_loop (int): Maximum time for the loop
    - logged_content (list, optional): The logged content from the original run. If None, tables will be reconstructed from reference_df
    - show_previous_table (str): Whether to show the previous table ("Yes" or "No").
    - aws_access_key_textbox (str): AWS access key.
    - aws_secret_key_textbox (str): AWS secret key.
    - progress: Progress bar object

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, list, str, int, int, int]: Updated reference_df, topic_summary_df, logged_content, conversation_metadata_str, total_input_tokens, total_output_tokens, total_llm_calls
    """
    print("Starting validation process...")

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice)

    # Calculate number of batches
    num_batches = (len(file_data) + batch_size - 1) // batch_size

    # Initialize model components
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_source = model_name_map[model_choice]["source"]

    if context_textbox and "The context of this analysis is" not in context_textbox:
        context_textbox = "The context of this analysis is '" + context_textbox + "'."

    # Initialize model objects
    local_model = None
    tokenizer = None
    bedrock_runtime = None

    # Load local model if needed
    if (model_name_map[model_choice]["source"] == "Local") & (RUN_LOCAL_MODEL == "1"):
        local_model = get_model()
        tokenizer = get_tokenizer()

    # Set up bedrock runtime if needed
    if model_source == "AWS":
        bedrock_runtime = connect_to_bedrock_runtime(
            model_name_map, model_choice, aws_access_key_textbox, aws_secret_key_textbox
        )

    # Clean file name for output
    file_name_clean = clean_column_name(
        file_name, max_length=20, front_characters=False
    )
    in_column_cleaned = clean_column_name(chosen_cols, max_length=20)
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    # Create validation-specific logged content lists
    validation_all_prompts_content = list()
    validation_all_summaries_content = list()
    validation_all_conversation_content = list()
    validation_all_metadata_content = list()
    validation_all_groups_content = list()
    validation_all_batches_content = list()
    validation_all_model_choice_content = list()
    validation_all_validated_content = list()
    validation_all_task_type_content = list()
    validation_all_file_names_content = list()

    # Extract previous summaries from logged content for validation
    if logged_content is None:
        logged_content = list()
    all_responses_content = [
        item.get("response", "") for item in logged_content if "response" in item
    ]

    # Initialize validation dataframes
    validation_reference_df = reference_df.copy()
    validation_topic_summary_df = topic_summary_df.copy()

    sentiment_prefix = "In the next column named 'Sentiment', "
    sentiment_suffix = "."
    if sentiment_checkbox == "Negative, Neutral, or Positive":
        sentiment_prompt = (
            sentiment_prefix
            + negative_neutral_positive_sentiment_prompt
            + sentiment_suffix
        )
    elif sentiment_checkbox == "Negative or Positive":
        sentiment_prompt = (
            sentiment_prefix + negative_or_positive_sentiment_prompt + sentiment_suffix
        )
    elif sentiment_checkbox == "Do not assess sentiment":
        sentiment_prompt = ""  # Just remove line completely. Previous: sentiment_prefix + do_not_assess_sentiment_prompt + sentiment_suffix
    else:
        sentiment_prompt = (
            sentiment_prefix + default_sentiment_prompt + sentiment_suffix
        )

    # Validation loop through all batches
    validation_latest_batch_completed = 0
    validation_loop = progress.tqdm(
        range(num_batches),
        total=num_batches,
        desc="Validating topic extraction batches",
        unit="validation batches",
    )

    tic = time.perf_counter()

    for validation_i in validation_loop:
        validation_reported_batch_no = validation_latest_batch_completed + 1
        print("Running validation batch:", validation_reported_batch_no)

        # Call the function to prepare the input table for validation
        (
            validation_simplified_csv_table_path,
            validation_normalised_simple_markdown_table,
            validation_start_row,
            validation_end_row,
            validation_batch_basic_response_df,
        ) = data_file_to_markdown_table(
            file_data,
            file_name,
            chosen_cols,
            validation_latest_batch_completed,
            batch_size,
        )

        if validation_batch_basic_response_df.shape[0] == 1:
            validation_response_reference_format = ""
        else:
            validation_response_reference_format = (
                "\n" + default_response_reference_format
            )

        if validation_normalised_simple_markdown_table:
            validation_response_table_prompt = (
                "Response table:\n" + validation_normalised_simple_markdown_table
            )
        else:
            validation_response_table_prompt = ""

        # If the validation batch of responses contains at least one instance of text. The function will first try to get the previous table from logged outputs, and will reconstruct the table from reference_df data if not available.
        if not reference_df.empty:
            validation_latest_batch_completed = int(validation_latest_batch_completed)
            validation_start_row = int(validation_start_row)
            validation_end_row = int(validation_end_row)

            # Get the previous table from all_responses_content for this batch
            if validation_latest_batch_completed < len(all_responses_content):
                previous_table_content = all_responses_content[
                    validation_latest_batch_completed
                ]
                _, previous_topic_df = reconstruct_markdown_table_from_reference_df(
                    reference_df, validation_start_row, validation_end_row
                )
            else:
                # Try to reconstruct markdown table from reference_df data
                previous_table_content, previous_topic_df = (
                    reconstruct_markdown_table_from_reference_df(
                        reference_df, validation_start_row, validation_end_row
                    )
                )

            # Always use the consolidated topics from the first run for validation
            validation_formatted_system_prompt = validation_system_prompt.format(
                consultation_context=context_textbox, column_name=chosen_cols
            )

            # Use the accumulated topic summary from previous validation batches (or initial if first batch)
            validation_existing_topic_summary_df = validation_topic_summary_df.copy()
            validation_existing_topic_summary_df["Number of responses"] = ""
            validation_existing_topic_summary_df.fillna("", inplace=True)
            validation_existing_topic_summary_df["General topic"] = (
                validation_existing_topic_summary_df["General topic"].str.replace(
                    "(?i)^Nan$", "", regex=True
                )
            )
            validation_existing_topic_summary_df["Subtopic"] = (
                validation_existing_topic_summary_df["Subtopic"].str.replace(
                    "(?i)^Nan$", "", regex=True
                )
            )
            validation_existing_topic_summary_df = (
                validation_existing_topic_summary_df.drop_duplicates()
            )

            # Create topics table to be presented to LLM for validation
            validation_keep_cols = [
                col
                for col in ["General topic", "Subtopic", "Description"]
                if col in validation_existing_topic_summary_df.columns
                and not validation_existing_topic_summary_df[col]
                .replace(r"^\s*$", pd.NA, regex=True)
                .isna()
                .all()
            ]

            validation_topics_df_for_markdown = validation_existing_topic_summary_df[
                validation_keep_cols
            ].drop_duplicates(validation_keep_cols)
            if (
                "General topic" in validation_topics_df_for_markdown.columns
                and "Subtopic" in validation_topics_df_for_markdown.columns
            ):
                validation_topics_df_for_markdown = (
                    validation_topics_df_for_markdown.sort_values(
                        ["General topic", "Subtopic"]
                    )
                )

            if "Description" in validation_existing_topic_summary_df:
                if validation_existing_topic_summary_df["Description"].isnull().all():
                    validation_existing_topic_summary_df.drop(
                        "Description", axis=1, inplace=True
                    )

            if produce_structured_summary_radio == "Yes":
                if "General topic" in validation_topics_df_for_markdown.columns:
                    validation_topics_df_for_markdown = (
                        validation_topics_df_for_markdown.rename(
                            columns={"General topic": "Main heading"}
                        )
                    )
                if "Subtopic" in validation_topics_df_for_markdown.columns:
                    validation_topics_df_for_markdown = (
                        validation_topics_df_for_markdown.rename(
                            columns={"Subtopic": "Subheading"}
                        )
                    )

            validation_unique_topics_markdown = (
                validation_topics_df_for_markdown.to_markdown(index=False)
            )
            validation_unique_topics_markdown = normalise_string(
                validation_unique_topics_markdown
            )

            if force_zero_shot_radio == "Yes":
                validation_topic_assignment_prompt = force_existing_topics_prompt
            else:
                validation_topic_assignment_prompt = allow_new_topics_prompt

            # Should the outputs force only one single topic assignment per response?
            if force_single_topic_radio != "Yes":
                validation_force_single_topic_prompt = ""
            else:
                validation_topic_assignment_prompt = (
                    validation_topic_assignment_prompt.replace(
                        "Assign topics", "Assign a topic"
                    )
                    .replace("assign Subtopics", "assign a Subtopic")
                    .replace("Subtopics", "Subtopic")
                    .replace("Topics", "Topic")
                    .replace("topics", "a topic")
                )

            # Provide new validation issues on a new line if provided
            # if additional_validation_issues_provided:
            #    additional_validation_issues_provided = "\n" + additional_validation_issues_provided

            # Format the validation prompt with the response table and topics
            if produce_structured_summary_radio != "Yes":
                validation_formatted_summary_prompt = add_existing_topics_prompt.format(
                    validate_prompt_prefix=validation_prompt_prefix_default,
                    response_table=validation_response_table_prompt,
                    topics=validation_unique_topics_markdown,
                    topic_assignment=validation_topic_assignment_prompt,
                    force_single_topic=validation_force_single_topic_prompt,
                    sentiment_choices=sentiment_prompt,
                    response_reference_format=validation_response_reference_format,
                    add_existing_topics_summary_format=additional_instructions_summary_format,
                    previous_table_introduction=previous_table_introduction_default,
                    previous_table=(
                        previous_table_content if show_previous_table == "Yes" else ""
                    ),
                    validate_prompt_suffix=validation_prompt_suffix_default.format(
                        additional_validation_issues=additional_validation_issues_provided
                    ),
                )
            else:
                # Ensure the validation wrapper is applied even for structured summaries
                structured_summary_instructions = structured_summary_prompt.format(
                    response_table=validation_response_table_prompt,
                    topics=validation_unique_topics_markdown,
                    summary_format=additional_instructions_summary_format,
                )

                validation_formatted_summary_prompt = (
                    f"{validation_prompt_prefix_default}"
                    f"{structured_summary_instructions}"
                    f"{previous_table_introduction_default}"
                    f"{previous_table_content if show_previous_table == 'Yes' else ''}"
                    f"{validation_prompt_suffix_struct_summary_default.format(additional_validation_issues=additional_validation_issues_provided)}"
                )

            validation_batch_file_path_details = f"{file_name_clean}_val_batch_{validation_latest_batch_completed + 1}_size_{batch_size}_col_{in_column_cleaned}"

            # Use the helper function to process the validation batch
            (
                validation_new_topic_df,
                validation_new_reference_df,
                validation_new_topic_summary_df,
                validation_is_error,
                validation_current_prompt_content_logged,
                validation_current_summary_content_logged,
                validation_current_conversation_content_logged,
                validation_current_metadata_content_logged,
                validation_topic_table_out_path,
                validation_reference_table_out_path,
                validation_topic_summary_df_out_path,
            ) = process_batch_with_llm(
                is_first_batch=False,
                formatted_system_prompt=validation_formatted_system_prompt,
                formatted_prompt=validation_formatted_summary_prompt,
                batch_file_path_details=validation_batch_file_path_details,
                model_source=model_source,
                model_choice=model_choice,
                in_api_key=in_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                azure_api_key_textbox=azure_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                reasoning_suffix=reasoning_suffix,
                local_model=local_model,
                tokenizer=tokenizer,
                bedrock_runtime=bedrock_runtime,
                reported_batch_no=validation_reported_batch_no,
                response_text="",
                whole_conversation=list(),
                all_metadata_content=list(),
                start_row=validation_start_row,
                end_row=validation_end_row,
                model_choice_clean=model_choice_clean,
                log_files_output_paths=list(),
                existing_reference_df=validation_reference_df,
                existing_topic_summary_df=validation_existing_topic_summary_df,
                batch_size=batch_size,
                batch_basic_response_df=validation_batch_basic_response_df,
                group_name=group_name,
                produce_structured_summary_radio=produce_structured_summary_radio,
                output_folder=output_folder,
                output_debug_files=output_debug_files,
                task_type="Validation",
                assistant_prefill=add_existing_topics_assistant_prefill,
                api_url=api_url,
            )

            if validation_new_topic_df.empty:
                validation_new_topic_df = previous_topic_df
                # print("Validation new topic df is empty, using previous topic df:", validation_new_topic_df)
                # print("Validation new topic df columns:", validation_new_topic_df.columns)

            # Collect conversation metadata from validation batch
            if validation_current_metadata_content_logged:
                validation_all_metadata_content.append(
                    validation_current_metadata_content_logged
                )

            validation_all_prompts_content.append(
                validation_current_prompt_content_logged
            )
            validation_all_summaries_content.append(
                validation_current_summary_content_logged
            )
            validation_all_conversation_content.append(
                validation_current_conversation_content_logged
            )
            validation_all_groups_content.append(group_name)
            validation_all_batches_content.append(
                f"Validation {validation_reported_batch_no}:"
            )
            validation_all_model_choice_content.append(model_choice_clean_short)
            validation_all_validated_content.append("Yes")
            validation_all_task_type_content.append("Validation")
            validation_all_file_names_content.append(original_full_file_name)

            print("Appended to logs")

            # Update validation dataframes with validation results
            # For validation, we need to accumulate results from each batch, not overwrite them
            # The validation_new_* dataframes contain the results for the current batch
            # We need to concatenate them with the existing validation dataframes

            # For reference_df, we need to be careful about duplicates
            if not validation_new_reference_df.empty:
                # Check if the new reference_df is the same as the existing one (indicating "no change" response)
                # This happens when the LLM responds with "no change" and returns the existing data
                if validation_new_reference_df.equals(validation_reference_df):
                    print(
                        "Validation new reference df is identical to existing df (no change response), skipping concatenation"
                    )
                else:
                    # print("Validation new reference df is not empty, appending new table to validation reference df")
                    # Remove any existing entries for this batch range to avoid duplicates
                    start_row_reported = int(validation_start_row) + 1
                    end_row_reported = int(validation_end_row) + 1
                    validation_reference_df["Start row of group"] = (
                        validation_reference_df["Start row of group"].astype(int)
                    )

                    # Remove existing entries for this batch range from validation_reference_df
                    if "Start row of group" in validation_reference_df.columns:
                        validation_reference_df = validation_reference_df[
                            ~(
                                (
                                    validation_reference_df["Start row of group"]
                                    >= start_row_reported
                                )
                                & (
                                    validation_reference_df["Start row of group"]
                                    <= end_row_reported
                                )
                            )
                        ]

                    # Concatenate the new results
                    validation_reference_df = pd.concat(
                        [validation_reference_df, validation_new_reference_df]
                    ).dropna(how="all")

            # For topic summary, we need to merge/concatenate carefully to avoid duplicates
            if not validation_new_topic_summary_df.empty:
                # Check if the new topic_summary_df is the same as the existing one (indicating "no change" response)
                if validation_new_topic_summary_df.equals(validation_topic_summary_df):
                    print(
                        "Validation new topic summary df is identical to existing df (no change response), skipping concatenation"
                    )
                else:
                    # Remove duplicates and concatenate
                    validation_topic_summary_df = (
                        pd.concat(
                            [
                                validation_topic_summary_df,
                                validation_new_topic_summary_df,
                            ]
                        )
                        .drop_duplicates(["General topic", "Subtopic", "Sentiment"])
                        .dropna(how="all")
                    )

        else:
            print(
                "Current validation batch of responses contains no text, moving onto next. Batch number:",
                str(validation_latest_batch_completed + 1),
                ". Start row:",
                validation_start_row,
                ". End row:",
                validation_end_row,
            )

        # Increase validation batch counter
        validation_latest_batch_completed += 1

        # Check if we've exceeded max time for validation loop
        validation_toc = time.perf_counter()
        validation_final_time = validation_toc - tic

        if validation_final_time > max_time_for_loop:
            print("Max time reached during validation, breaking validation loop.")
            if progress:
                validation_loop.close()
                tqdm._instances.clear()
            break

    # Combine validation logged content
    validation_all_logged_content = [
        {
            "prompt": prompt,
            "response": summary,
            "metadata": metadata,
            "batch": batch,
            "model_choice": model_choice,
            "validated": validated,
            "group": group,
            "task_type": task_type,
            "file_name": file_name,
        }
        for prompt, summary, metadata, batch, model_choice, validated, group, task_type, file_name in zip(
            validation_all_prompts_content,
            validation_all_summaries_content,
            validation_all_metadata_content,
            validation_all_batches_content,
            validation_all_model_choice_content,
            validation_all_validated_content,
            validation_all_groups_content,
            validation_all_task_type_content,
            validation_all_file_names_content,
        )
    ]

    # Append validation content to original logged content
    updated_logged_content = list(logged_content) + list(validation_all_logged_content)

    # Combine validation conversation metadata
    validation_conversation_metadata_str = " ".join(validation_all_metadata_content)

    # Ensure consistent Topic number assignment by recreating topic_summary_df from reference_df
    if not validation_reference_df.empty:
        validation_topic_summary_df = create_topic_summary_df_from_reference_table(
            validation_reference_df
        )

    # Sort output dataframes
    validation_reference_df["Response References"] = (
        validation_reference_df["Response References"].astype(float).astype(int)
    )
    validation_reference_df["Start row of group"] = validation_reference_df[
        "Start row of group"
    ].astype(int)
    validation_reference_df.drop_duplicates(
        ["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True
    )
    validation_reference_df.sort_values(
        [
            "Group",
            "Start row of group",
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
        ],
        inplace=True,
    )

    validation_topic_summary_df["Number of responses"] = validation_topic_summary_df[
        "Number of responses"
    ].astype(int)
    validation_topic_summary_df.drop_duplicates(
        ["General topic", "Subtopic", "Sentiment"], inplace=True
    )
    validation_topic_summary_df.sort_values(
        ["Group", "Number of responses", "General topic", "Subtopic", "Sentiment"],
        ascending=[True, False, True, True, True],
        inplace=True,
    )

    print("Validation process completed.")

    return (
        validation_reference_df,
        validation_topic_summary_df,
        updated_logged_content,
        validation_conversation_metadata_str,
    )


# Define validation wrapper function
def validate_topics_wrapper(
    file_data: pd.DataFrame,
    reference_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    file_name: str,
    chosen_cols: List[str],
    batch_size: int,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    max_tokens: int,
    azure_api_key_textbox: str,
    azure_endpoint_textbox: str,
    reasoning_suffix: str,
    group_name: str,
    produce_structured_summary_radio: str,
    force_zero_shot_radio: str,
    force_single_topic_radio: str,
    context_textbox: str,
    additional_instructions_summary_format: str,
    output_folder: str,
    output_debug_files: str,
    original_full_file_name: str,
    additional_validation_issues_provided: str,
    max_time_for_loop: int,
    in_data_files: Any = None,
    sentiment_checkbox: str = "Negative or Positive",
    logged_content: List[dict] = None,
    show_previous_table: str = "Yes",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    api_url: str = None,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], str, int, int, int, List[str]]:
    """
    Wrapper function for validate_topics that processes data grouped by Group values and accumulates results,
    similar to wrapper_extract_topics_per_column_value.

    Args:
        file_data (pd.DataFrame): The input data to validate.
        reference_df (pd.DataFrame): The reference dataframe from the original run.
        topic_summary_df (pd.DataFrame): The topic summary dataframe from the original run.
        file_name (str): Name of the file being processed.
        chosen_cols (List[str]): Columns to process.
        batch_size (int): Size of each batch.
        model_choice (str): The model to use for validation.
        in_api_key (str): API key for the model.
        temperature (float): Temperature for the model.
        max_tokens (int): Maximum tokens for the model.
        azure_api_key_textbox (str): Azure API key if using Azure.
        azure_endpoint_textbox (str): Azure endpoint if using Azure.
        reasoning_suffix (str): Suffix for reasoning.
        group_name (str): Name of the group.
        produce_structured_summary_radio (str): Whether to produce structured summaries ("Yes" or "No").
        force_zero_shot_radio (str): Whether to force zero-shot ("Yes" or "No").
        force_single_topic_radio (str): Whether to force single topic ("Yes" or "No").
        context_textbox (str): Context for the validation.
        additional_instructions_summary_format (str): Additional instructions for summary format.
        output_folder (str): Output folder for files.
        output_debug_files (str): Whether to output debug files ("True" or "False").
        original_full_file_name (str): Original file name.
        additional_validation_issues_provided (str): Additional validation issues provided.
        max_time_for_loop (int): Maximum time for the loop.
        in_data_files (Any, optional): The input data files (e.g., Gradio FileData). If None, file_data must be provided.
        sentiment_checkbox (str): Sentiment analysis option.
        logged_content (List[dict], optional): The logged content from the original run. If None, tables will be reconstructed from reference_df.
        show_previous_table (str): Whether to show the previous table ("Yes" or "No").
        aws_access_key_textbox (str): AWS access key.
        aws_secret_key_textbox (str): AWS secret key.
        progress (gr.Progress): Progress bar object.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[dict], str, int, int, int, List[str]]:
            Accumulated reference_df, topic_summary_df, logged_content, conversation_metadata_str,
            total_input_tokens, total_output_tokens, total_llm_calls, and a list of output file paths.
    """

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice)

    # Handle None logged_content
    if logged_content is None:
        logged_content = list()

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            (
                in_colnames_drop,
                in_excel_sheets,
                file_name,
                join_colnames,
                join_colnames_drop,
            ) = put_columns_in_df(in_data_files)
            file_data, file_name, num_batches = load_in_data_file(
                in_data_files, chosen_cols, batch_size_default, in_excel_sheets
            )
        except Exception as e:
            out_message = "Could not load in data file due to: " + str(e)
            print(out_message)
            raise Exception(out_message)

    if file_data.shape[0] > max_rows:
        out_message = (
            "Your data has more than "
            + str(max_rows)
            + " rows, which has been set as the maximum in the application configuration."
        )
        print(out_message)
        raise Exception(out_message)

    if group_name is None:
        print("No grouping column found")
        file_data["group_col"] = "All"
        group_name = "group_col"

    if group_name not in file_data.columns:
        raise ValueError(f"Selected column '{group_name}' not found in file_data.")

    # Get unique Group values from the input dataframes
    unique_groups = list()
    if "Group" in reference_df.columns and not reference_df["Group"].isnull().all():
        unique_groups = reference_df["Group"].unique()
    elif (
        "Group" in topic_summary_df.columns
        and not topic_summary_df["Group"].isnull().all()
    ):
        unique_groups = topic_summary_df["Group"].unique()
    else:
        # If no Group column exists, use the provided group_name
        unique_groups = [group_name]

    # Limit to MAX_GROUPS if there are too many
    if len(unique_groups) > MAX_GROUPS:
        print(
            f"Warning: More than {MAX_GROUPS} unique groups found. Processing only the first {MAX_GROUPS}."
        )
        unique_groups = unique_groups[:MAX_GROUPS]

    print(f"Processing validation for {len(unique_groups)} groups: {unique_groups}")

    # Initialise accumulators for results across all groups
    acc_reference_df = pd.DataFrame()
    acc_topic_summary_df = pd.DataFrame()
    acc_logged_content = list()
    acc_conversation_metadata = ""
    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_llm_calls = 0
    acc_output_files = list()

    if len(unique_groups) == 1:
        # If only one unique value, no need for progress bar, iterate directly
        loop_object = unique_groups
    else:
        # If multiple unique values, use tqdm progress bar
        loop_object = progress.tqdm(
            unique_groups, desc="Validating groups", unit="groups"
        )

    # Process each group separately
    for i, current_group in enumerate(loop_object):
        print(
            f"\nProcessing validation for group: {current_group} ({i+1}/{len(unique_groups)})"
        )

        # Filter data for current group
        if "Group" in reference_df.columns:
            group_reference_df = reference_df[
                reference_df["Group"] == current_group
            ].copy()
        else:
            group_reference_df = reference_df.copy()

        if "Group" in topic_summary_df.columns:
            group_topic_summary_df = topic_summary_df[
                topic_summary_df["Group"] == current_group
            ].copy()
        else:
            group_topic_summary_df = topic_summary_df.copy()

        # Filter file_data if it has a Group column
        if "Group" in file_data.columns:
            group_file_data = file_data[file_data["Group"] == current_group].copy()
        else:
            group_file_data = file_data.copy()

        # Skip if no data for this group
        if group_reference_df.empty and group_topic_summary_df.empty:
            print(f"No data for group {current_group}. Skipping.")
            continue

        try:
            # Call validate_topics for this group
            (
                validation_reference_df,
                validation_topic_summary_df,
                updated_logged_content,
                validation_conversation_metadata_str,
            ) = validate_topics(
                file_data=group_file_data,
                reference_df=group_reference_df,
                topic_summary_df=group_topic_summary_df,
                file_name=file_name,
                chosen_cols=chosen_cols,
                batch_size=batch_size,
                model_choice=model_choice,
                in_api_key=in_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                azure_api_key_textbox=azure_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                reasoning_suffix=reasoning_suffix,
                group_name=current_group,
                produce_structured_summary_radio=produce_structured_summary_radio,
                force_zero_shot_radio=force_zero_shot_radio,
                force_single_topic_radio=force_single_topic_radio,
                context_textbox=context_textbox,
                additional_instructions_summary_format=additional_instructions_summary_format,
                output_folder=output_folder,
                output_debug_files=output_debug_files,
                original_full_file_name=original_full_file_name,
                additional_validation_issues_provided=additional_validation_issues_provided,
                max_time_for_loop=max_time_for_loop,
                sentiment_checkbox=sentiment_checkbox,
                logged_content=logged_content,
                show_previous_table=show_previous_table,
                aws_access_key_textbox=aws_access_key_textbox,
                aws_secret_key_textbox=aws_secret_key_textbox,
                api_url=api_url,
            )

            # Accumulate results
            if not validation_reference_df.empty:
                acc_reference_df = pd.concat(
                    [acc_reference_df, validation_reference_df], ignore_index=True
                )
                acc_reference_df.drop_duplicates(
                    ["Response References", "General topic", "Subtopic", "Sentiment"],
                    inplace=True,
                )
            if not validation_topic_summary_df.empty:
                acc_topic_summary_df = pd.concat(
                    [acc_topic_summary_df, validation_topic_summary_df],
                    ignore_index=True,
                )
                acc_topic_summary_df.drop_duplicates(
                    ["General topic", "Subtopic", "Sentiment"], inplace=True
                )

            acc_logged_content.extend(updated_logged_content)
            acc_conversation_metadata += (
                ("\n---\n" if acc_conversation_metadata else "")
                + f"Validation for group {current_group}:\n"
                + validation_conversation_metadata_str
            )

            # Calculate token counts for this group
            group_input_tokens, group_output_tokens, group_llm_calls = (
                calculate_tokens_from_metadata(
                    validation_conversation_metadata_str, model_choice, model_name_map
                )
            )

            acc_input_tokens += int(group_input_tokens)
            acc_output_tokens += int(group_output_tokens)
            acc_llm_calls += int(group_llm_calls)

            print(f"Group {current_group} validation completed.")

        except Exception as e:
            print(f"Error processing validation for group {current_group}: {e}")
            continue

    # Create consolidated output files
    file_name_clean = clean_column_name(file_name, max_length=20)
    clean_column_name(chosen_cols, max_length=20)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    # Create consolidated output file paths
    validation_reference_table_path = f"{output_folder}{file_name_clean}_all_final_reference_table_{model_choice_clean_short}_valid.csv"
    validation_unique_topics_path = f"{output_folder}{file_name_clean}_all_final_unique_topics_{model_choice_clean_short}_valid.csv"

    # Need to join "Topic number" onto acc_reference_df
    # If any blanks, there is an issue somewhere, drop and redo
    if "Topic number" in acc_reference_df.columns:
        if acc_reference_df["Topic number"].isnull().any():
            acc_reference_df = acc_reference_df.drop("Topic number", axis=1)

    if "Topic number" not in acc_reference_df.columns:
        if "Topic number" in acc_topic_summary_df.columns:
            if "General topic" in acc_topic_summary_df.columns:
                acc_reference_df = acc_reference_df.merge(
                    acc_topic_summary_df[
                        ["General topic", "Subtopic", "Sentiment", "Topic number"]
                    ],
                    on=["General topic", "Subtopic", "Sentiment"],
                    how="left",
                )
                # Sort output dataframes
                acc_reference_df["Response References"] = (
                    acc_reference_df["Response References"].astype(float).astype(int)
                )
                acc_reference_df["Start row of group"] = acc_reference_df[
                    "Start row of group"
                ].astype(int)
                acc_reference_df.sort_values(
                    [
                        "Group",
                        "Start row of group",
                        "Response References",
                        "General topic",
                        "Subtopic",
                        "Sentiment",
                    ],
                    inplace=True,
                )
            elif "Main heading" in acc_topic_summary_df.columns:
                acc_reference_df = acc_reference_df.merge(
                    acc_topic_summary_df[
                        ["Main heading", "Subheading", "Topic number"]
                    ],
                    on=["Main heading", "Subheading"],
                    how="left",
                )
                # Sort output dataframes
                acc_reference_df["Response References"] = (
                    acc_reference_df["Response References"].astype(float).astype(int)
                )
                acc_reference_df["Start row of group"] = acc_reference_df[
                    "Start row of group"
                ].astype(int)
                acc_reference_df.sort_values(
                    [
                        "Group",
                        "Start row of group",
                        "Response References",
                        "Main heading",
                        "Subheading",
                        "Topic number",
                    ],
                    inplace=True,
                )

    if "General topic" in acc_topic_summary_df.columns:
        acc_topic_summary_df["Number of responses"] = acc_topic_summary_df[
            "Number of responses"
        ].astype(int)
        acc_topic_summary_df.sort_values(
            ["Group", "Number of responses", "General topic", "Subtopic", "Sentiment"],
            ascending=[True, False, True, True, True],
            inplace=True,
        )
    elif "Main heading" in acc_topic_summary_df.columns:
        acc_topic_summary_df["Number of responses"] = acc_topic_summary_df[
            "Number of responses"
        ].astype(int)
        acc_topic_summary_df.sort_values(
            [
                "Group",
                "Number of responses",
                "Main heading",
                "Subheading",
                "Topic number",
            ],
            ascending=[True, False, True, True, True],
            inplace=True,
        )

    # Save consolidated validation dataframes to CSV
    if not acc_reference_df.empty:
        acc_reference_df.to_csv(
            validation_reference_table_path, index=None, encoding="utf-8-sig"
        )
        acc_output_files.append(validation_reference_table_path)

    if not acc_topic_summary_df.empty:
        acc_topic_summary_df.to_csv(
            validation_unique_topics_path, index=None, encoding="utf-8-sig"
        )
        acc_output_files.append(validation_unique_topics_path)

    if "Group" in acc_reference_df.columns:
        # Create missing references dataframe using consolidated data from all groups
        # This ensures we correctly identify missing references across all groups
        # Get all basic_response_data from all groups
        all_basic_response_data = list()
        for logged_item in acc_logged_content:
            if "basic_response_data" in logged_item:
                all_basic_response_data.extend(logged_item["basic_response_data"])

        if all_basic_response_data:
            all_basic_response_df = pd.DataFrame(all_basic_response_data)
            acc_missing_df = create_missing_references_df(
                all_basic_response_df, acc_reference_df
            )
        else:
            # Fallback: if no logged content, create empty missing_df
            acc_missing_df = pd.DataFrame(
                columns=["Missing Reference", "Response Character Count"]
            )
    else:
        # Fallback: if no logged content, create empty missing_df
        acc_missing_df = pd.DataFrame(
            columns=["Missing Reference", "Response Character Count"]
        )

    # Create display table markdown for validation results
    if not acc_topic_summary_df.empty:
        validation_display_table = acc_topic_summary_df.copy()
        if "Summary" in validation_display_table.columns:
            validation_display_table = validation_display_table.drop("Summary", axis=1)

        # Apply text wrapping for display
        validation_display_table = validation_display_table.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )

        # Handle structured summary format
        if produce_structured_summary_radio == "Yes":
            if "General topic" in validation_display_table.columns:
                validation_display_table = validation_display_table.rename(
                    columns={"General topic": "Main heading"}
                )
            if "Subtopic" in validation_display_table.columns:
                validation_display_table = validation_display_table.rename(
                    columns={"Subtopic": "Subheading"}
                )
            validation_display_table_markdown = validation_display_table.to_markdown(
                index=False
            )
        else:
            validation_display_table_markdown = validation_display_table.to_markdown(
                index=False
            )
    else:
        validation_display_table_markdown = "No validation results available."

    print(
        f"Validation completed for all groups. Total tokens: {acc_input_tokens} input, {acc_output_tokens} output"
    )

    # Return the same format as wrapper_extract_topics_per_column_value
    return (
        validation_display_table_markdown,  # display_topic_table_markdown
        acc_topic_summary_df,  # master_unique_topics_df_state
        acc_topic_summary_df,  # master_unique_topics_df_state (duplicate for compatibility)
        acc_reference_df,  # master_reference_df_state
        acc_output_files,  # topic_extraction_output_files
        acc_output_files,  # text_output_file_list_state
        0,  # latest_batch_completed (reset for validation)
        [],  # log_files_output (empty for validation)
        [],  # log_files_output_list_state (empty for validation)
        acc_conversation_metadata,  # conversation_metadata_textbox
        0.0,  # estimated_time_taken_number (reset for validation)
        acc_output_files,  # deduplication_input_files
        acc_output_files,  # summarisation_input_files
        acc_topic_summary_df,  # modifiable_unique_topics_df_state
        acc_output_files,  # modification_input_files
        [],  # in_join_files (empty for validation)
        acc_missing_df,  # missing_df_state (empty for validation)
        acc_input_tokens,  # input_tokens_num
        acc_output_tokens,  # output_tokens_num
        acc_llm_calls,  # number_of_calls_num
        f"Validation completed for {len(unique_groups)} groups",  # output_messages_textbox
        acc_logged_content,  # logged_content_df
    )


def data_file_to_markdown_table(
    file_data: pd.DataFrame,
    file_name: str,
    chosen_cols: List[str],
    batch_number: int,
    batch_size: int,
    verify_titles: bool = False,
) -> Tuple[str, str, str]:
    """
    Processes a file by simplifying its content based on chosen columns and saves the result to a specified output folder.

    Parameters:
    - file_data (pd.DataFrame): Tabular data file with responses.
    - file_name (str): File name with extension.
    - chosen_cols (List[str]): A list of column names to include in the simplified file.
    - batch_number (int): The current batch number for processing.
    - batch_size (int): The number of rows to process in each batch.

    Returns:
    - Tuple[str, str, str]: A tuple containing the path to the simplified CSV file, the simplified markdown table as a string, and the file path end (used for naming the output file).
    """

    normalised_simple_markdown_table = ""
    simplified_csv_table_path = ""

    # Simplify table to just responses column and the Response reference number
    basic_response_data = get_basic_response_data(
        file_data, chosen_cols, verify_titles=verify_titles
    )

    file_len = int(len(basic_response_data["Reference"]))
    batch_size = int(batch_size)
    batch_number = int(batch_number)

    # Subset the data for the current batch
    start_row = int(batch_number * batch_size)

    if start_row > file_len + 1:
        print("Start row greater than file row length")
        return simplified_csv_table_path, normalised_simple_markdown_table, file_name
    if start_row < 0:
        raise Exception("Start row is below 0")

    if ((start_row + batch_size) - 1) <= file_len + 1:
        end_row = int((start_row + batch_size) - 1)
    else:
        end_row = file_len + 1

    batch_basic_response_data = basic_response_data.loc[
        start_row:end_row, ["Reference", "Response", "Original Reference"]
    ]  # Select the current batch

    # Now replace the reference numbers with numbers starting from 1
    batch_basic_response_data.loc[:, "Reference"] = (
        batch_basic_response_data["Reference"] - start_row
    )

    # Remove problematic characters including control characters, special characters, and excessive leading/trailing whitespace
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data[
        "Response"
    ].str.replace(
        r'[\x00-\x1F\x7F]|[""<>]|\\', "", regex=True
    )  # Remove control and special characters
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data[
        "Response"
    ].str.strip()  # Remove leading and trailing whitespace
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data[
        "Response"
    ].str.replace(
        r"\s+", " ", regex=True
    )  # Replace multiple spaces with a single space
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data[
        "Response"
    ].str.replace(
        r"\n{2,}", "\n", regex=True
    )  # Replace multiple line breaks with a single line break
    batch_basic_response_data.loc[:, "Response"] = batch_basic_response_data[
        "Response"
    ].str.slice(
        0, max_comment_character_length
    )  # Maximum 1,500 character responses

    # Remove blank and extremely short responses
    batch_basic_response_data = batch_basic_response_data.loc[
        ~(batch_basic_response_data["Response"].isnull())
        & ~(batch_basic_response_data["Response"] == "None")
        & ~(batch_basic_response_data["Response"] == " ")
        & ~(batch_basic_response_data["Response"] == ""),
        :,
    ]  # ~(batch_basic_response_data["Response"].str.len() < 5), :]

    simple_markdown_table = batch_basic_response_data[
        ["Reference", "Response"]
    ].to_markdown(index=None)

    normalised_simple_markdown_table = normalise_string(simple_markdown_table)

    # print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

    return (
        simplified_csv_table_path,
        normalised_simple_markdown_table,
        start_row,
        end_row,
        batch_basic_response_data,
    )


def replace_punctuation_with_underscore(input_string: str):
    # Create a translation table where each punctuation character maps to '_'
    translation_table = str.maketrans(string.punctuation, "_" * len(string.punctuation))

    # Translate the input string using the translation table
    return input_string.translate(translation_table)


### INITIAL TOPIC MODEL DEVELOPMENT FUNCTIONS


def clean_markdown_table(text: str):
    # Split text into lines
    lines = text.splitlines()

    # Step 1: Identify table structure and process line continuations
    table_rows = list()
    current_row = None

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if this is a table row or alignment row
        is_table_row = "|" in stripped or stripped.startswith(":-") or ":-:" in stripped

        if is_table_row:
            # If we have a current row being built, add it to our list
            if current_row is not None:
                table_rows.append(current_row)

            # Start a new row
            current_row = stripped
        elif current_row is not None:
            # This must be a continuation of the previous row
            current_row += " " + stripped
        else:
            # Not part of the table
            current_row = stripped

    # Don't forget the last row
    if current_row is not None:
        table_rows.append(current_row)

    # Step 2: Properly format the table
    # First, determine the maximum number of columns
    max_columns = 0
    for row in table_rows:
        cells = row.split("|")
        # Account for rows that may not start/end with a pipe
        if row.startswith("|"):
            cells = cells[1:]
        if row.endswith("|"):
            cells = cells[:-1]
        max_columns = max(max_columns, len(cells))

    # Now format each row
    formatted_rows = list()
    for row in table_rows:
        # Ensure the row starts and ends with pipes
        if not row.startswith("|"):
            row = "|" + row
        if not row.endswith("|"):
            row = row + "|"

        # Split into cells
        cells = row.split("|")[1:-1]  # Remove empty entries from split

        # Ensure we have the right number of cells
        while len(cells) < max_columns:
            cells.append("")

        # Rebuild the row
        formatted_row = "|" + "|".join(cells) + "|"
        formatted_rows.append(formatted_row)

    # Join everything back together
    result = "\n".join(formatted_rows)

    return result


# Convert output table to markdown and then to a pandas dataframe to csv
def remove_before_last_term(input_string: str) -> str:
    # Use regex to find the last occurrence of the term
    match = re.search(r"(\| ?General topic)", input_string)
    if match:
        # Find the last occurrence by using rfind
        last_index = input_string.rfind(match.group(0))
        return input_string[last_index:]  # Return everything from the last match onward
    return input_string  # Return the original string if the term is not found


def convert_to_html_table(input_string: str, table_type: str = "Main table"):
    # Remove HTML tags from input string
    input_string = input_string.replace("<p>", "").replace("</p>", "")

    if "<table" in input_string:
        # Input is already in HTML format
        html_table = input_string
    else:
        # Input is in Markdown format
        lines = input_string.strip().split("\n")
        clean_md_text = "\n".join(
            [lines[0]] + lines[2:]
        )  # Keep header, skip separator, keep data

        # Read Markdown table into a DataFrame
        df = pd.read_csv(
            pd.io.common.StringIO(clean_md_text),
            sep="|",
            skipinitialspace=True,
            dtype={"Response References": str},
        )

        # Ensure unique column names
        df.columns = [
            f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col
            for i, col in enumerate(df.columns)
        ]

        # Convert DataFrame to HTML
        html_table = df.to_html(index=False, border=1)

    # Ensure that the HTML structure is correct
    if table_type == "Main table":
        if "<table" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>General topic</th>
                    <th>Subtopic</th>
                    <th>Sentiment</th>                
                    <th>Response References</th>
                    <th>Summary</th>
                </tr>
                {html_table}
            </table>
            """
    elif table_type == "Revised topics table":
        if "<table" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>General topic</th>
                    <th>Subtopic</th>
                </tr>
                {html_table}
            </table>
            """
    elif table_type == "Verify titles table":
        if "<table" not in html_table:
            html_table = f"""
            <table>
                <tr>
                    <th>Response References</th>
                    <th>Is this a suitable title</th>
                    <th>Explanation</th>
                    <th>Alternative title</th>
                </tr>
                {html_table}
            </table>
            """

    return html_table


def convert_response_text_to_dataframe(
    response_text: str, table_type: str = "Main table"
):
    is_error = False
    start_of_table_response = remove_before_last_term(response_text)

    cleaned_response = clean_markdown_table(start_of_table_response)

    # Add a space after commas between numbers (e.g., "1,2" -> "1, 2")
    cleaned_response = re.sub(r"(\d),(\d)", r"\1, \2", cleaned_response)

    try:
        string_html_table = markdown.markdown(
            cleaned_response, extensions=["markdown.extensions.tables"]
        )
    except Exception as e:
        print("Unable to convert response to string_html_table due to", e)
        string_html_table = ""

    html_table = convert_to_html_table(string_html_table)

    html_buffer = StringIO(html_table)

    try:
        tables = pd.read_html(html_buffer)
        if tables:
            out_df = tables[0]  # Use the first table if available
        else:
            raise ValueError("No tables found in the provided HTML.")
            is_error = True
            out_df = pd.DataFrame()
    except Exception as e:
        print("Error when trying to parse table:", e)
        is_error = True
        out_df = pd.DataFrame()

    return out_df, is_error


def write_llm_output_and_logs(
    response_text: str,
    whole_conversation: List[str],
    all_metadata_content: List[str],
    batch_file_path_details: str,
    start_row: int,
    end_row: int,
    model_choice_clean: str,
    log_files_output_paths: List[str],
    existing_reference_df: pd.DataFrame,
    existing_topics_df: pd.DataFrame,
    batch_size_number: int,
    batch_basic_response_df: pd.DataFrame,
    group_name: str = "All",
    produce_structured_summary_radio: str = "No",
    return_logs: bool = False,
    output_folder: str = OUTPUT_FOLDER,
) -> Tuple:
    """
    Writes the output of the large language model requests and logs to files.

    Parameters:
    - response_text (str): The text of the response from the model.
    - whole_conversation (List[str]): A list of strings representing the complete conversation including prompts and responses.
    - all_metadata_content (List[str]): A list of strings representing metadata about the whole conversation.
    - batch_file_path_details (str): String containing details for constructing batch-specific file paths.
    - start_row (int): Start row of the current batch.
    - end_row (int): End row of the current batch.
    - model_choice_clean (str): The cleaned model choice string.
    - log_files_output_paths (List[str]): A list of paths to the log files.
    - existing_reference_df (pd.DataFrame): The existing reference dataframe mapping response numbers to topics.
    - existing_topics_df (pd.DataFrame): The existing unique topics dataframe.
    - batch_size_number (int): The size of batches in terms of number of responses.
    - batch_basic_response_df (pd.DataFrame): The dataframe that contains the response data.
    - group_name (str, optional): The name of the current group.
    - produce_structured_summary_radio (str, optional): Whether the option to produce structured summaries has been selected.
    - return_logs (bool): A boolean indicating if logs should be returned. Defaults to False.
    - output_folder (str): The name of the folder where output files are saved.
    """
    topic_summary_df_out_path = list()
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    topic_summary_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame(
        columns=[
            "General topic",
            "Subtopic",
            "Sentiment",
            "Response References",
            "Summary",
        ]
    )
    out_reference_df = pd.DataFrame(
        columns=[
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
            "Summary",
            "Start row of group",
        ]
    )
    out_topic_summary_df = pd.DataFrame(
        columns=["General topic", "Subtopic", "Sentiment"]
    )
    is_error = False  # If there was an error in parsing, return boolean saying error

    if produce_structured_summary_radio == "Yes":
        existing_topics_df.rename(
            columns={"Main heading": "General topic", "Subheading": "Subtopic"},
            inplace=True,
            errors="ignore",
        )
        existing_reference_df.rename(
            columns={"Main heading": "General topic", "Subheading": "Subtopic"},
            inplace=True,
            errors="ignore",
        )
        topic_with_response_df.rename(
            columns={"Main heading": "General topic", "Subheading": "Subtopic"},
            inplace=True,
            errors="ignore",
        )
        out_reference_df.rename(
            columns={"Main heading": "General topic", "Subheading": "Subtopic"},
            inplace=True,
            errors="ignore",
        )
        out_topic_summary_df.rename(
            columns={"Main heading": "General topic", "Subheading": "Subtopic"},
            inplace=True,
            errors="ignore",
        )

    # Convert conversation to string and add to log outputs
    whole_conversation_str = "\n".join(whole_conversation)
    all_metadata_content_str = "\n".join(all_metadata_content)
    start_row_reported = int(start_row) + 1

    # Need to reduce output file names as full length files may be too long
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    row_number_string_start = f"Rows {start_row_reported} to {end_row + 1}: "

    if output_debug_files == "True" and return_logs is True:
        whole_conversation_path = (
            output_folder
            + batch_file_path_details
            + "_full_conversation_"
            + model_choice_clean_short
            + ".txt"
        )
        whole_conversation_path_meta = (
            output_folder
            + batch_file_path_details
            + "_metadata_"
            + model_choice_clean_short
            + ".txt"
        )
        with open(
            whole_conversation_path, "w", encoding="utf-8-sig", errors="replace"
        ) as f:
            f.write(whole_conversation_str)
        with open(
            whole_conversation_path_meta, "w", encoding="utf-8-sig", errors="replace"
        ) as f:
            f.write(all_metadata_content_str)
        log_files_output_paths.append(whole_conversation_path_meta)

    # Check if response is "No change" - if so, return input dataframes
    stripped_response = response_text.strip()
    if stripped_response.lower().startswith("no change"):
        print("LLM response indicates no changes needed, returning input dataframes")

        # For "No change" responses, we need to return the existing dataframes
        # but we still need to process them through the same logic as normal processing

        # Create empty topic_with_response_df since no new topics were generated
        if produce_structured_summary_radio == "Yes":
            topic_with_response_df = pd.DataFrame(
                columns=[
                    "Main heading",
                    "Subheading",
                    "Sentiment",
                    "Response References",
                    "Summary",
                ]
            )
        else:
            topic_with_response_df = pd.DataFrame(
                columns=[
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                    "Response References",
                    "Summary",
                ]
            )

        # For "No change", we return the existing dataframes as-is (they already contain all the data)
        # This is equivalent to the normal processing where new_reference_df would be empty
        out_reference_df = existing_reference_df.copy()
        out_topic_summary_df = existing_topics_df.copy()

        # Set up output file paths
        topic_table_out_path = (
            output_folder
            + batch_file_path_details
            + "_topic_table_"
            + model_choice_clean_short
            + ".csv"
        )
        reference_table_out_path = (
            output_folder
            + batch_file_path_details
            + "_reference_table_"
            + model_choice_clean_short
            + ".csv"
        )
        topic_summary_df_out_path = (
            output_folder
            + batch_file_path_details
            + "_unique_topics_"
            + model_choice_clean_short
            + ".csv"
        )

        # Return the existing dataframes (no changes needed)
        return (
            topic_table_out_path,
            reference_table_out_path,
            topic_summary_df_out_path,
            topic_with_response_df,
            out_reference_df,
            out_topic_summary_df,
            batch_file_path_details,
            is_error,
        )

    # Convert response text to a markdown table
    try:
        topic_with_response_df, is_error = convert_response_text_to_dataframe(
            response_text
        )
    except Exception as e:
        print("Error in parsing markdown table from response text:", e)

        return (
            topic_table_out_path,
            reference_table_out_path,
            topic_summary_df_out_path,
            topic_with_response_df,
            out_reference_df,
            out_topic_summary_df,
            batch_file_path_details,
            is_error,
        )

    # If the table has 5 columns, rename them
    # Rename columns to ensure consistent use of data frames later in code
    if topic_with_response_df.shape[1] == 5:
        new_column_names = {
            topic_with_response_df.columns[0]: "General topic",
            topic_with_response_df.columns[1]: "Subtopic",
            topic_with_response_df.columns[2]: "Sentiment",
            topic_with_response_df.columns[3]: "Response References",
            topic_with_response_df.columns[4]: "Summary",
        }

        topic_with_response_df = topic_with_response_df.rename(columns=new_column_names)

    else:
        # Something went wrong with the table output, so add empty columns
        print("Table output has wrong number of columns, adding with blank values")
        # First, rename first two columns that should always exist.
        new_column_names = {
            topic_with_response_df.columns[0]: "General topic",
            topic_with_response_df.columns[1]: "Subtopic",
        }
        topic_with_response_df.rename(columns=new_column_names, inplace=True)

        # Add empty columns if they are not present
        if "Sentiment" not in topic_with_response_df.columns:
            topic_with_response_df["Sentiment"] = "Not assessed"
        if "Response References" not in topic_with_response_df.columns:
            if batch_size_number == 1:
                topic_with_response_df["Response References"] = "1"
            else:
                topic_with_response_df["Response References"] = ""
        if "Summary" not in topic_with_response_df.columns:
            topic_with_response_df["Summary"] = ""

        topic_with_response_df = topic_with_response_df[
            ["General topic", "Subtopic", "Sentiment", "Response References", "Summary"]
        ]

    # Fill in NA rows with values from above (topics seem to be included only on one row):
    topic_with_response_df = topic_with_response_df.ffill()

    # For instances where you end up with float values in Response References
    topic_with_response_df["Response References"] = (
        topic_with_response_df["Response References"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    # Strip and lower case topic names to remove issues where model is randomly capitalising topics/sentiment
    topic_with_response_df["General topic"] = (
        topic_with_response_df["General topic"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.capitalize()
    )
    topic_with_response_df["Subtopic"] = (
        topic_with_response_df["Subtopic"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.capitalize()
    )
    topic_with_response_df["Sentiment"] = (
        topic_with_response_df["Sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.capitalize()
    )

    topic_table_out_path = (
        output_folder
        + batch_file_path_details
        + "_topic_table_"
        + model_choice_clean_short
        + ".csv"
    )

    # Table to map references to topics
    reference_data = list()
    existing_reference_numbers = False

    batch_basic_response_df["Reference"] = batch_basic_response_df["Reference"].astype(
        str
    )
    batch_size_number = int(batch_size_number)

    # Iterate through each row in the original DataFrame
    for index, row in topic_with_response_df.iterrows():
        references_raw = str(row.iloc[3]) if pd.notna(row.iloc[3]) else ""
        references = re.findall(r"\d+", references_raw)

        if batch_size_number == 1:
            references = ["1"]

        # Filter out references that are outside the valid range
        if references:
            try:
                # Convert all references to integers and keep only those within valid range
                ref_numbers = [int(ref) for ref in references]
                references = [
                    ref
                    for ref in ref_numbers
                    if 1 <= int(ref) <= int(batch_size_number)
                ]
            except ValueError:
                # If any reference can't be converted to int, skip this row
                print("Response value could not be converted to number:", references)
                continue
        else:
            references = []

        topic = row.iloc[0] if pd.notna(row.iloc[0]) else ""
        subtopic = row.iloc[1] if pd.notna(row.iloc[1]) else ""
        sentiment = row.iloc[2] if pd.notna(row.iloc[2]) else ""
        summary = row.iloc[4] if pd.notna(row.iloc[4]) else ""

        # If the reference response column is very long, and there's nothing in the summary column, assume that the summary was put in the reference column
        if not summary and (len(str(row.iloc[3])) > 30):
            summary = row.iloc[3]

        if produce_structured_summary_radio != "Yes":
            summary = row_number_string_start + summary

        # Check if the 'references' list exists and is not empty

        if references:
            existing_reference_numbers = True

            # We process one reference at a time to create one dictionary entry per reference.
            for ref in references:
                # This variable will hold the final reference number for the current 'ref'
                response_ref_no = None

                # Now, we decide how to calculate 'response_ref_no' for the current 'ref'
                if batch_basic_response_df.empty:
                    # --- Scenario 1: The DataFrame is empty, so we calculate the reference ---
                    try:
                        response_ref_no = int(ref) + int(start_row)
                    except ValueError:
                        print(f"Reference '{ref}' is not a number and was skipped.")
                        continue  # Skip to the next 'ref' in the loop

                else:
                    # --- Scenario 2: The DataFrame is NOT empty, so we look up the reference ---
                    matching_series = batch_basic_response_df.loc[
                        batch_basic_response_df["Reference"] == str(ref),
                        "Original Reference",
                    ]

                    if not matching_series.empty:
                        # If found, get the first match
                        response_ref_no = matching_series.iloc[0]
                    else:
                        # If not found, report it and skip this reference
                        print(f"Reference '{ref}' not found in the DataFrame.")
                        continue  # Skip to the next 'ref' in the loop

                # This code runs for every *valid* reference that wasn't skipped by 'continue'.
                # It uses the 'response_ref_no' calculated in the if/else block above.
                reference_data.append(
                    {
                        "Response References": str(response_ref_no),
                        "General topic": topic,
                        "Subtopic": subtopic,
                        "Sentiment": sentiment,
                        "Summary": summary,
                        "Start row of group": start_row_reported,
                    }
                )

        # This 'else' corresponds to the 'if references:' at the top
        else:
            # This block runs only if the 'references' list was empty or None to begin with
            existing_reference_numbers = False
            response_ref_no = 0  # Default value when no references are provided

            reference_data.append(
                {
                    "Response References": str(response_ref_no),
                    "General topic": topic,
                    "Subtopic": subtopic,
                    "Sentiment": sentiment,
                    "Summary": summary,
                    "Start row of group": start_row_reported,
                }
            )

    # Create a new DataFrame from the reference data
    if reference_data:
        new_reference_df = pd.DataFrame(reference_data)
    else:
        new_reference_df = pd.DataFrame(
            columns=[
                "Response References",
                "General topic",
                "Subtopic",
                "Sentiment",
                "Summary",
                "Start row of group",
            ]
        )

    # Append on old reference data
    if not new_reference_df.empty:
        out_reference_df = pd.concat([new_reference_df, existing_reference_df]).dropna(
            how="all"
        )
    else:
        out_reference_df = existing_reference_df

    # Remove duplicate Response References for the same topic
    out_reference_df.drop_duplicates(
        ["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True
    )

    # Try converting response references column to int, keep as string if fails
    if existing_reference_numbers is True:
        try:
            out_reference_df["Response References"] = (
                out_reference_df["Response References"].astype(float).astype(int)
            )
        except Exception as e:
            print("Could not convert Response References column to integer due to", e)

    out_reference_df.sort_values(
        [
            "Start row of group",
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
        ],
        inplace=True,
    )

    # Each topic should only be associated with each individual response once
    out_reference_df.drop_duplicates(
        ["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True
    )
    out_reference_df["Group"] = group_name

    # Save the new DataFrame to CSV
    reference_table_out_path = (
        output_folder
        + batch_file_path_details
        + "_reference_table_"
        + model_choice_clean_short
        + ".csv"
    )

    # Table of all unique topics with descriptions
    new_topic_summary_df = topic_with_response_df[
        ["General topic", "Subtopic", "Sentiment"]
    ]

    new_topic_summary_df = new_topic_summary_df.rename(
        columns={
            new_topic_summary_df.columns[0]: "General topic",
            new_topic_summary_df.columns[1]: "Subtopic",
            new_topic_summary_df.columns[2]: "Sentiment",
        }
    )

    # Join existing and new unique topics
    out_topic_summary_df = pd.concat([new_topic_summary_df, existing_topics_df]).dropna(
        how="all"
    )

    out_topic_summary_df = out_topic_summary_df.rename(
        columns={
            out_topic_summary_df.columns[0]: "General topic",
            out_topic_summary_df.columns[1]: "Subtopic",
            out_topic_summary_df.columns[2]: "Sentiment",
        }
    )

    # print("out_topic_summary_df:", out_topic_summary_df)

    out_topic_summary_df = out_topic_summary_df.drop_duplicates(
        ["General topic", "Subtopic", "Sentiment"]
    ).drop(["Number of responses", "Summary"], axis=1, errors="ignore")

    # Get count of rows that refer to particular topics
    reference_counts = (
        out_reference_df.groupby(["General topic", "Subtopic", "Sentiment"])
        .agg(
            {
                "Response References": "size",  # Count the number of references
                "Summary": " <br> ".join,
            }
        )
        .reset_index()
    )

    # Join the counts to existing_topic_summary_df
    out_topic_summary_df = out_topic_summary_df.merge(
        reference_counts, how="left", on=["General topic", "Subtopic", "Sentiment"]
    ).sort_values("Response References", ascending=False)

    out_topic_summary_df = out_topic_summary_df.rename(
        columns={"Response References": "Number of responses"}, errors="ignore"
    )

    out_topic_summary_df["Group"] = group_name

    topic_summary_df_out_path = (
        output_folder
        + batch_file_path_details
        + "_unique_topics_"
        + model_choice_clean_short
        + ".csv"
    )

    return (
        topic_table_out_path,
        reference_table_out_path,
        topic_summary_df_out_path,
        topic_with_response_df,
        out_reference_df,
        out_topic_summary_df,
        batch_file_path_details,
        is_error,
    )


def process_batch_with_llm(
    is_first_batch: bool,
    formatted_system_prompt: str,
    formatted_prompt: str,
    batch_file_path_details: str,
    model_source: str,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    max_tokens: int,
    azure_api_key_textbox: str,
    azure_endpoint_textbox: str,
    reasoning_suffix: str,
    local_model: object,
    tokenizer: object,
    bedrock_runtime: object,
    reported_batch_no: int,
    response_text: str,
    whole_conversation: list,
    all_metadata_content: list,
    start_row: int,
    end_row: int,
    model_choice_clean: str,
    log_files_output_paths: list,
    existing_reference_df: pd.DataFrame,
    existing_topic_summary_df: pd.DataFrame,
    batch_size: int,
    batch_basic_response_df: pd.DataFrame,
    group_name: str,
    produce_structured_summary_radio: str,
    output_folder: str,
    output_debug_files: str,
    task_type: str,
    assistant_prefill: str = "",
    api_url: str = None,
):
    """Helper function to process a batch with LLM, handling the common logic between first and subsequent batches.

    This function orchestrates the interaction with various LLM providers (Gemini, Azure/OpenAI, AWS Bedrock, Local)
    to process a given batch of data. It constructs the client, calls the LLM with the specified prompts,
    and then processes the LLM's response to extract topics, references, and summaries, writing them to output files.
    It also handles error conditions related to LLM output parsing.

    Args:
        is_first_batch (bool): True if this is the first batch being processed, False otherwise.
        formatted_system_prompt (str): The system prompt to be sent to the LLM.
        formatted_prompt (str): The main user prompt for the LLM.
        batch_file_path_details (str): String containing details for constructing batch-specific file paths.
        model_source (str): The source of the LLM (e.g., "Gemini", "Azure/OpenAI", "AWS Bedrock", "Local").
        model_choice (str): The specific model chosen (e.g., "gemini-pro", "gpt-4", "anthropic.claude-v2").
        in_api_key (str): API key for the chosen model source (if applicable).
        temperature (float): The sampling temperature for the LLM, controlling randomness.
        max_tokens (int): The maximum number of tokens to generate in the LLM's response.
        azure_api_key_textbox (str): API key for Azure OpenAI (if `model_source` is Azure/OpenAI).
        azure_endpoint_textbox (str): Endpoint URL for Azure OpenAI (if `model_source` is Azure/OpenAI).
        reasoning_suffix (str): Additional text to append to the system prompt for reasoning (primarily for local models).
        local_model (object): The loaded local model object (if `model_source` is Local).
        tokenizer (object): The tokenizer object for local models.
        bedrock_runtime (object): AWS Bedrock runtime client object (if `model_source` is AWS Bedrock).
        reported_batch_no (int): The current batch number being processed and reported.
        response_text (str): The raw text response from the LLM (can be pre-filled or from a previous step).
        whole_conversation (list): A list representing the entire conversation history.
        all_metadata_content (list): Metadata associated with each turn in the conversation.
        start_row (int): The starting row index of the current batch in the original dataset.
        end_row (int): The ending row index of the current batch in the original dataset.
        model_choice_clean (str): A cleaned, short name for the chosen model.
        log_files_output_paths (list): A list of paths to log files generated during processing.
        existing_reference_df (pd.DataFrame): DataFrame containing existing reference data.
        existing_topic_summary_df (pd.DataFrame): DataFrame containing existing topic summary data.
        batch_size (int): The number of items processed in each batch.
        batch_basic_response_df (pd.DataFrame): DataFrame containing basic responses for the current batch.
        group_name (str): The name of the group associated with the current batch.
        produce_structured_summary_radio (str): Indicates whether to produce structured summaries ("Yes" or "No").
        output_folder (str): The directory where all output files will be saved.
        output_debug_files (str): Flag indicating whether to output debug files ("Yes" or "No").
        task_type (str): The type of task being performed (e.g., "topic_extraction", "summarisation").
        assistant_prefill (str, optional): Optional prefill text for the assistant's response. Defaults to "".

    Returns:
        tuple: A tuple containing various output paths and DataFrames after processing the batch:
            - topic_table_out_path (str): Path to the output CSV for the topic table.
            - reference_table_out_path (str): Path to the output CSV for the reference table.
            - topic_summary_df_out_path (str): Path to the output CSV for the topic summary DataFrame.
            - new_topic_df (pd.DataFrame): DataFrame of newly extracted topics.
            - new_reference_df (pd.DataFrame): DataFrame of newly extracted references.
            - new_topic_summary_df (pd.DataFrame): DataFrame of the updated topic summary.
            - batch_file_path_details (str): The batch file path details used.
            - is_error (bool): True if an error occurred during processing, False otherwise.
    """
    client = list()
    client_config = dict()

    # Prepare clients before query
    if "Gemini" in model_source:
        print("Using Gemini model:", model_choice)
        client, client_config = construct_gemini_generative_model(
            in_api_key=in_api_key,
            temperature=temperature,
            model_choice=model_choice,
            system_prompt=formatted_system_prompt,
            max_tokens=max_tokens,
        )
    elif "Azure/OpenAI" in model_source:
        print("Using Azure/OpenAI AI Inference model:", model_choice)
        if azure_api_key_textbox:
            os.environ["AZURE_INFERENCE_CREDENTIAL"] = azure_api_key_textbox
        client, client_config = construct_azure_client(
            in_api_key=azure_api_key_textbox, endpoint=azure_endpoint_textbox
        )
    elif "anthropic.claude" in model_choice:
        print("Using AWS Bedrock model:", model_choice)
        pass
    else:
        print("Using local model:", model_choice)
        pass

    batch_prompts = [formatted_prompt]

    if "Local" in model_source and reasoning_suffix:
        formatted_system_prompt = formatted_system_prompt + "\n" + reasoning_suffix

    # Combine system prompt and user prompt for token counting
    full_input_text = formatted_system_prompt + "\n" + formatted_prompt

    # Count tokens in the input text
    from tools.dedup_summaries import count_tokens_in_text

    input_token_count = count_tokens_in_text(full_input_text, tokenizer, model_source)

    # Check if input exceeds context length
    if input_token_count > LLM_CONTEXT_LENGTH:
        error_message = f"Input text exceeds LLM context length. Input tokens: {input_token_count}, Max context length: {LLM_CONTEXT_LENGTH}. Please reduce the input text size."
        print(error_message)
        raise ValueError(error_message)

    print(f"Input token count: {input_token_count} (Max: {LLM_CONTEXT_LENGTH})")

    conversation_history = list()
    whole_conversation = list()

    # Process requests to large language model
    (
        responses,
        conversation_history,
        whole_conversation,
        all_metadata_content,
        response_text,
    ) = call_llm_with_markdown_table_checks(
        batch_prompts,
        formatted_system_prompt,
        conversation_history,
        whole_conversation,
        all_metadata_content,
        client,
        client_config,
        model_choice,
        temperature,
        reported_batch_no,
        local_model,
        tokenizer,
        bedrock_runtime,
        model_source,
        MAX_OUTPUT_VALIDATION_ATTEMPTS,
        assistant_prefill=assistant_prefill,
        master=not is_first_batch,
        api_url=api_url,
    )

    # print("Response text:", response_text)

    # Return output tables
    (
        topic_table_out_path,
        reference_table_out_path,
        topic_summary_df_out_path,
        new_topic_df,
        new_reference_df,
        new_topic_summary_df,
        batch_file_path_details,
        is_error,
    ) = write_llm_output_and_logs(
        response_text,
        whole_conversation,
        all_metadata_content,
        batch_file_path_details,
        start_row,
        end_row,
        model_choice_clean,
        log_files_output_paths,
        existing_reference_df,
        existing_topic_summary_df,
        batch_size,
        batch_basic_response_df,
        group_name,
        produce_structured_summary_radio,
        output_folder=output_folder,
    )

    # If error in table parsing, leave function
    if is_error is True:
        if is_first_batch:
            raise Exception("Error in output table parsing")
        else:
            final_message_out = "Could not complete summary, error in LLM output."
            raise Exception(final_message_out)

    # Write final output to text file and objects for logging purposes
    full_prompt = formatted_system_prompt + "\n" + formatted_prompt

    (
        current_prompt_content_logged,
        current_summary_content_logged,
        current_conversation_content_logged,
        current_metadata_content_logged,
    ) = process_debug_output_iteration(
        output_debug_files,
        output_folder,
        batch_file_path_details,
        model_choice_clean,
        full_prompt,
        response_text,
        whole_conversation,
        all_metadata_content,
        log_files_output_paths,
        task_type=task_type,
    )

    print("Finished processing batch with LLM")

    return (
        new_topic_df,
        new_reference_df,
        new_topic_summary_df,
        is_error,
        current_prompt_content_logged,
        current_summary_content_logged,
        current_conversation_content_logged,
        current_metadata_content_logged,
        topic_table_out_path,
        reference_table_out_path,
        topic_summary_df_out_path,
    )


def extract_topics(
    in_data_file: gr.FileData,
    file_data: pd.DataFrame,
    existing_topics_table: pd.DataFrame,
    existing_reference_df: pd.DataFrame,
    existing_topic_summary_df: pd.DataFrame,
    unique_table_df_display_table_markdown: str,
    file_name: str,
    num_batches: int,
    in_api_key: str,
    temperature: float,
    chosen_cols: List[str],
    model_choice: str,
    candidate_topics: gr.FileData = None,
    latest_batch_completed: int = 0,
    out_message: List = list(),
    out_file_paths: List = list(),
    log_files_output_paths: List = list(),
    first_loop_state: bool = False,
    all_metadata_content_str: str = "",
    initial_table_prompt: str = initial_table_prompt,
    initial_table_system_prompt: str = initial_table_system_prompt,
    add_existing_topics_system_prompt: str = add_existing_topics_system_prompt,
    add_existing_topics_prompt: str = add_existing_topics_prompt,
    number_of_prompts_used: int = 1,
    batch_size: int = 5,
    context_textbox: str = "",
    time_taken: float = 0,
    sentiment_checkbox: str = "Negative, Neutral, or Positive",
    force_zero_shot_radio: str = "No",
    in_excel_sheets: List[str] = list(),
    force_single_topic_radio: str = "No",
    output_folder: str = OUTPUT_FOLDER,
    force_single_topic_prompt: str = force_single_topic_prompt,
    group_name: str = "All",
    produce_structured_summary_radio: str = "No",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    hf_api_key_textbox: str = "",
    azure_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    max_tokens: int = max_tokens,
    model_name_map: dict = model_name_map,
    existing_logged_content: list = list(),
    max_time_for_loop: int = max_time_for_loop,
    CHOSEN_LOCAL_MODEL_TYPE: str = CHOSEN_LOCAL_MODEL_TYPE,
    reasoning_suffix: str = reasoning_suffix,
    output_debug_files: str = output_debug_files,
    model: object = list(),
    tokenizer: object = list(),
    assistant_model: object = list(),
    max_rows: int = max_rows,
    original_full_file_name: str = "",
    additional_instructions_summary_format: str = "",
    additional_validation_issues_provided: str = "",
    api_url: str = None,
    progress=Progress(track_tqdm=True),
):
    """
    Query an LLM (local, (Gemma/GPT-OSS if local, Gemini, AWS Bedrock or Azure/OpenAI AI Inference) with up to three prompts about a table of open text data. Up to 'batch_size' rows will be queried at a time.

    Parameters:
    - in_data_file (gr.File): Gradio file object containing input data
    - file_data (pd.DataFrame): Pandas dataframe containing the consultation response data.
    - existing_topics_table (pd.DataFrame): Pandas dataframe containing the latest master topic table that has been iterated through batches.
    - existing_reference_df (pd.DataFrame): Pandas dataframe containing the list of Response reference numbers alongside the derived topics and subtopics.
    - existing_topic_summary_df (pd.DataFrame): Pandas dataframe containing the unique list of topics, subtopics, sentiment and summaries until this point.
    - unique_table_df_display_table_markdown (str): Table for display in markdown format.
    - file_name (str): File name of the data file.
    - num_batches (int): Number of batches required to go through all the response rows.
    - in_api_key (str): The API key for authentication (Google Gemini).
    - temperature (float): The temperature parameter for the model.
    - chosen_cols (List[str]): A list of chosen columns to process.
    - candidate_topics (gr.FileData): File with a table of existing candidate topics files submitted by the user.
    - model_choice (str): The choice of model to use.
    - latest_batch_completed (int): The index of the latest file completed.
    - out_message (list): A list to store output messages.
    - out_file_paths (list): A list to store output file paths.
    - log_files_output_paths (list): A list to store log file output paths.
    - first_loop_state (bool): A flag indicating the first loop state.
    - all_metadata_content_str (str): A string to store whole conversation metadata.
    - initial_table_prompt (str): The first prompt for the model.
    - initial_table_system_prompt (str): The system prompt for the model.
    - add_existing_topics_system_prompt (str): The system prompt for the summary part of the model.
    - add_existing_topics_prompt (str): The prompt for the model summary.
    - number of requests (int): The number of prompts to send to the model.
    - batch_size (int): The number of data rows to consider in each request.
    - context_textbox (str, optional): A string giving some context to the consultation/task.
    - time_taken (float, optional): The amount of time taken to process the responses up until this point.
    - sentiment_checkbox (str, optional): What type of sentiment analysis should the topic modeller do?
    - force_zero_shot_radio (str, optional): Should responses be forced into a zero shot topic or not.
    - in_excel_sheets (List[str], optional): List of excel sheets to load from input file.
    - force_single_topic_radio (str, optional): Should the model be forced to assign only one single topic to each response (effectively a classifier).
    - produce_structured_summary_radio (str, optional): Should the model create a structured summary instead of extracting topics.
    - output_folder (str, optional): Output folder where results will be stored.
    - force_single_topic_prompt (str, optional): The prompt for forcing the model to assign only one single topic to each response.
    - aws_access_key_textbox (str, optional): AWS access key for account with Bedrock permissions.
    - aws_secret_key_textbox (str, optional): AWS secret key for account with Bedrock permissions.
    - hf_api_key_textbox (str, optional): Hugging Face API key for account with Hugging Face permissions.
    - max_tokens (int): The maximum number of tokens for the model.
    - model_name_map (dict, optional): A dictionary mapping full model name to shortened.
    - existing_logged_content (list, optional): A list of existing logged content.
    - max_time_for_loop (int, optional): The number of seconds maximum that the function should run for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there).
    - CHOSEN_LOCAL_MODEL_TYPE (str, optional): The name of the chosen local model.
    - reasoning_suffix (str, optional): The suffix for the reasoning system prompt.
    - output_debug_files (str, optional): Flag indicating whether to output debug files ("True" or "False").
    - model: Model object for local inference.
    - tokenizer: Tokenizer object for local inference.
    - assistant_model: Assistant model object for local inference.
    - max_rows: The maximum number of rows to process.
    - original_full_file_name: The original full file name.
    - additional_instructions_summary_format: Initial instructions to guide the format for the initial summary of the topics.
    - additional_validation_issues_provided: Additional validation issues provided by the user.
    - progress (Progress): A progress tracker.

    """

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    tic = time.perf_counter()

    final_time = 0.0
    all_metadata_content = list()
    create_revised_general_topics = False
    local_model = None
    tokenizer = None
    zero_shot_topics_df = pd.DataFrame()
    missing_df = pd.DataFrame()
    new_reference_df = pd.DataFrame(
        columns=[
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
            "Start row of group",
            "Group",
            "Topic number",
            "Summary",
        ]
    )
    new_topic_summary_df = pd.DataFrame(
        columns=[
            "General topic",
            "Subtopic",
            "Sentiment",
            "Group",
            "Number of responses",
            "Summary",
        ]
    )
    if existing_topic_summary_df.empty:
        existing_topic_summary_df = pd.DataFrame(
            columns=[
                "General topic",
                "Subtopic",
                "Sentiment",
                "Group",
                "Number of responses",
                "Summary",
            ]
        )
    if existing_reference_df.empty:
        existing_reference_df = pd.DataFrame(
            columns=[
                "Response References",
                "General topic",
                "Subtopic",
                "Sentiment",
                "Start row of group",
                "Group",
                "Topic number",
                "Summary",
            ]
        )
    new_topic_df = pd.DataFrame(
        columns=[
            "General topic",
            "Subtopic",
            "Sentiment",
            "Group",
            "Number of responses",
            "Summary",
        ]
    )
    pd.DataFrame(
        columns=[
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
            "Start row of group",
            "Group",
            "Topic number",
            "Summary",
        ]
    )
    pd.DataFrame(
        columns=[
            "General topic",
            "Subtopic",
            "Sentiment",
            "Group",
            "Number of responses",
            "Summary",
        ]
    )
    task_type = "Topic extraction"

    # Logged content
    all_prompts_content = list()
    all_responses_content = list()
    all_conversation_content = list()
    all_metadata_content = list()
    all_groups_content = list()
    all_batches_content = list()
    all_model_choice_content = list()
    all_validated_content = list()
    all_task_type_content = list()
    all_file_names_content = list()
    all_groups_logged_content = list()
    # Need to reduce output file names as full length files may be too long
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )
    in_column_cleaned = clean_column_name(chosen_cols, max_length=20)
    file_name_clean = clean_column_name(
        file_name, max_length=20, front_characters=False
    )

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            (
                in_colnames_drop,
                in_excel_sheets,
                file_name,
                join_colnames,
                join_colnames_drop,
            ) = put_columns_in_df(in_data_file)
            file_data, file_name, num_batches = load_in_data_file(
                in_data_file, chosen_cols, batch_size_default, in_excel_sheets
            )
        except Exception as e:
            out_message = "Could not load in data file due to: " + str(e)
            print(out_message)
            raise Exception(out_message)

    if file_data.shape[0] > max_rows:
        out_message = (
            "Your data has more than "
            + str(max_rows)
            + " rows, which has been set as the maximum in the application configuration."
        )
        print(out_message)
        raise Exception(out_message)

    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_source = model_name_map[model_choice]["source"]

    bedrock_runtime = connect_to_bedrock_runtime(
        model_name_map, model_choice, aws_access_key_textbox, aws_secret_key_textbox
    )

    # If this is the first time around, set variables to 0/blank
    if first_loop_state is True:
        if (latest_batch_completed == 999) | (latest_batch_completed == 0):
            latest_batch_completed = 0
            out_message = list()
            out_file_paths = list()
            final_time = 0

            if (model_source == "Local") & (RUN_LOCAL_MODEL == "1") & (not model):
                progress(0.1, f"Using local model: {model_choice_clean}")
                local_model = get_model()
                tokenizer = get_tokenizer()
                get_assistant_model()

    if num_batches > 0:
        progress_measure = round(latest_batch_completed / num_batches, 1)
        progress(progress_measure, desc="Querying large language model")
    else:
        progress(0.1, desc="Querying large language model")

    latest_batch_completed = int(latest_batch_completed)
    num_batches = int(num_batches)

    if latest_batch_completed < num_batches:

        # Load file
        # If out message or out_file_paths are blank, change to a list so it can be appended to
        if isinstance(out_message, str):
            out_message = [out_message]

        if not out_file_paths:
            out_file_paths = list()

        if "anthropic.claude-3-sonnet" in model_choice and file_data.shape[1] > 300:
            out_message = "Your data has more than 300 rows, using the Sonnet model will be too expensive. Please choose the Haiku model instead."
            print(out_message)
            raise Exception(out_message)

        sentiment_prefix = "In the next column named 'Sentiment', "
        sentiment_suffix = "."
        if sentiment_checkbox == "Negative, Neutral, or Positive":
            sentiment_prompt = (
                sentiment_prefix
                + negative_neutral_positive_sentiment_prompt
                + sentiment_suffix
            )
        elif sentiment_checkbox == "Negative or Positive":
            sentiment_prompt = (
                sentiment_prefix
                + negative_or_positive_sentiment_prompt
                + sentiment_suffix
            )
        elif sentiment_checkbox == "Do not assess sentiment":
            sentiment_prompt = ""  # Just remove line completely. Previous: sentiment_prefix + do_not_assess_sentiment_prompt + sentiment_suffix
        else:
            sentiment_prompt = (
                sentiment_prefix + default_sentiment_prompt + sentiment_suffix
            )

        if context_textbox and "The context of this analysis is" not in context_textbox:
            context_textbox = (
                "The context of this analysis is '" + context_textbox + "'."
            )

        topics_loop_description = (
            "Extracting topics from response batches (each batch of "
            + str(batch_size)
            + " responses)."
        )
        total_batches_to_do = num_batches - latest_batch_completed
        topics_loop = progress.tqdm(
            range(total_batches_to_do),
            desc=topics_loop_description,
            unit="batches remaining",
        )

        for i in topics_loop:
            reported_batch_no = latest_batch_completed + 1
            print("Running response batch:", reported_batch_no)

            # Call the function to prepare the input table
            (
                simplified_csv_table_path,
                normalised_simple_markdown_table,
                start_row,
                end_row,
                batch_basic_response_df,
            ) = data_file_to_markdown_table(
                file_data, file_name, chosen_cols, latest_batch_completed, batch_size
            )

            if batch_basic_response_df.shape[0] == 1:
                response_reference_format = ""  # Blank, as the topics will always refer to the single response provided, '1'
            else:
                response_reference_format = "\n" + default_response_reference_format

            # If the response table is not empty, add it to the prompt with an intro line
            if normalised_simple_markdown_table:
                response_table_prompt = (
                    "Response table:\n" + normalised_simple_markdown_table
                )
            else:
                response_table_prompt = ""

            existing_topic_summary_df.rename(
                columns={"Main heading": "General topic", "Subheading": "Subtopic"},
                inplace=True,
                errors="ignore",
            )
            existing_reference_df.rename(
                columns={"Main heading": "General topic", "Subheading": "Subtopic"},
                inplace=True,
                errors="ignore",
            )

            # If the latest batch of responses contains at least one instance of text
            if not batch_basic_response_df.empty:

                # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
                if latest_batch_completed >= 1 or candidate_topics is not None:

                    formatted_system_prompt = add_existing_topics_system_prompt.format(
                        consultation_context=context_textbox, column_name=chosen_cols
                    )

                    # Preparing candidate topics if no topics currently exist
                    if candidate_topics and existing_topic_summary_df.empty:

                        # 'Zero shot topics' are those supplied by the user
                        # Handle both string paths (CLI) and gr.FileData objects (Gradio)
                        # Supports CSV, Excel (.xlsx), and Parquet files
                        candidate_topics_path = (
                            candidate_topics
                            if isinstance(candidate_topics, str)
                            else getattr(candidate_topics, "name", None)
                        )
                        if candidate_topics_path is None:
                            raise ValueError(
                                "candidate_topics must be a file path string or a FileData object with a 'name' attribute"
                            )

                        # Read the file (supports CSV, Excel .xlsx, and Parquet)
                        # For Excel files, reads the first sheet by default
                        try:
                            zero_shot_topics = read_file(candidate_topics_path)
                        except Exception as e:
                            raise ValueError(
                                f"Error reading candidate topics file '{candidate_topics_path}': {str(e)}. "
                                f"Supported formats: CSV (.csv), Excel (.xlsx), and Parquet (.parquet). "
                                f"For Excel files, the first sheet will be used."
                            ) from e

                        zero_shot_topics = zero_shot_topics.fillna(
                            ""
                        )  # Replace NaN with empty string
                        zero_shot_topics = zero_shot_topics.astype(str)

                        zero_shot_topics_df = generate_zero_shot_topics_df(
                            zero_shot_topics,
                            force_zero_shot_radio,
                            create_revised_general_topics,
                        )

                        # This part concatenates all zero shot and new topics together, so that for the next prompt the LLM will have the full list available
                        if (
                            not existing_topic_summary_df.empty
                            and force_zero_shot_radio != "Yes"
                        ):
                            existing_topic_summary_df = pd.concat(
                                [existing_topic_summary_df, zero_shot_topics_df]
                            ).drop_duplicates("Subtopic")
                        else:
                            existing_topic_summary_df = zero_shot_topics_df

                    if candidate_topics and not zero_shot_topics_df.empty:
                        # If you have already created revised zero shot topics, concat to the current
                        existing_topic_summary_df = pd.concat(
                            [existing_topic_summary_df, zero_shot_topics_df]
                        )

                    existing_topic_summary_df["Number of responses"] = ""
                    existing_topic_summary_df.fillna("", inplace=True)
                    existing_topic_summary_df["General topic"] = (
                        existing_topic_summary_df["General topic"].str.replace(
                            "(?i)^Nan$", "", regex=True
                        )
                    )
                    existing_topic_summary_df["Subtopic"] = existing_topic_summary_df[
                        "Subtopic"
                    ].str.replace("(?i)^Nan$", "", regex=True)
                    existing_topic_summary_df = (
                        existing_topic_summary_df.drop_duplicates()
                    )

                    # If user has chosen to try to force zero shot topics, then the prompt is changed to ask the model not to deviate at all from submitted topic list.
                    keep_cols = [
                        col
                        for col in ["General topic", "Subtopic", "Description"]
                        if col in existing_topic_summary_df.columns
                        and not existing_topic_summary_df[col]
                        .replace(r"^\s*$", pd.NA, regex=True)
                        .isna()
                        .all()
                    ]

                    # Create topics table to be presented to LLM
                    topics_df_for_markdown = existing_topic_summary_df[
                        keep_cols
                    ].drop_duplicates(keep_cols)
                    if (
                        "General topic" in topics_df_for_markdown.columns
                        and "Subtopic" in topics_df_for_markdown.columns
                    ):
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["General topic", "Subtopic"]
                        )

                    if "Description" in existing_topic_summary_df:
                        if existing_topic_summary_df["Description"].isnull().all():
                            existing_topic_summary_df.drop(
                                "Description", axis=1, inplace=True
                            )

                    if produce_structured_summary_radio == "Yes":
                        if "General topic" in topics_df_for_markdown.columns:
                            topics_df_for_markdown.rename(
                                columns={"General topic": "Main heading"},
                                inplace=True,
                                errors="ignore",
                            )
                        if "Subtopic" in topics_df_for_markdown.columns:
                            topics_df_for_markdown.rename(
                                columns={"Subtopic": "Subheading"},
                                inplace=True,
                                errors="ignore",
                            )

                    # Remove duplicate General topic and subtopic names, prioritising topics where a general topic is provided
                    if (
                        "General topic" in topics_df_for_markdown.columns
                        and "Subtopic" in topics_df_for_markdown.columns
                    ):
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["General topic", "Subtopic"], ascending=[False, True]
                        )
                        topics_df_for_markdown = topics_df_for_markdown.drop_duplicates(
                            ["General topic", "Subtopic"], keep="first"
                        )
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["General topic", "Subtopic"], ascending=[True, True]
                        )
                    elif "Subtopic" in topics_df_for_markdown.columns:
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["Subtopic"], ascending=[True]
                        )
                        topics_df_for_markdown = topics_df_for_markdown.drop_duplicates(
                            ["Subtopic"], keep="first"
                        )
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["Subtopic"], ascending=[True]
                        )
                    elif (
                        "Main heading" in topics_df_for_markdown.columns
                        and "Subheading" in topics_df_for_markdown.columns
                    ):
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["Main heading", "Subheading"], ascending=[True, True]
                        )
                        topics_df_for_markdown = topics_df_for_markdown.drop_duplicates(
                            ["Main heading", "Subheading"], keep="first"
                        )
                        topics_df_for_markdown = topics_df_for_markdown.sort_values(
                            ["Main heading", "Subheading"], ascending=[True, True]
                        )

                    unique_topics_markdown = topics_df_for_markdown.to_markdown(
                        index=False
                    )
                    unique_topics_markdown = normalise_string(unique_topics_markdown)

                    if force_zero_shot_radio == "Yes":
                        topic_assignment_prompt = force_existing_topics_prompt
                    else:
                        topic_assignment_prompt = allow_new_topics_prompt

                    # Should the outputs force only one single topic assignment per response?
                    if force_single_topic_radio != "Yes":
                        force_single_topic_prompt = ""
                    else:
                        topic_assignment_prompt = (
                            topic_assignment_prompt.replace(
                                "Assign topics", "Assign a topic"
                            )
                            .replace("assign Subtopics", "assign a Subtopic")
                            .replace("Subtopics", "Subtopic")
                            .replace("Topics", "Topic")
                            .replace("topics", "a topic")
                        )

                    # Format the summary prompt with the response table and topics
                    if produce_structured_summary_radio != "Yes":
                        formatted_summary_prompt = add_existing_topics_prompt.format(
                            validate_prompt_prefix="",
                            response_table=response_table_prompt,
                            topics=unique_topics_markdown,
                            topic_assignment=topic_assignment_prompt,
                            force_single_topic=force_single_topic_prompt,
                            sentiment_choices=sentiment_prompt,
                            response_reference_format=response_reference_format,
                            add_existing_topics_summary_format=additional_instructions_summary_format,
                            previous_table_introduction="",
                            previous_table="",
                            validate_prompt_suffix="",
                        )
                    else:
                        formatted_summary_prompt = structured_summary_prompt.format(
                            response_table=response_table_prompt,
                            topics=unique_topics_markdown,
                            summary_format=additional_instructions_summary_format,
                        )

                    batch_file_path_details = f"{file_name_clean}_batch_{latest_batch_completed + 1}_size_{batch_size}_col_{in_column_cleaned}"

                    # Use the helper function to process the batch
                    (
                        new_topic_df,
                        new_reference_df,
                        new_topic_summary_df,
                        is_error,
                        current_prompt_content_logged,
                        current_summary_content_logged,
                        current_conversation_content_logged,
                        current_metadata_content_logged,
                        topic_table_out_path,
                        reference_table_out_path,
                        topic_summary_df_out_path,
                    ) = process_batch_with_llm(
                        is_first_batch=False,
                        formatted_system_prompt=formatted_system_prompt,
                        formatted_prompt=formatted_summary_prompt,
                        batch_file_path_details=batch_file_path_details,
                        model_source=model_source,
                        model_choice=model_choice,
                        in_api_key=in_api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        azure_api_key_textbox=azure_api_key_textbox,
                        azure_endpoint_textbox=azure_endpoint_textbox,
                        reasoning_suffix=reasoning_suffix,
                        local_model=local_model,
                        tokenizer=tokenizer,
                        bedrock_runtime=bedrock_runtime,
                        reported_batch_no=reported_batch_no,
                        response_text="",
                        whole_conversation=list(),
                        all_metadata_content=list(),
                        start_row=start_row,
                        end_row=end_row,
                        model_choice_clean=model_choice_clean,
                        log_files_output_paths=log_files_output_paths,
                        existing_reference_df=existing_reference_df,
                        existing_topic_summary_df=existing_topic_summary_df,
                        batch_size=batch_size,
                        batch_basic_response_df=batch_basic_response_df,
                        group_name=group_name,
                        produce_structured_summary_radio=produce_structured_summary_radio,
                        output_folder=output_folder,
                        output_debug_files=output_debug_files,
                        task_type=task_type,
                        assistant_prefill=add_existing_topics_assistant_prefill,
                        api_url=api_url,
                    )

                    print("Completed batch processing")

                    all_prompts_content.append(current_prompt_content_logged)
                    all_responses_content.append(current_summary_content_logged)
                    all_conversation_content.append(current_conversation_content_logged)
                    all_metadata_content.append(current_metadata_content_logged)
                    all_groups_content.append(group_name)
                    all_batches_content.append(f"{reported_batch_no}:")
                    all_model_choice_content.append(model_choice_clean_short)
                    all_validated_content.append("No")
                    all_task_type_content.append(task_type)
                    all_file_names_content.append(original_full_file_name)

                    ## Reference table mapping response numbers to topics
                    if output_debug_files == "True":
                        new_reference_df.to_csv(
                            reference_table_out_path, index=None, encoding="utf-8-sig"
                        )
                        out_file_paths.append(reference_table_out_path)

                    ## Unique topic list
                    new_topic_summary_df = pd.concat(
                        [new_topic_summary_df, existing_topic_summary_df]
                    ).drop_duplicates("Subtopic")

                    new_topic_summary_df["Group"] = group_name

                    if output_debug_files == "True":
                        new_topic_summary_df.to_csv(
                            topic_summary_df_out_path, index=None, encoding="utf-8-sig"
                        )
                        out_file_paths.append(topic_summary_df_out_path)

                    # Outputs for markdown table output
                    unique_table_df_display_table = new_topic_summary_df.apply(
                        lambda col: col.map(
                            lambda x: wrap_text(x, max_text_length=max_text_length)
                        )
                    )

                    if produce_structured_summary_radio == "Yes":
                        unique_table_df_display_table = unique_table_df_display_table[
                            ["General topic", "Subtopic", "Summary"]
                        ]
                        unique_table_df_display_table.rename(
                            columns={
                                "General topic": "Main heading",
                                "Subtopic": "Subheading",
                            },
                            inplace=True,
                        )
                    else:
                        unique_table_df_display_table = unique_table_df_display_table[
                            [
                                "General topic",
                                "Subtopic",
                                "Sentiment",
                                "Number of responses",
                                "Summary",
                            ]
                        ]

                    unique_table_df_display_table_markdown = (
                        unique_table_df_display_table.to_markdown(index=False)
                    )

                    all_metadata_content_str = " ".join(all_metadata_content)

                    out_file_paths = [
                        col for col in out_file_paths if str(reported_batch_no) in col
                    ]
                    log_files_output_paths = [
                        col for col in out_file_paths if str(reported_batch_no) in col
                    ]
                # If this is the first batch, run this
                else:
                    formatted_system_prompt = initial_table_system_prompt.format(
                        consultation_context=context_textbox, column_name=chosen_cols
                    )

                    # Format the summary prompt with the response table and topics
                    if produce_structured_summary_radio != "Yes":
                        formatted_initial_table_prompt = initial_table_prompt.format(
                            validate_prompt_prefix="",
                            response_table=response_table_prompt,
                            sentiment_choices=sentiment_prompt,
                            response_reference_format=response_reference_format,
                            add_existing_topics_summary_format=additional_instructions_summary_format,
                            previous_table_introduction="",
                            previous_table="",
                            validate_prompt_suffix="",
                        )
                    else:
                        unique_topics_markdown = (
                            "No suggested headings for this summary"
                        )
                        formatted_initial_table_prompt = (
                            structured_summary_prompt.format(
                                response_table=response_table_prompt,
                                topics=unique_topics_markdown,
                            )
                        )

                    batch_file_path_details = f"{file_name_clean}_batch_{latest_batch_completed + 1}_size_{batch_size}_col_{in_column_cleaned}"

                    # Use the helper function to process the batch
                    (
                        new_topic_df,
                        new_reference_df,
                        new_topic_summary_df,
                        is_error,
                        current_prompt_content_logged,
                        current_summary_content_logged,
                        current_conversation_content_logged,
                        current_metadata_content_logged,
                        topic_table_out_path,
                        reference_table_out_path,
                        topic_summary_df_out_path,
                    ) = process_batch_with_llm(
                        is_first_batch=True,
                        formatted_system_prompt=formatted_system_prompt,
                        formatted_prompt=formatted_initial_table_prompt,
                        batch_file_path_details=batch_file_path_details,
                        model_source=model_source,
                        model_choice=model_choice,
                        in_api_key=in_api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        azure_api_key_textbox=azure_api_key_textbox,
                        azure_endpoint_textbox=azure_endpoint_textbox,
                        reasoning_suffix=reasoning_suffix,
                        local_model=local_model,
                        tokenizer=tokenizer,
                        bedrock_runtime=bedrock_runtime,
                        reported_batch_no=reported_batch_no,
                        response_text="",
                        whole_conversation=list(),
                        all_metadata_content=list(),
                        start_row=start_row,
                        end_row=end_row,
                        model_choice_clean=model_choice_clean,
                        log_files_output_paths=log_files_output_paths,
                        existing_reference_df=existing_reference_df,
                        existing_topic_summary_df=existing_topic_summary_df,
                        batch_size=batch_size,
                        batch_basic_response_df=batch_basic_response_df,
                        group_name=group_name,
                        produce_structured_summary_radio=produce_structured_summary_radio,
                        output_folder=output_folder,
                        output_debug_files=output_debug_files,
                        task_type=task_type,
                        assistant_prefill=initial_table_assistant_prefill,
                        api_url=api_url,
                    )

                    all_prompts_content.append(current_prompt_content_logged)
                    all_responses_content.append(current_summary_content_logged)
                    all_conversation_content.append(current_conversation_content_logged)
                    all_metadata_content.append(current_metadata_content_logged)
                    all_groups_content.append(group_name)
                    all_batches_content.append(f"{reported_batch_no}:")
                    all_model_choice_content.append(model_choice_clean_short)
                    all_validated_content.append("No")
                    all_task_type_content.append(task_type)
                    all_file_names_content.append(original_full_file_name)

                    if output_debug_files == "True":

                        # Output reference table
                        new_reference_df.to_csv(
                            reference_table_out_path, index=None, encoding="utf-8-sig"
                        )
                        out_file_paths.append(reference_table_out_path)

                    ## Unique topic list

                    new_topic_summary_df = pd.concat(
                        [new_topic_summary_df, existing_topic_summary_df]
                    ).drop_duplicates("Subtopic")

                    new_topic_summary_df["Group"] = group_name

                    if output_debug_files == "True":
                        new_topic_summary_df.to_csv(
                            topic_summary_df_out_path, index=None, encoding="utf-8-sig"
                        )
                        out_file_paths.append(topic_summary_df_out_path)

                    all_metadata_content.append(all_metadata_content_str)
                    all_metadata_content_str = ". ".join(all_metadata_content)

            else:
                print(
                    "Current batch of responses contains no text, moving onto next. Batch number:",
                    str(latest_batch_completed + 1),
                    ". Start row:",
                    start_row,
                    ". End row:",
                    end_row,
                )

            # Increase latest file completed count unless we are over the last batch number, then go back around
            num_batches = int(num_batches)
            latest_batch_completed = int(latest_batch_completed)
            if latest_batch_completed <= num_batches:
                latest_batch_completed += 1

            toc = time.perf_counter()
            final_time = toc - tic

            if final_time > max_time_for_loop:
                print("Max time reached, breaking loop.")
                topics_loop.close()
                tqdm._instances.clear()
                break

            # Overwrite 'existing' elements to add new tables
            existing_reference_df = new_reference_df.dropna(how="all")
            existing_topic_summary_df = new_topic_summary_df.dropna(how="all")
            existing_topics_table = new_topic_df.dropna(how="all")

            # The topic table that can be modified does not need the summary column
            modifiable_topic_summary_df = existing_topic_summary_df.drop(
                "Summary", axis=1
            )

        out_time = f"{final_time:0.1f} seconds."

        out_message.append("All queries successfully completed in")

        final_message_out = "\n".join(out_message)
        final_message_out = final_message_out + " " + out_time

        print(final_message_out)

    # If we have extracted topics from the last batch, return the input out_message and file list to the relevant components
    if latest_batch_completed >= num_batches:

        group_combined_logged_content = [
            {
                "prompt": prompt,
                "response": summary,
                "metadata": metadata,
                "batch": batch,
                "model_choice": model_choice,
                "validated": validated,
                "group": group,
                "task_type": task_type,
                "file_name": file_name,
            }
            for prompt, summary, metadata, batch, model_choice, validated, group, task_type, file_name in zip(
                all_prompts_content,
                all_responses_content,
                all_metadata_content,
                all_batches_content,
                all_model_choice_content,
                all_validated_content,
                all_groups_content,
                all_task_type_content,
                all_file_names_content,
            )
        ]

        # VALIDATION LOOP - Run validation if enabled
        if ENABLE_VALIDATION == "True":

            # Use the standalone validation function
            (
                existing_reference_df,
                existing_topic_summary_df,
                group_combined_logged_content,
                validation_conversation_metadata_str,
            ) = validate_topics(
                file_data=file_data,
                reference_df=existing_reference_df,
                topic_summary_df=existing_topic_summary_df,
                file_name=file_name,
                chosen_cols=chosen_cols,
                batch_size=batch_size,
                model_choice=model_choice,
                in_api_key=in_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                azure_api_key_textbox=azure_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                reasoning_suffix=reasoning_suffix,
                group_name=group_name,
                produce_structured_summary_radio=produce_structured_summary_radio,
                force_zero_shot_radio=force_zero_shot_radio,
                force_single_topic_radio=force_single_topic_radio,
                context_textbox=context_textbox,
                additional_instructions_summary_format=additional_instructions_summary_format,
                additional_validation_issues_provided=additional_validation_issues_provided,
                output_folder=output_folder,
                output_debug_files=output_debug_files,
                original_full_file_name=original_full_file_name,
                max_time_for_loop=max_time_for_loop,
                sentiment_checkbox=sentiment_checkbox,
                logged_content=group_combined_logged_content,
                api_url=api_url,
            )

            # Add validation conversation metadata to the main conversation metadata
            if validation_conversation_metadata_str:
                all_metadata_content_str = (
                    all_metadata_content_str
                    + ". "
                    + validation_conversation_metadata_str
                )

        print("Last batch reached, returning batch:", str(latest_batch_completed))

        join_file_paths = list()

        toc = time.perf_counter()
        final_time = (toc - tic) + time_taken
        out_time = f"Everything finished in {round(final_time,1)} seconds."
        print(out_time)

        print("All batches completed. Exporting outputs.")

        all_groups_logged_content = (
            all_groups_logged_content + group_combined_logged_content
        )

        # file_path_details = create_batch_file_path_details(file_name, in_column=chosen_cols)

        # Create a pivoted reference table
        existing_reference_df_pivot = convert_reference_table_to_pivot_table(
            existing_reference_df
        )

        # Save the new DataFrame to CSV
        reference_table_out_pivot_path = (
            output_folder
            + file_name_clean
            + "_final_reference_table_pivot_"
            + model_choice_clean_short
            + "_temp_"
            + str(temperature)
            + ".csv"
        )
        reference_table_out_path = (
            output_folder
            + file_name_clean
            + "_final_reference_table_"
            + model_choice_clean_short
            + "_temp_"
            + str(temperature)
            + ".csv"
        )
        topic_summary_df_out_path = (
            output_folder
            + file_name_clean
            + "_final_unique_topics_"
            + model_choice_clean_short
            + "_temp_"
            + str(temperature)
            + ".csv"
        )
        basic_response_data_out_path = (
            output_folder
            + file_name_clean
            + "_simplified_data_file_"
            + model_choice_clean_short
            + "_temp_"
            + str(temperature)
            + ".csv"
        )

        ## Reference table mapping response numbers to topics
        existing_reference_df.to_csv(
            reference_table_out_path, index=None, encoding="utf-8-sig"
        )
        out_file_paths.append(reference_table_out_path)
        join_file_paths.append(reference_table_out_path)

        # Create final unique topics table from reference table to ensure consistent numbers
        final_out_topic_summary_df = create_topic_summary_df_from_reference_table(
            existing_reference_df
        )
        final_out_topic_summary_df["Group"] = group_name

        ## Unique topic list
        final_out_topic_summary_df.to_csv(
            topic_summary_df_out_path, index=None, encoding="utf-8-sig"
        )
        out_file_paths.append(topic_summary_df_out_path)

        # Outputs for markdown table output
        unique_table_df_display_table = final_out_topic_summary_df.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )

        if produce_structured_summary_radio == "Yes":
            unique_table_df_display_table = unique_table_df_display_table[
                ["General topic", "Subtopic", "Summary"]
            ]
            unique_table_df_display_table.rename(
                columns={"General topic": "Main heading", "Subtopic": "Subheading"},
                inplace=True,
            )
        else:
            unique_table_df_display_table = unique_table_df_display_table[
                [
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                    "Number of responses",
                    "Summary",
                ]
            ]

        unique_table_df_display_table_markdown = (
            unique_table_df_display_table.to_markdown(index=False)
        )

        # Ensure that we are only returning the final results to outputs
        out_file_paths = [x for x in out_file_paths if "_final_" in x]

        ## Reference table mapping response numbers to topics
        existing_reference_df_pivot["Group"] = group_name
        existing_reference_df_pivot.to_csv(
            reference_table_out_pivot_path, index=None, encoding="utf-8-sig"
        )
        log_files_output_paths.append(reference_table_out_pivot_path)

        ## Create a dataframe for missing response references:
        # Assuming existing_reference_df and file_data are already defined
        # Simplify table to just responses column and the Response reference number
        basic_response_data = get_basic_response_data(file_data, chosen_cols)

        # Save simplified file data to log outputs
        pd.DataFrame(basic_response_data).to_csv(
            basic_response_data_out_path, index=None, encoding="utf-8-sig"
        )
        log_files_output_paths.append(basic_response_data_out_path)

        # Note: missing_df creation moved to wrapper functions to handle grouped processing correctly
        missing_df = pd.DataFrame()

        out_file_paths = list(set(out_file_paths))
        log_files_output_paths = list(set(log_files_output_paths))

        final_out_file_paths = [
            file_path for file_path in out_file_paths if "final_" in file_path
        ]

        # The topic table that can be modified does not need the summary column
        modifiable_topic_summary_df = final_out_topic_summary_df.drop("Summary", axis=1)

        return (
            unique_table_df_display_table_markdown,
            existing_topics_table,
            final_out_topic_summary_df,
            existing_reference_df,
            final_out_file_paths,
            final_out_file_paths,
            latest_batch_completed,
            log_files_output_paths,
            log_files_output_paths,
            all_metadata_content_str,
            final_time,
            final_out_file_paths,
            final_out_file_paths,
            modifiable_topic_summary_df,
            final_out_file_paths,
            join_file_paths,
            existing_reference_df_pivot,
            missing_df,
            all_groups_logged_content,
        )

    return (
        unique_table_df_display_table_markdown,
        existing_topics_table,
        existing_topic_summary_df,
        existing_reference_df,
        out_file_paths,
        out_file_paths,
        latest_batch_completed,
        log_files_output_paths,
        log_files_output_paths,
        all_metadata_content_str,
        final_time,
        out_file_paths,
        out_file_paths,
        modifiable_topic_summary_df,
        out_file_paths,
        join_file_paths,
        existing_reference_df_pivot,
        missing_df,
        all_groups_logged_content,
    )


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def wrapper_extract_topics_per_column_value(
    grouping_col: str,
    in_data_file: Any,
    file_data: pd.DataFrame,
    initial_existing_topics_table: pd.DataFrame,
    initial_existing_reference_df: pd.DataFrame,
    initial_existing_topic_summary_df: pd.DataFrame,
    initial_unique_table_df_display_table_markdown: str,
    original_file_name: str,  # Original file name, to be modified per segment
    total_number_of_batches: int,
    in_api_key: str,
    temperature: float,
    chosen_cols: List[str],
    model_choice: str,
    candidate_topics: gr.FileData = None,
    initial_first_loop_state: bool = True,
    initial_all_metadata_content_str: str = "",
    initial_latest_batch_completed: int = 0,
    initial_time_taken: float = 0,
    initial_table_prompt: str = initial_table_prompt,
    initial_table_system_prompt: str = initial_table_system_prompt,
    add_existing_topics_system_prompt: str = add_existing_topics_system_prompt,
    add_existing_topics_prompt: str = add_existing_topics_prompt,
    number_of_prompts_used: int = 1,
    batch_size: int = 50,  # Crucial for calculating num_batches per segment
    context_textbox: str = "",
    sentiment_checkbox: str = "Negative, Neutral, or Positive",
    force_zero_shot_radio: str = "No",
    in_excel_sheets: List[str] = list(),
    force_single_topic_radio: str = "No",
    produce_structured_summary_radio: str = "No",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    hf_api_key_textbox: str = "",
    azure_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    output_folder: str = OUTPUT_FOLDER,
    existing_logged_content: list = list(),
    additional_instructions_summary_format: str = "",
    additional_validation_issues_provided: str = "",
    show_previous_table: str = "Yes",
    force_single_topic_prompt: str = force_single_topic_prompt,
    max_tokens: int = max_tokens,
    model_name_map: dict = model_name_map,
    max_time_for_loop: int = max_time_for_loop,  # This applies per call to extract_topics
    reasoning_suffix: str = reasoning_suffix,
    CHOSEN_LOCAL_MODEL_TYPE: str = CHOSEN_LOCAL_MODEL_TYPE,
    output_debug_files: str = output_debug_files,
    model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    max_rows: int = max_rows,
    api_url: str = None,
    progress=Progress(track_tqdm=True),  # type: ignore
) -> Tuple:  # Mimicking the return tuple structure of extract_topics
    """
    A wrapper function that iterates through unique values in a specified grouping column
    and calls the `extract_topics` function for each segment of the data.
    It accumulates results from each call and returns a consolidated output.

    :param grouping_col: The name of the column to group the data by.
    :param in_data_file: The input data file object (e.g., Gradio FileData).
    :param file_data: The full DataFrame containing all data.
    :param initial_existing_topics_table: Initial DataFrame of existing topics.
    :param initial_existing_reference_df: Initial DataFrame mapping responses to topics.
    :param initial_existing_topic_summary_df: Initial DataFrame summarizing topics.
    :param initial_unique_table_df_display_table_markdown: Initial markdown string for topic display.
    :param original_file_name: The original name of the input file.
    :param total_number_of_batches: The total number of batches across all data.
    :param in_api_key: API key for the chosen LLM.
    :param temperature: Temperature setting for the LLM.
    :param chosen_cols: List of columns from `file_data` to be processed.
    :param model_choice: The chosen LLM model (e.g., "Gemini", "AWS Claude").
    :param candidate_topics: Optional Gradio FileData for candidate topics (zero-shot).
    :param initial_first_loop_state: Boolean indicating if this is the very first loop iteration.
    :param initial_all_metadata_content_str: Initial metadata string for the whole conversation.
    :param initial_latest_batch_completed: The batch number completed in the previous run.
    :param initial_time_taken: Initial time taken for processing.
    :param initial_table_prompt: The initial prompt for table summarization.
    :param initial_table_system_prompt: The initial system prompt for table summarization.
    :param add_existing_topics_system_prompt: System prompt for adding existing topics.
    :param add_existing_topics_prompt: Prompt for adding existing topics.
    :param number_of_prompts_used: Number of prompts used in the LLM call.
    :param batch_size: Number of rows to process in each batch for the LLM.
    :param context_textbox: Additional context provided by the user.
    :param sentiment_checkbox: Choice for sentiment assessment (e.g., "Negative, Neutral, or Positive").
    :param force_zero_shot_radio: Option to force responses into zero-shot topics.
    :param in_excel_sheets: List of Excel sheet names if applicable.
    :param force_single_topic_radio: Option to force a single topic per response.
    :param produce_structured_summary_radio: Option to produce a structured summary.
    :param aws_access_key_textbox: AWS access key for Bedrock.
    :param aws_secret_key_textbox: AWS secret key for Bedrock.
    :param hf_api_key_textbox: Hugging Face API key for local models.
    :param azure_api_key_textbox: Azure/OpenAI API key for Azure/OpenAI AI Inference.
    :param output_folder: The folder where output files will be saved.
    :param existing_logged_content: A list of existing logged content.
    :param force_single_topic_prompt: Prompt for forcing a single topic.
    :param additional_instructions_summary_format: Initial instructions to guide the format for the initial summary of the topics.
    :param additional_validation_issues_provided: Additional validation issues provided by the user.
    :param show_previous_table: Whether to show the previous table ("Yes" or "No").
    :param max_tokens: Maximum tokens for LLM generation.
    :param model_name_map: Dictionary mapping model names to their properties.
    :param max_time_for_loop: Maximum time allowed for the processing loop.
    :param reasoning_suffix: Suffix to append for reasoning.
    :param CHOSEN_LOCAL_MODEL_TYPE: Type of local model chosen.
    :param output_debug_files: Whether to output debug files ("True" or "False").
    :param model: Model object for local inference.
    :param tokenizer: Tokenizer object for local inference.
    :param assistant_model: Assistant model object for local inference.
    :param max_rows: The maximum number of rows to process.
    :param progress: Gradio Progress object for tracking progress.
    :return: A tuple containing consolidated results, mimicking the return structure of `extract_topics`.
    """

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_number_of_calls = 0
    out_message = list()

    # Logged content
    all_groups_logged_content = existing_logged_content

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            (
                in_colnames_drop,
                in_excel_sheets,
                file_name,
                join_colnames,
                join_colnames_drop,
            ) = put_columns_in_df(in_data_file)
            file_data, file_name, num_batches = load_in_data_file(
                in_data_file, chosen_cols, batch_size_default, in_excel_sheets
            )
        except Exception as e:
            out_message = "Could not load in data file due to: " + str(e)
            print(out_message)
            raise Exception(out_message)

    if file_data.shape[0] > max_rows:
        out_message = (
            "Your data has more than "
            + str(max_rows)
            + " rows, which has been set as the maximum in the application configuration."
        )
        print(out_message)
        raise Exception(out_message)

    if grouping_col is None:
        print("No grouping column found")
        file_data["group_col"] = "All"
        grouping_col = "group_col"

    if grouping_col not in file_data.columns:
        raise ValueError(f"Selected column '{grouping_col}' not found in file_data.")

    unique_values = file_data[grouping_col].unique()

    if len(unique_values) > MAX_GROUPS:
        print(
            f"Warning: More than {MAX_GROUPS} unique values found in '{grouping_col}'. Processing only the first {MAX_GROUPS}."
        )
        unique_values = unique_values[:MAX_GROUPS]

    # Initialise accumulators for results across all unique values
    # DataFrames are built upon iteratively
    acc_topics_table = initial_existing_topics_table.copy()
    acc_reference_df = initial_existing_reference_df.copy()
    acc_topic_summary_df = initial_existing_topic_summary_df.copy()
    acc_reference_df_pivot = pd.DataFrame()
    acc_missing_df = pd.DataFrame()

    # Lists are extended
    acc_out_file_paths = list()
    acc_log_files_output_paths = list()
    acc_join_file_paths = (
        list()
    )  # join_file_paths seems to be overwritten, so maybe last one or extend? Let's extend.

    # Single value outputs - typically the last one is most relevant, or sum for time
    acc_markdown_output = initial_unique_table_df_display_table_markdown
    acc_latest_batch_completed = (
        initial_latest_batch_completed  # From the last segment processed
    )
    acc_all_metadata_content = initial_all_metadata_content_str
    acc_total_time_taken = float(initial_time_taken)
    acc_gradio_df = gr.Dataframe(value=pd.DataFrame())  # type: ignore # Placeholder for the last Gradio DF
    acc_logged_content = list()

    wrapper_first_loop = initial_first_loop_state

    if len(unique_values) == 1:
        # If only one unique value, no need for progress bar, iterate directly
        loop_object = unique_values
    else:
        # If multiple unique values, use tqdm progress bar
        loop_object = progress.tqdm(
            unique_values, desc="Analysing group", unit="groups"
        )

    for i, group_value in enumerate(loop_object):
        print(
            f"\nProcessing group: {grouping_col} = {group_value} ({i+1}/{len(unique_values)})"
        )

        filtered_file_data = file_data.copy()

        filtered_file_data = filtered_file_data[
            filtered_file_data[grouping_col] == group_value
        ]

        if filtered_file_data.empty:
            print(f"No data for {grouping_col} = {group_value}. Skipping.")
            continue

        # Calculate num_batches for this specific segment
        current_num_batches = (len(filtered_file_data) + batch_size - 1) // batch_size

        # Modify file_name to be unique for this segment's outputs
        # _grp_{clean_column_name(grouping_col, max_length=15)}
        segment_file_name = f"{clean_column_name(original_file_name, max_length=15)}_{clean_column_name(str(group_value), max_length=15).replace(' ','_')}"

        # Determine first_loop_state for this call to extract_topics
        # It's True only if this is the very first segment *and* the wrapper was told it's the first loop.
        # For subsequent segments, it's False, as we are building on accumulated DFs.
        current_first_loop_state = wrapper_first_loop if i == 0 else False

        # latest_batch_completed for extract_topics should be 0 for each new segment,
        # as it processes the new filtered_file_data from its beginning.
        # However, if it's the very first call, respect initial_latest_batch_completed.
        current_latest_batch_completed = (
            initial_latest_batch_completed if i == 0 and wrapper_first_loop else 0
        )

        # Call extract_topics for the current segment
        try:
            (
                seg_markdown,
                seg_topics_table,
                seg_topic_summary_df,
                seg_reference_df,
                seg_out_files1,
                _seg_out_files2,  # Often same as 1
                seg_batch_completed,  # Specific to this segment's run
                seg_log_files1,
                _seg_log_files2,  # Often same as 1
                seg_conversation_metadata,
                seg_time_taken,
                _seg_out_files3,  # Often same as 1
                _seg_out_files4,  # Often same as 1
                seg_gradio_df,
                _seg_out_files5,  # Often same as 1
                seg_join_files,
                seg_reference_df_pivot,
                seg_missing_df,
                seg_logged_content,
            ) = extract_topics(
                in_data_file=in_data_file,
                file_data=filtered_file_data,
                existing_topics_table=pd.DataFrame(),  # acc_topics_table.copy(), # Pass the accumulated table
                existing_reference_df=pd.DataFrame(),  # acc_reference_df.copy(), # Pass the accumulated table
                existing_topic_summary_df=pd.DataFrame(),  # acc_topic_summary_df.copy(), # Pass the accumulated table
                unique_table_df_display_table_markdown="",  # extract_topics will generate this
                file_name=segment_file_name,
                num_batches=current_num_batches,
                in_api_key=in_api_key,
                temperature=temperature,
                chosen_cols=chosen_cols,
                model_choice=model_choice,
                candidate_topics=candidate_topics,
                latest_batch_completed=current_latest_batch_completed,  # Reset for each new segment's internal batching
                out_message=list(),  # Fresh for each call
                out_file_paths=list(),  # Fresh for each call
                log_files_output_paths=list(),  # Fresh for each call
                first_loop_state=current_first_loop_state,  # True only for the very first iteration of wrapper
                all_metadata_content_str="",  # Fresh for each call
                initial_table_prompt=initial_table_prompt,
                initial_table_system_prompt=initial_table_system_prompt,
                add_existing_topics_system_prompt=add_existing_topics_system_prompt,
                add_existing_topics_prompt=add_existing_topics_prompt,
                number_of_prompts_used=number_of_prompts_used,
                batch_size=batch_size,
                context_textbox=context_textbox,
                time_taken=0,  # Time taken for this specific call, wrapper sums it.
                sentiment_checkbox=sentiment_checkbox,
                force_zero_shot_radio=force_zero_shot_radio,
                in_excel_sheets=in_excel_sheets,
                force_single_topic_radio=force_single_topic_radio,
                output_folder=output_folder,
                force_single_topic_prompt=force_single_topic_prompt,
                group_name=group_value,
                produce_structured_summary_radio=produce_structured_summary_radio,
                aws_access_key_textbox=aws_access_key_textbox,
                aws_secret_key_textbox=aws_secret_key_textbox,
                hf_api_key_textbox=hf_api_key_textbox,
                azure_api_key_textbox=azure_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                max_tokens=max_tokens,
                model_name_map=model_name_map,
                max_time_for_loop=max_time_for_loop,
                CHOSEN_LOCAL_MODEL_TYPE=CHOSEN_LOCAL_MODEL_TYPE,
                output_debug_files=output_debug_files,
                reasoning_suffix=reasoning_suffix,
                model=model,
                tokenizer=tokenizer,
                assistant_model=assistant_model,
                max_rows=max_rows,
                existing_logged_content=all_groups_logged_content,
                original_full_file_name=original_file_name,
                additional_instructions_summary_format=additional_instructions_summary_format,
                additional_validation_issues_provided=additional_validation_issues_provided,
                api_url=api_url,
                progress=progress,
            )

            # Aggregate results
            # The DFs returned by extract_topics are already cumulative for *its own run*.
            # We now make them cumulative for the *wrapper's run*.
            acc_reference_df = pd.concat([acc_reference_df, seg_reference_df])
            acc_topic_summary_df = pd.concat(
                [acc_topic_summary_df, seg_topic_summary_df]
            )
            acc_reference_df_pivot = pd.concat(
                [acc_reference_df_pivot, seg_reference_df_pivot]
            )
            acc_missing_df = pd.concat([acc_missing_df, seg_missing_df])

            # For lists, extend. Use set to remove duplicates if paths might be re-added.
            acc_out_file_paths.extend(
                f for f in seg_out_files1 if f not in acc_out_file_paths
            )
            acc_log_files_output_paths.extend(
                f for f in seg_log_files1 if f not in acc_log_files_output_paths
            )
            acc_join_file_paths.extend(
                f for f in seg_join_files if f not in acc_join_file_paths
            )

            acc_markdown_output = seg_markdown  # Keep the latest markdown
            acc_latest_batch_completed = seg_batch_completed  # Keep latest batch count
            acc_all_metadata_content += (
                ("\n---\n" if acc_all_metadata_content else "")
                + f"Segment {grouping_col}={group_value}:\n"
                + seg_conversation_metadata
            )
            acc_total_time_taken += float(seg_time_taken)
            acc_gradio_df = seg_gradio_df  # Keep the latest Gradio DF
            acc_logged_content.extend(seg_logged_content)

            print(
                f"Group {grouping_col} = {group_value} processed. Time: {seg_time_taken:.2f}s"
            )

        except Exception as e:
            print(f"Error processing segment {grouping_col} = {group_value}: {e}")
            # Optionally, decide if you want to continue with other segments or stop
            # For now, it will continue
            continue

    overall_file_name = clean_column_name(original_file_name, max_length=20)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )
    column_clean = clean_column_name(chosen_cols, max_length=20)

    # Need to join "Topic number" onto acc_reference_df
    # If any blanks, there is an issue somewhere, drop and redo
    if "Topic number" in acc_reference_df.columns:
        if acc_reference_df["Topic number"].isnull().any():
            acc_reference_df = acc_reference_df.drop("Topic number", axis=1)

    if "Topic number" not in acc_reference_df.columns:
        if "Topic number" in acc_topic_summary_df.columns:
            if "General topic" in acc_topic_summary_df.columns:
                acc_reference_df = acc_reference_df.merge(
                    acc_topic_summary_df[
                        ["General topic", "Subtopic", "Sentiment", "Topic number"]
                    ],
                    on=["General topic", "Subtopic", "Sentiment"],
                    how="left",
                )
                # Sort output dataframes
                acc_reference_df["Response References"] = (
                    acc_reference_df["Response References"].astype(float).astype(int)
                )
                acc_reference_df["Start row of group"] = acc_reference_df[
                    "Start row of group"
                ].astype(int)
                acc_reference_df.sort_values(
                    [
                        "Group",
                        "Start row of group",
                        "Response References",
                        "General topic",
                        "Subtopic",
                        "Sentiment",
                    ],
                    inplace=True,
                )
            elif "Main heading" in acc_topic_summary_df.columns:
                acc_reference_df = acc_reference_df.merge(
                    acc_topic_summary_df[
                        ["Main heading", "Subheading", "Topic number"]
                    ],
                    on=["Main heading", "Subheading"],
                    how="left",
                )
                # Sort output dataframes
                acc_reference_df["Response References"] = (
                    acc_reference_df["Response References"].astype(float).astype(int)
                )
                acc_reference_df["Start row of group"] = acc_reference_df[
                    "Start row of group"
                ].astype(int)
                acc_reference_df.sort_values(
                    [
                        "Group",
                        "Start row of group",
                        "Response References",
                        "Main heading",
                        "Subheading",
                        "Topic number",
                    ],
                    inplace=True,
                )

    if "General topic" in acc_topic_summary_df.columns:
        acc_topic_summary_df["Number of responses"] = acc_topic_summary_df[
            "Number of responses"
        ].astype(int)
        acc_topic_summary_df.sort_values(
            ["Group", "Number of responses", "General topic", "Subtopic", "Sentiment"],
            ascending=[True, False, True, True, True],
            inplace=True,
        )
    elif "Main heading" in acc_topic_summary_df.columns:
        acc_topic_summary_df["Number of responses"] = acc_topic_summary_df[
            "Number of responses"
        ].astype(int)
        acc_topic_summary_df.sort_values(
            [
                "Group",
                "Number of responses",
                "Main heading",
                "Subheading",
                "Topic number",
            ],
            ascending=[True, False, True, True, True],
            inplace=True,
        )

    if "Group" in acc_reference_df.columns:
        # Create missing references dataframe using consolidated data from all groups
        # This ensures we correctly identify missing references across all groups
        # Get all basic_response_data from all groups
        all_basic_response_data = list()
        for logged_item in acc_logged_content:
            if "basic_response_data" in logged_item:
                all_basic_response_data.extend(logged_item["basic_response_data"])

        if all_basic_response_data:
            all_basic_response_df = pd.DataFrame(all_basic_response_data)
            acc_missing_df = create_missing_references_df(
                all_basic_response_df, acc_reference_df
            )
        else:
            # Fallback: if no logged content, create empty missing_df
            acc_missing_df = pd.DataFrame(
                columns=["Missing Reference", "Response Character Count"]
            )

        acc_reference_df_path = (
            output_folder
            + overall_file_name
            + "_col_"
            + column_clean
            + "_all_final_reference_table_"
            + model_choice_clean_short
            + ".csv"
        )
        acc_topic_summary_df_path = (
            output_folder
            + overall_file_name
            + "_col_"
            + column_clean
            + "_all_final_unique_topics_"
            + model_choice_clean_short
            + ".csv"
        )
        acc_reference_df_pivot_path = (
            output_folder
            + overall_file_name
            + "_col_"
            + column_clean
            + "_all_final_reference_pivot_"
            + model_choice_clean_short
            + ".csv"
        )
        acc_missing_df_path = (
            output_folder
            + overall_file_name
            + "_col_"
            + column_clean
            + "_all_missing_df_"
            + model_choice_clean_short
            + ".csv"
        )

        acc_reference_df.to_csv(acc_reference_df_path, index=None, encoding="utf-8-sig")
        acc_topic_summary_df.to_csv(
            acc_topic_summary_df_path, index=None, encoding="utf-8-sig"
        )
        acc_reference_df_pivot.to_csv(
            acc_reference_df_pivot_path, index=None, encoding="utf-8-sig"
        )
        acc_missing_df.to_csv(acc_missing_df_path, index=None, encoding="utf-8-sig")

        acc_log_files_output_paths.append(acc_missing_df_path)

        # Remove the existing output file list and replace with the updated concatenated outputs
        substring_list_to_remove = [
            "_final_reference_table_pivot_",
            "_final_reference_table_",
            "_final_unique_topics_",
        ]
        acc_out_file_paths = [
            x
            for x in acc_out_file_paths
            if not any(sub in x for sub in substring_list_to_remove)
        ]

        acc_out_file_paths.extend([acc_reference_df_path, acc_topic_summary_df_path])

        # Outputs for markdown table output
        unique_table_df_display_table = acc_topic_summary_df.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )
        if produce_structured_summary_radio == "Yes":
            unique_table_df_display_table = unique_table_df_display_table[
                ["General topic", "Subtopic", "Summary", "Group"]
            ]
            unique_table_df_display_table.rename(
                columns={"General topic": "Main heading", "Subtopic": "Subheading"},
                inplace=True,
            )
            acc_markdown_output = unique_table_df_display_table.to_markdown(index=False)
        else:
            acc_markdown_output = unique_table_df_display_table[
                [
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                    "Number of responses",
                    "Summary",
                    "Group",
                ]
            ].to_markdown(index=False)

    acc_input_tokens, acc_output_tokens, acc_number_of_calls = (
        calculate_tokens_from_metadata(
            acc_all_metadata_content, model_choice, model_name_map
        )
    )

    out_message = "\n".join(out_message)
    out_message = (
        out_message
        + " "
        + f"Topic extraction finished processing all groups. Total time: {acc_total_time_taken:.2f}s"
    )
    print(out_message)

    out_logged_content_df_path = (
        output_folder
        + overall_file_name
        + "_col_"
        + column_clean
        + "_logs_"
        + model_choice_clean_short
        + ".json"
    )

    with open(
        out_logged_content_df_path, "w", encoding="utf-8-sig", errors="replace"
    ) as f:
        f.write(json.dumps(acc_logged_content))

    acc_log_files_output_paths.append(out_logged_content_df_path)

    # The return signature should match extract_topics.
    # The aggregated lists will be returned in the multiple slots.
    return (
        acc_markdown_output,
        acc_topics_table,
        acc_topic_summary_df,
        acc_reference_df,
        acc_out_file_paths,  # Slot 1 for out_file_paths
        acc_out_file_paths,  # Slot 2 for out_file_paths
        acc_latest_batch_completed,  # From the last successfully processed segment
        acc_log_files_output_paths,  # Slot 1 for log_files_output_paths
        acc_log_files_output_paths,  # Slot 2 for log_files_output_paths
        acc_all_metadata_content,
        acc_total_time_taken,
        acc_out_file_paths,  # Slot 3
        acc_out_file_paths,  # Slot 4
        acc_gradio_df,  # Last Gradio DF
        acc_out_file_paths,  # Slot 5
        acc_join_file_paths,
        acc_missing_df,
        acc_input_tokens,
        acc_output_tokens,
        acc_number_of_calls,
        out_message,
        acc_logged_content,
    )


def join_modified_topic_names_to_ref_table(
    modified_topic_summary_df: pd.DataFrame,
    original_topic_summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
):
    """
    Take a unique topic table that has been modified by the user, and apply the topic name changes to the long-form reference table.
    """

    # Drop rows where Number of responses is either NA or null
    modified_topic_summary_df = modified_topic_summary_df[
        ~modified_topic_summary_df["Number of responses"].isnull()
    ]
    modified_topic_summary_df.drop_duplicates(
        ["General topic", "Subtopic", "Sentiment", "Topic number"], inplace=True
    )

    # First, join the modified topics to the original topics dataframe based on index to have the modified names alongside the original names
    original_topic_summary_df_m = original_topic_summary_df.merge(
        modified_topic_summary_df[
            ["General topic", "Subtopic", "Sentiment", "Topic number"]
        ],
        on="Topic number",
        how="left",
        suffixes=("", "_mod"),
    )

    original_topic_summary_df_m.drop_duplicates(
        ["General topic", "Subtopic", "Sentiment", "Topic number"], inplace=True
    )

    # Then, join these new topic names onto the reference_df, merge based on the original names
    modified_reference_df = reference_df.merge(
        original_topic_summary_df_m[
            ["Topic number", "General Topic_mod", "Subtopic_mod", "Sentiment_mod"]
        ],
        on=["Topic number"],
        how="left",
    )

    modified_reference_df.drop(
        ["General topic", "Subtopic", "Sentiment"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    modified_reference_df.rename(
        columns={
            "General Topic_mod": "General topic",
            "Subtopic_mod": "Subtopic",
            "Sentiment_mod": "Sentiment",
        },
        inplace=True,
    )

    modified_reference_df.drop(
        ["General Topic_mod", "Subtopic_mod", "Sentiment_mod"],
        inplace=True,
        errors="ignore",
    )

    # modified_reference_df.drop_duplicates(["Response References", "General topic", "Subtopic", "Sentiment"], inplace=True)

    modified_reference_df.sort_values(
        [
            "Start row of group",
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
        ],
        inplace=True,
    )

    modified_reference_df = modified_reference_df.loc[
        :,
        [
            "Response References",
            "General topic",
            "Subtopic",
            "Sentiment",
            "Summary",
            "Start row of group",
            "Topic number",
        ],
    ]

    # Drop rows where Response References is either NA or null
    modified_reference_df = modified_reference_df[
        ~modified_reference_df["Response References"].isnull()
    ]

    return modified_reference_df


# MODIFY EXISTING TABLE
def modify_existing_output_tables(
    original_topic_summary_df: pd.DataFrame,
    modifiable_topic_summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    text_output_file_list_state: List[str],
    output_folder: str = OUTPUT_FOLDER,
) -> Tuple:
    """
    Take a unique_topics table that has been modified, apply these new topic names to the long-form reference_df, and save both tables to file.
    """

    # Ensure text_output_file_list_state is a flat list
    if any(isinstance(i, list) for i in text_output_file_list_state):
        text_output_file_list_state = [
            item for sublist in text_output_file_list_state for item in sublist
        ]  # Flatten list

    # Extract file paths safely
    reference_files = [x for x in text_output_file_list_state if "reference" in x]
    unique_files = [x for x in text_output_file_list_state if "unique" in x]

    # Ensure files exist before accessing
    reference_file_path = (
        os.path.basename(reference_files[0]) if reference_files else None
    )
    unique_table_file_path = os.path.basename(unique_files[0]) if unique_files else None

    output_file_list = list()

    if reference_file_path and unique_table_file_path:

        reference_df = join_modified_topic_names_to_ref_table(
            modifiable_topic_summary_df, original_topic_summary_df, reference_df
        )

        ## Reference table mapping response numbers to topics
        reference_table_file_name = reference_file_path.replace(".csv", "_mod")
        new_reference_df_file_path = output_folder + reference_table_file_name + ".csv"
        reference_df.to_csv(
            new_reference_df_file_path, index=None, encoding="utf-8-sig"
        )
        output_file_list.append(new_reference_df_file_path)

        # Drop rows where Number of responses is NA or null
        modifiable_topic_summary_df = modifiable_topic_summary_df[
            ~modifiable_topic_summary_df["Number of responses"].isnull()
        ]

        # Convert 'Number of responses' to numeric (forcing errors to NaN if conversion fails)
        modifiable_topic_summary_df["Number of responses"] = pd.to_numeric(
            modifiable_topic_summary_df["Number of responses"], errors="coerce"
        )

        # Drop any rows where conversion failed (original non-numeric values)
        modifiable_topic_summary_df.dropna(subset=["Number of responses"], inplace=True)

        # Sort values
        modifiable_topic_summary_df.sort_values(
            ["Number of responses"], ascending=False, inplace=True
        )

        unique_table_file_name = unique_table_file_path.replace(".csv", "_mod")
        modified_unique_table_file_path = (
            output_folder + unique_table_file_name + ".csv"
        )
        modifiable_topic_summary_df.to_csv(
            modified_unique_table_file_path, index=None, encoding="utf-8-sig"
        )
        output_file_list.append(modified_unique_table_file_path)

    else:
        output_file_list = text_output_file_list_state
        reference_table_file_name = reference_file_path
        unique_table_file_name = unique_table_file_path
        raise Exception("Reference and unique topic tables not found.")

    # Outputs for markdown table output
    unique_table_df_revised_display = modifiable_topic_summary_df.apply(
        lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
    )
    deduplicated_unique_table_markdown = unique_table_df_revised_display.to_markdown(
        index=False
    )

    return (
        modifiable_topic_summary_df,
        reference_df,
        output_file_list,
        output_file_list,
        output_file_list,
        output_file_list,
        reference_table_file_name,
        unique_table_file_name,
        deduplicated_unique_table_markdown,
    )


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def all_in_one_pipeline(
    grouping_col: str,
    in_data_files: List[str],
    file_data: pd.DataFrame,
    existing_topics_table: pd.DataFrame,
    existing_reference_df: pd.DataFrame,
    existing_topic_summary_df: pd.DataFrame,
    unique_table_df_display_table_markdown: str,
    original_file_name: str,
    total_number_of_batches: int,
    in_api_key: str,
    temperature: float,
    chosen_cols: List[str],
    model_choice: str,
    candidate_topics: gr.FileData,
    first_loop_state: bool,
    conversation_metadata_text: str,
    latest_batch_completed: int,
    time_taken_so_far: float,
    initial_table_prompt_text: str,
    initial_table_system_prompt_text: str,
    add_existing_topics_system_prompt_text: str,
    add_existing_topics_prompt_text: str,
    number_of_prompts_used: int,
    batch_size: int,
    context_text: str,
    sentiment_choice: str,
    force_zero_shot_choice: str,
    in_excel_sheets: List[str],
    force_single_topic_choice: str,
    produce_structures_summary_choice: str,
    aws_access_key_text: str,
    aws_secret_key_text: str,
    hf_api_key_text: str,
    azure_api_key_text: str,
    azure_endpoint_text: str,
    output_folder: str = OUTPUT_FOLDER,
    merge_sentiment: str = "No",
    merge_general_topics: str = "Yes",
    score_threshold: int = 90,
    summarise_format: str = "",
    random_seed: int = 42,
    log_files_output_list_state: List[str] = list(),
    model_name_map_state: dict = model_name_map,
    usage_logs_location: str = "",
    existing_logged_content: list = list(),
    additional_instructions_summary_format: str = "",
    additional_validation_issues_provided: str = "",
    show_previous_table: str = "Yes",
    sample_reference_table_checkbox: bool = True,
    api_url: str = None,
    output_debug_files: str = output_debug_files,
    model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    max_rows: int = max_rows,
    progress=Progress(track_tqdm=True),
):
    """
    Orchestrates the full All-in-one flow: extract → deduplicate → summarise → overall summary → Excel export.

    Args:
        grouping_col (str): The column used for grouping data.
        in_data_files (List[str]): List of input data file paths.
        file_data (pd.DataFrame): The input data as a pandas DataFrame.
        existing_topics_table (pd.DataFrame): DataFrame of existing topics.
        existing_reference_df (pd.DataFrame): DataFrame of existing reference data.
        existing_topic_summary_df (pd.DataFrame): DataFrame of existing topic summaries.
        unique_table_df_display_table_markdown (str): Markdown string for displaying unique topics.
        original_file_name (str): The original name of the input file.
        total_number_of_batches (int): Total number of batches for processing.
        in_api_key (str): API key for the LLM.
        temperature (float): Temperature setting for the LLM.
        chosen_cols (List[str]): List of columns chosen for analysis.
        model_choice (str): The chosen LLM model.
        candidate_topics (gr.FileData): Gradio file data for candidate topics.
        first_loop_state (bool): State indicating if it's the first loop.
        conversation_metadata_text (str): Text containing conversation metadata.
        latest_batch_completed (int): The latest batch number completed.
        time_taken_so_far (float): Cumulative time taken so far.
        initial_table_prompt_text (str): Initial prompt text for table generation.
        initial_table_system_prompt_text (str): Initial system prompt text for table generation.
        add_existing_topics_system_prompt_text (str): System prompt for adding existing topics.
        add_existing_topics_prompt_text (str): Prompt for adding existing topics.
        number_of_prompts_used (int): Number of prompts used in sequence.
        batch_size (int): Size of each processing batch.
        context_text (str): Additional context for the LLM.
        sentiment_choice (str): Choice for sentiment analysis (e.g., "Yes", "No").
        force_zero_shot_choice (str): Choice to force zero-shot prompting.
        in_excel_sheets (List[str]): List of sheet names in the input Excel file.
        force_single_topic_choice (str): Choice to force single topic extraction.
        produce_structures_summary_choice (str): Choice to produce structured summaries.
        aws_access_key_text (str): AWS access key.
        aws_secret_key_text (str): AWS secret key.
        hf_api_key_text (str): Hugging Face API key.
        azure_api_key_text (str): Azure/OpenAI API key.
        output_folder (str, optional): Folder to save output files. Defaults to OUTPUT_FOLDER.
        merge_sentiment (str, optional): Whether to merge sentiment. Defaults to "No".
        merge_general_topics (str, optional): Whether to merge general topics. Defaults to "Yes".
        score_threshold (int, optional): Score threshold for topic matching. Defaults to 90.
        summarise_format (str, optional): Format for summarization. Defaults to "".
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        log_files_output_list_state (List[str], optional): List of log file paths. Defaults to list().
        model_name_map_state (dict, optional): Mapping of model names. Defaults to model_name_map.
        usage_logs_location (str, optional): Location for usage logs. Defaults to "".
        existing_logged_content (list, optional): Existing logged content. Defaults to list().
        additional_instructions_summary_format (str, optional): Summary format for adding existing topics. Defaults to "".
        additional_validation_issues_provided (str, optional): Additional validation issues provided by the user. Defaults to "".
        show_previous_table (str, optional): Whether to show the previous table ("Yes" or "No"). Defaults to "Yes".
        sample_reference_table_checkbox (bool, optional): Whether to sample summaries before creating revised summaries.
        api_url (str, optional): API URL for inference-server models. Defaults to None.
        output_debug_files (str, optional): Whether to output debug files. Defaults to "False".
        model (object, optional): Loaded local model object. Defaults to None.
        tokenizer (object, optional): Loaded local tokenizer object. Defaults to None.
        assistant_model (object, optional): Loaded local assistant model object. Defaults to None.
        max_rows (int, optional): Maximum number of rows to process. Defaults to max_rows.
        progress (Progress, optional): Gradio Progress object for tracking. Defaults to Progress(track_tqdm=True).

    Returns:
        A tuple matching the UI components updated during the original chained flow.
    """

    # Ensure custom model_choice is registered in model_name_map_state
    ensure_model_in_map(model_choice, model_name_map_state)

    # Load local model if it's not already loaded
    if (
        (model_name_map_state[model_choice]["source"] == "Local")
        & (RUN_LOCAL_MODEL == "1")
        & (not model)
    ):
        model = get_model()
        tokenizer = get_tokenizer()
        assistant_model = get_assistant_model()

    total_input_tokens = 0
    total_output_tokens = 0
    total_number_of_calls = 0
    total_time_taken = 0
    out_message = list()
    out_logged_content = list()

    # 1) Extract topics (group-aware)
    (
        display_markdown,
        out_topics_table,
        out_topic_summary_df,
        out_reference_df,
        out_file_paths_1,
        _out_file_paths_dup,
        out_latest_batch_completed,
        out_log_files,
        _out_log_files_dup,
        out_conversation_metadata,
        out_time_taken,
        out_file_paths_2,
        _out_file_paths_3,
        out_gradio_df,
        out_file_paths_4,
        out_join_files,
        out_missing_df,
        out_input_tokens,
        out_output_tokens,
        out_number_of_calls,
        out_message_text,
        out_logged_content,
    ) = wrapper_extract_topics_per_column_value(
        grouping_col=grouping_col,
        in_data_file=in_data_files,
        file_data=file_data,
        initial_existing_topics_table=existing_topics_table,
        initial_existing_reference_df=existing_reference_df,
        initial_existing_topic_summary_df=existing_topic_summary_df,
        initial_unique_table_df_display_table_markdown=unique_table_df_display_table_markdown,
        original_file_name=original_file_name,
        total_number_of_batches=total_number_of_batches,
        in_api_key=in_api_key,
        temperature=temperature,
        chosen_cols=chosen_cols,
        model_choice=model_choice,
        candidate_topics=candidate_topics,
        initial_first_loop_state=first_loop_state,
        initial_all_metadata_content_str=conversation_metadata_text,
        initial_latest_batch_completed=latest_batch_completed,
        initial_time_taken=time_taken_so_far,
        initial_table_prompt=initial_table_prompt_text,
        initial_table_system_prompt=initial_table_system_prompt_text,
        add_existing_topics_system_prompt=add_existing_topics_system_prompt_text,
        add_existing_topics_prompt=add_existing_topics_prompt_text,
        number_of_prompts_used=number_of_prompts_used,
        batch_size=batch_size,
        context_textbox=context_text,
        sentiment_checkbox=sentiment_choice,
        force_zero_shot_radio=force_zero_shot_choice,
        in_excel_sheets=in_excel_sheets,
        force_single_topic_radio=force_single_topic_choice,
        produce_structured_summary_radio=produce_structures_summary_choice,
        aws_access_key_textbox=aws_access_key_text,
        aws_secret_key_textbox=aws_secret_key_text,
        hf_api_key_textbox=hf_api_key_text,
        azure_api_key_textbox=azure_api_key_text,
        azure_endpoint_textbox=azure_endpoint_text,
        output_folder=output_folder,
        existing_logged_content=existing_logged_content,
        model_name_map=model_name_map_state,
        output_debug_files=output_debug_files,
        model=model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        max_rows=max_rows,
        additional_instructions_summary_format=additional_instructions_summary_format,
        additional_validation_issues_provided=additional_validation_issues_provided,
        show_previous_table=show_previous_table,
        api_url=api_url,
    )

    total_input_tokens += out_input_tokens
    total_output_tokens += out_output_tokens
    total_number_of_calls += out_number_of_calls
    total_time_taken += out_time_taken
    out_message.append(out_message_text)

    # Prepare outputs after extraction, matching wrapper outputs
    topic_extraction_output_files = out_file_paths_1
    text_output_file_list_state = out_file_paths_1
    log_files_output_list_state = out_log_files

    # If producing structured summaries, return the outputs after extraction
    if produce_structures_summary_choice == "Yes":

        # Write logged content to file
        column_clean = clean_column_name(chosen_cols, max_length=20)
        model_choice_clean = model_name_map[model_choice]["short_name"]
        model_choice_clean_short = clean_column_name(
            model_choice_clean, max_length=20, front_characters=False
        )

        out_logged_content_df_path = (
            output_folder
            + original_file_name
            + "_col_"
            + column_clean
            + "_logs_"
            + model_choice_clean_short
            + ".json"
        )

        with open(
            out_logged_content_df_path, "w", encoding="utf-8-sig", errors="replace"
        ) as f:
            f.write(json.dumps(out_logged_content))

        log_files_output_list_state.append(out_logged_content_df_path)
        out_log_files.append(out_logged_content_df_path)

        # Map to the UI outputs list expected by the new single-call wiring
        return (
            display_markdown,
            out_topics_table,
            out_topic_summary_df,
            out_reference_df,
            topic_extraction_output_files,
            text_output_file_list_state,
            out_latest_batch_completed,
            out_log_files,
            log_files_output_list_state,
            out_conversation_metadata,
            total_time_taken,
            out_file_paths_1,
            list(),  # summarisation_input_files is not available yet
            out_gradio_df,
            list(),  # modification_input_files placeholder
            out_join_files,
            out_missing_df,
            total_input_tokens,
            total_output_tokens,
            total_number_of_calls,
            out_message[0],
            pd.DataFrame(),  # summary_reference_table_sample_state is not available yet
            "",  # summarised_references_markdown is not available yet
            out_topic_summary_df,
            out_reference_df,
            list(),  # summary_output_files is not available yet
            list(),  # summarised_outputs_list is not available yet
            0,  # latest_summary_completed_num is not available yet
            list(),  # overall_summarisation_input_files is not available yet
            list(),  # overall_summary_output_files is not available yet
            "",  # overall_summarised_output_markdown is not available yet
            pd.DataFrame(),  # summarised_output_df is not available yet
            out_logged_content,
        )

    # 2) Deduplication
    (
        ref_df_loaded,
        unique_df_loaded,
        latest_batch_completed_no_loop,
        deduplication_input_files_status,
        working_data_file_name_textbox,
        unique_topics_table_file_name_textbox,
    ) = load_in_previous_data_files(out_file_paths_1)

    (
        ref_df_after_dedup,
        unique_df_after_dedup,
        summarisation_input_files,
        log_files_output_dedup,
        summarised_output_markdown,
    ) = deduplicate_topics(
        reference_df=ref_df_loaded if not ref_df_loaded.empty else out_reference_df,
        topic_summary_df=(
            unique_df_loaded if not unique_df_loaded.empty else out_topic_summary_df
        ),
        reference_table_file_name=working_data_file_name_textbox,
        unique_topics_table_file_name=unique_topics_table_file_name_textbox,
        in_excel_sheets=in_excel_sheets,
        merge_sentiment=merge_sentiment,
        merge_general_topics=merge_general_topics,
        score_threshold=score_threshold,
        in_data_files=in_data_files,
        chosen_cols=chosen_cols,
        output_folder=output_folder,
    )

    # 3) Summarisation
    (
        ref_df_loaded_2,
        unique_df_loaded_2,
        _latest_batch_completed_no_loop_2,
        _deduplication_input_files_status_2,
        _working_name_2,
        _unique_name_2,
    ) = load_in_previous_data_files(summarisation_input_files)

    (
        summary_reference_table_sample_state,
        master_unique_topics_df_revised_summaries_state,
        master_reference_df_revised_summaries_state,
        summary_output_files,
        summarised_outputs_list,
        latest_summary_completed_num,
        conversation_metadata_text_updated,
        display_markdown_updated,
        log_files_output_after_sum,
        overall_summarisation_input_files,
        input_tokens_num,
        output_tokens_num,
        number_of_calls_num,
        estimated_time_taken_number,
        output_messages_textbox,
        out_logged_content,
    ) = wrapper_summarise_output_topics_per_group(
        grouping_col=grouping_col,
        sampled_reference_table_df=ref_df_after_dedup,
        topic_summary_df=unique_df_after_dedup,
        reference_table_df=ref_df_after_dedup,
        model_choice=model_choice,
        in_api_key=in_api_key,
        temperature=temperature,
        reference_data_file_name=working_data_file_name_textbox,
        summarised_outputs=list(),
        latest_summary_completed=0,
        out_metadata_str=out_conversation_metadata,
        in_data_files=in_data_files,
        in_excel_sheets=in_excel_sheets,
        chosen_cols=chosen_cols,
        log_output_files=log_files_output_list_state,
        summarise_format_radio=summarise_format,
        output_folder=output_folder,
        context_textbox=context_text,
        aws_access_key_textbox=aws_access_key_text,
        aws_secret_key_textbox=aws_secret_key_text,
        model_name_map=model_name_map_state,
        hf_api_key_textbox=hf_api_key_text,
        azure_endpoint_textbox=azure_endpoint_text,
        additional_summary_instructions_provided=additional_instructions_summary_format,
        local_model=model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        existing_logged_content=out_logged_content,
        sample_reference_table=sample_reference_table_checkbox,
        no_of_sampled_summaries=100,
        random_seed=random_seed,
        output_debug_files=output_debug_files,
        api_url=api_url,
    )

    # Generate summarised_references_markdown from the sampled reference table
    summarised_references_markdown = summary_reference_table_sample_state.to_markdown(
        index=False
    )

    total_input_tokens += input_tokens_num
    total_output_tokens += output_tokens_num
    total_number_of_calls += number_of_calls_num
    total_time_taken += estimated_time_taken_number
    out_message.append(output_messages_textbox)

    # 4) Overall summary
    (
        _ref_df_loaded_3,
        _unique_df_loaded_3,
        _latest_batch_completed_no_loop_3,
        _deduplication_input_files_status_3,
        _working_name_3,
        _unique_name_3,
    ) = load_in_previous_data_files(overall_summarisation_input_files)

    (
        overall_summary_output_files,
        overall_summarised_output_markdown,
        summarised_output_df,
        conversation_metadata_textbox,
        input_tokens_num,
        output_tokens_num,
        number_of_calls_num,
        estimated_time_taken_number,
        output_messages_textbox,
        out_logged_content,
    ) = overall_summary(
        topic_summary_df=master_unique_topics_df_revised_summaries_state,
        model_choice=model_choice,
        in_api_key=in_api_key,
        temperature=temperature,
        reference_data_file_name=working_data_file_name_textbox,
        output_folder=output_folder,
        chosen_cols=chosen_cols,
        context_textbox=context_text,
        aws_access_key_textbox=aws_access_key_text,
        aws_secret_key_textbox=aws_secret_key_text,
        model_name_map=model_name_map_state,
        hf_api_key_textbox=hf_api_key_text,
        azure_endpoint_textbox=azure_endpoint_text,
        local_model=model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        existing_logged_content=out_logged_content,
        output_debug_files=output_debug_files,
        api_url=api_url,
    )

    total_input_tokens += input_tokens_num
    total_output_tokens += output_tokens_num
    total_number_of_calls += number_of_calls_num
    total_time_taken += estimated_time_taken_number
    out_message.append(output_messages_textbox)

    out_message = "\n".join(out_message)
    out_message = (
        out_message + "\n" + f"Overall time for all processes: {total_time_taken:.2f}s"
    )
    print(out_message)

    # Write logged content to file
    column_clean = clean_column_name(chosen_cols, max_length=20)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    out_logged_content_df_path = (
        output_folder
        + original_file_name
        + "_col_"
        + column_clean
        + "_logs_"
        + model_choice_clean_short
        + ".json"
    )

    with open(
        out_logged_content_df_path, "w", encoding="utf-8-sig", errors="replace"
    ) as f:
        f.write(json.dumps(out_logged_content))

    log_files_output_list_state.append(out_logged_content_df_path)
    log_files_output_after_sum.append(out_logged_content_df_path)

    # Map to the UI outputs list expected by the new single-call wiring
    # Use the original markdown with renamed columns if produce_structured_summary_radio is "Yes"
    final_display_markdown = (
        display_markdown_updated if display_markdown_updated else display_markdown
    )
    if produce_structures_summary_choice == "Yes":
        final_display_markdown = unique_table_df_display_table_markdown

    return (
        final_display_markdown,
        out_topics_table,
        unique_df_after_dedup,
        ref_df_after_dedup,
        topic_extraction_output_files,
        text_output_file_list_state,
        out_latest_batch_completed,
        log_files_output_after_sum if log_files_output_after_sum else out_log_files,
        log_files_output_list_state,
        (
            conversation_metadata_text_updated
            if conversation_metadata_text_updated
            else out_conversation_metadata
        ),
        total_time_taken,
        out_file_paths_1,
        summarisation_input_files,
        out_gradio_df,
        list(),  # modification_input_files placeholder
        out_join_files,
        out_missing_df,
        total_input_tokens,
        total_output_tokens,
        total_number_of_calls,
        out_message,
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
        out_logged_content,
    )
