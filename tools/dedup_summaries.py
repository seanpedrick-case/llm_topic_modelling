import os
import re
import time
from io import StringIO
from typing import List, Tuple, Union

import boto3
import gradio as gr
import markdown
import pandas as pd
import spaces
from rapidfuzz import fuzz, process
from tqdm import tqdm

from tools.aws_functions import connect_to_bedrock_runtime
from tools.config import (
    BATCH_SIZE_DEFAULT,
    CHOSEN_LOCAL_MODEL_TYPE,
    DEDUPLICATION_THRESHOLD,
    DEFAULT_SAMPLED_SUMMARIES,
    LLM_CONTEXT_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LLM_SEED,
    MAX_COMMENT_CHARS,
    MAX_GROUPS,
    MAX_SPACES_GPU_RUN_TIME,
    MAX_TIME_FOR_LOOP,
    MAXIMUM_ALLOWED_TOPICS,
    NUMBER_OF_RETRY_ATTEMPTS,
    OUTPUT_DEBUG_FILES,
    OUTPUT_FOLDER,
    REASONING_SUFFIX,
    RUN_LOCAL_MODEL,
    TIMEOUT_WAIT,
    model_name_map,
)
from tools.helper_functions import (
    clean_column_name,
    convert_reference_table_to_pivot_table,
    create_batch_file_path_details,
    create_topic_summary_df_from_reference_table,
    ensure_model_in_map,
    generate_zero_shot_topics_df,
    get_basic_response_data,
    get_file_name_no_ext,
    initial_clean,
    load_in_data_file,
    normalize_topic_name_for_llm,
    read_file,
    wrap_text,
)
from tools.llm_funcs import (
    calculate_tokens_from_metadata,
    call_llm_with_markdown_table_checks,
    construct_azure_client,
    construct_gemini_generative_model,
    get_assistant_model,
    get_model,
    get_tokenizer,
    process_requests,
)
from tools.prompts import (
    comprehensive_summary_format_prompt,
    comprehensive_summary_format_prompt_by_group,
    llm_deduplication_prompt,
    llm_deduplication_prompt_with_candidates,
    llm_deduplication_system_prompt,
    summarise_everything_prompt,
    summarise_everything_system_prompt,
    summarise_topic_descriptions_prompt,
    summarise_topic_descriptions_system_prompt,
    summary_assistant_prefill,
    system_prompt,
)

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS
reasoning_suffix = REASONING_SUFFIX
output_debug_files = OUTPUT_DEBUG_FILES
default_number_of_sampled_summaries = DEFAULT_SAMPLED_SUMMARIES
max_text_length = 500
max_number_of_topics = MAXIMUM_ALLOWED_TOPICS


# DEDUPLICATION/SUMMARISATION FUNCTIONS
def deduplicate_categories(
    category_series: pd.Series,
    join_series: pd.Series,
    reference_df: pd.DataFrame,
    general_topic_series: pd.Series = None,
    merge_general_topics="No",
    merge_sentiment: str = "No",
    threshold: float = 90,
) -> pd.DataFrame:
    """
    Deduplicates similar category names in a pandas Series based on a fuzzy matching threshold,
    merging smaller topics into larger topics.

    Parameters:
        category_series (pd.Series): Series containing category names to deduplicate.
        join_series (pd.Series): Additional series used for joining back to original results.
        reference_df (pd.DataFrame): DataFrame containing the reference data to count occurrences.
        threshold (float): Similarity threshold for considering two strings as duplicates.

    Returns:
        pd.DataFrame: DataFrame with columns ['old_category', 'deduplicated_category'].
    """
    # Count occurrences of each category in the reference_df
    category_counts = reference_df["Subtopic"].value_counts().to_dict()

    # Initialize dictionaries for both category mapping and scores
    deduplication_map = {}
    match_scores = {}  # New dictionary to store match scores

    # First pass: Handle exact matches
    for category in category_series.unique():
        if category in deduplication_map:
            continue

        # Find all exact matches
        exact_matches = category_series[
            category_series.str.lower() == category.lower()
        ].index.tolist()
        if len(exact_matches) > 1:
            # Find the variant with the highest count
            match_counts = {
                match: category_counts.get(category_series[match], 0)
                for match in exact_matches
            }
            most_common = max(match_counts.items(), key=lambda x: x[1])[0]
            most_common_category = category_series[most_common]

            # Map all exact matches to the most common variant and store score
            for match in exact_matches:
                deduplication_map[category_series[match]] = most_common_category
                match_scores[category_series[match]] = (
                    100  # Exact matches get score of 100
                )

    # Second pass: Handle fuzzy matches for remaining categories
    # Create a DataFrame to maintain the relationship between categories and general topics
    categories_df = pd.DataFrame(
        {"category": category_series, "general_topic": general_topic_series}
    ).drop_duplicates()

    for _, row in categories_df.iterrows():
        category = row["category"]
        if category in deduplication_map:
            continue

        current_general_topic = row["general_topic"]

        # Filter potential matches to only those within the same General topic if relevant
        if merge_general_topics == "No":
            potential_matches = categories_df[
                (categories_df["category"] != category)
                & (categories_df["general_topic"] == current_general_topic)
            ]["category"].tolist()
        else:
            potential_matches = categories_df[(categories_df["category"] != category)][
                "category"
            ].tolist()

        matches = process.extract(
            category, potential_matches, scorer=fuzz.WRatio, score_cutoff=threshold
        )

        if matches:
            best_match = max(matches, key=lambda x: x[1])
            match, score, _ = best_match

            if category_counts.get(category, 0) < category_counts.get(match, 0):
                deduplication_map[category] = match
                match_scores[category] = score
            else:
                deduplication_map[match] = category
                match_scores[match] = score
        else:
            deduplication_map[category] = category
            match_scores[category] = 100

    # Create the result DataFrame with scores
    result_df = pd.DataFrame(
        {
            "old_category": category_series + " | " + join_series,
            "deduplicated_category": category_series.map(
                lambda x: deduplication_map.get(x, x)
            ),
            "match_score": category_series.map(
                lambda x: match_scores.get(x, 100)
            ),  # Add scores column
        }
    )

    # print(result_df)

    return result_df


def deduplicate_topics(
    reference_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    reference_table_file_name: str,
    unique_topics_table_file_name: str,
    in_excel_sheets: str = "",
    merge_sentiment: str = "No",
    merge_general_topics: str = "No",
    score_threshold: int = 90,
    in_data_files: Union[List[str], pd.DataFrame] = list(),
    chosen_cols: List[str] = "",
    output_folder: str = OUTPUT_FOLDER,
    deduplicate_topics: str = "Yes",
    output_files: str = "True",
    total_number_of_batches: int = None,
    data_file_names_textbox: str = None,
    sentiment_checkbox: str = "Negative, Neutral, or Positive",
):
    """
    Deduplicate topics based on a reference and unique topics table, merging similar topics.

    Args:
        reference_df (pd.DataFrame): DataFrame containing reference data with topics.
        topic_summary_df (pd.DataFrame): DataFrame summarizing unique topics.
        reference_table_file_name (str): Base file name for the output reference table.
        unique_topics_table_file_name (str): Base file name for the output unique topics table.
        in_excel_sheets (str, optional): Comma-separated list of Excel sheet names to load. Defaults to "".
        merge_sentiment (str, optional): Whether to merge topics regardless of sentiment ("Yes" or "No"). Defaults to "No".
        merge_general_topics (str, optional): Whether to merge topics across different general topics ("Yes" or "No"). Defaults to "No".
        score_threshold (int, optional): Fuzzy matching score threshold for deduplication. Defaults to 90.
        in_data_files (Union[List[str], pd.DataFrame], optional): List of input data file paths or a pandas DataFrame. If a DataFrame is provided, it will be used directly without loading from file. Defaults to [].
        chosen_cols (List[str], optional): List of chosen columns from the input data files. Defaults to "".
        output_folder (str, optional): Folder path to save output files. Defaults to OUTPUT_FOLDER.
        deduplicate_topics (str, optional): Whether to perform topic deduplication ("Yes" or "No"). Defaults to "Yes".
        output_files (str, optional): Whether to output files ("True" or "False"). Defaults to "True".
        total_number_of_batches (int, optional): Total number of batches when in_data_files is a DataFrame. If None and in_data_files is a DataFrame, defaults to 1. Defaults to None.
        data_file_names_textbox (str, optional): File name when in_data_files is a DataFrame. If None and in_data_files is a DataFrame, defaults to "dataframe". Defaults to None.
    """
    # Save the parameter value before it gets overwritten
    should_output_files = output_files
    output_files = list()  # This is now the list of output file paths
    log_output_files = list()
    file_data = pd.DataFrame()
    deduplicated_unique_table_markdown = ""
    # Use provided values if available, otherwise use defaults
    if data_file_names_textbox is None:
        data_file_names_textbox = "dataframe"
    if total_number_of_batches is None:
        total_number_of_batches = 1

    # Validate that required columns exist in reference_df
    if "Response References" not in reference_df.columns:
        raise ValueError(
            "reference_df must contain 'Response References' column. "
            f"Available columns: {list(reference_df.columns)}"
        )

    # Add 'Topic number' column if it doesn't exist (row number starting from 1)
    if "Topic number" not in topic_summary_df.columns:
        if not topic_summary_df.empty:
            topic_summary_df["Topic number"] = range(1, len(topic_summary_df) + 1)
        else:
            # If empty, create empty column - it will be populated when DataFrame is recreated
            topic_summary_df["Topic number"] = pd.Series(dtype="int64")

    if "Response References" in reference_df.columns:
        # Convert float or str to int
        reference_df["Response References"] = (
            reference_df["Response References"].astype(float).astype(int)
        )
    if "Start row of group" in reference_df.columns:
        # Convert float or str to int
        reference_df["Start row of group"] = (
            reference_df["Start row of group"].astype(float).astype(int)
        )

    if (len(reference_df["Response References"].unique()) == 1) | (
        len(topic_summary_df["Topic number"].unique()) == 1
    ):
        print(
            "Data file outputs are too short for deduplicating. Returning original data."
        )

        # Get file name without extension and create proper output paths
        reference_table_file_name_no_ext = get_file_name_no_ext(
            reference_table_file_name
        )
        unique_topics_table_file_name_no_ext = get_file_name_no_ext(
            unique_topics_table_file_name
        )

        # Create output paths with _dedup suffix to match normal path
        reference_file_out_path = (
            output_folder + reference_table_file_name_no_ext + "_dedup.csv"
        )
        unique_topics_file_out_path = (
            output_folder + unique_topics_table_file_name_no_ext + "_dedup.csv"
        )

        if should_output_files == "True":
            # Save the DataFrames to CSV files
            reference_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                reference_file_out_path, index=None, encoding="utf-8-sig"
            )
            topic_summary_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                unique_topics_file_out_path, index=None, encoding="utf-8-sig"
            )
            output_files.append(reference_file_out_path)
            output_files.append(unique_topics_file_out_path)

        # Create markdown output for display
        topic_summary_df_revised_display = topic_summary_df.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )
        deduplicated_unique_table_markdown = (
            topic_summary_df_revised_display.to_markdown(index=False)
        )

        return (
            reference_df,
            topic_summary_df,
            output_files,
            log_output_files,
            deduplicated_unique_table_markdown,
        )

    # For checking that data is not lost during the process
    initial_unique_references = len(reference_df["Response References"].unique())

    if topic_summary_df.empty:
        topic_summary_df = create_topic_summary_df_from_reference_table(
            reference_df, sentiment_checkbox=sentiment_checkbox
        )

        # Then merge the topic numbers back to the original dataframe
        reference_df = reference_df.merge(
            topic_summary_df[
                ["General topic", "Subtopic", "Sentiment", "Topic number"]
            ],
            on=["General topic", "Subtopic", "Sentiment"],
            how="left",
        )

    # Check if in_data_files is not empty (handles both DataFrame and list)
    has_data_files = (
        (isinstance(in_data_files, pd.DataFrame) and not in_data_files.empty)
        or (isinstance(in_data_files, list) and len(in_data_files) > 0)
        or (not isinstance(in_data_files, (pd.DataFrame, list)) and in_data_files)
    )
    if has_data_files and chosen_cols:
        # Check if in_data_files is already a DataFrame
        if isinstance(in_data_files, pd.DataFrame):
            # Use the DataFrame directly
            file_data = in_data_files.copy()
            # Filter to chosen columns if specified
            if chosen_cols and isinstance(chosen_cols, list):
                # Ensure all chosen columns exist in the DataFrame
                available_cols = [
                    col for col in chosen_cols if col in file_data.columns
                ]
                if available_cols:
                    file_data = file_data[available_cols]
                else:
                    print(
                        f"Warning: None of the chosen columns {chosen_cols} found in DataFrame. Using all columns."
                    )

        else:
            # Load from file path(s) as before
            file_data, data_file_names_textbox, total_number_of_batches = (
                load_in_data_file(
                    in_data_files, chosen_cols, total_number_of_batches, in_excel_sheets
                )
            )
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)
        # raise Exception(out_message)

    # Run through this x times to try to get all duplicate topics
    if deduplicate_topics == "Yes":
        if "Group" not in reference_df.columns:
            reference_df["Group"] = "All"
        for i in range(0, 8):
            if merge_sentiment == "No":
                if merge_general_topics == "No":
                    reference_df["old_category"] = (
                        reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    )
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    # Create an empty list to store results from each group
                    results = list()
                    # Iterate over each group instead of using .apply()
                    for name, group in reference_df_unique.groupby(
                        ["General topic", "Sentiment", "Group"]
                    ):
                        # Run your function on the 'group' DataFrame
                        result = deduplicate_categories(
                            group["Subtopic"],
                            group["Sentiment"],
                            reference_df,
                            general_topic_series=group["General topic"],
                            merge_general_topics="No",
                            threshold=score_threshold,
                        )
                        results.append(result)

                    # Concatenate all the results into a single DataFrame
                    deduplicated_topic_map_df = pd.concat(results).reset_index(
                        drop=True
                    )

                else:
                    # This case should allow cross-topic matching but is still grouping by Sentiment
                    reference_df["old_category"] = (
                        reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    )
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    results = list()
                    for name, group in reference_df_unique.groupby("Sentiment"):
                        result = deduplicate_categories(
                            group["Subtopic"],
                            group["Sentiment"],
                            reference_df,
                            general_topic_series=None,
                            merge_general_topics="Yes",
                            threshold=score_threshold,
                        )
                        results.append(result)
                    deduplicated_topic_map_df = pd.concat(results).reset_index(
                        drop=True
                    )

            else:
                if merge_general_topics == "No":
                    reference_df["old_category"] = (
                        reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    )
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    results = list()
                    for name, group in reference_df_unique.groupby("General topic"):
                        result = deduplicate_categories(
                            group["Subtopic"],
                            group["Sentiment"],
                            reference_df,
                            general_topic_series=group["General topic"],
                            merge_general_topics="No",
                            merge_sentiment=merge_sentiment,
                            threshold=score_threshold,
                        )
                        results.append(result)
                    deduplicated_topic_map_df = pd.concat(results).reset_index(
                        drop=True
                    )

                else:
                    reference_df["old_category"] = (
                        reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    )
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    deduplicated_topic_map_df = deduplicate_categories(
                        reference_df_unique["Subtopic"],
                        reference_df_unique["Sentiment"],
                        reference_df,
                        general_topic_series=None,
                        merge_general_topics="Yes",
                        merge_sentiment=merge_sentiment,
                        threshold=score_threshold,
                    ).reset_index(drop=True)

            if deduplicated_topic_map_df["deduplicated_category"].isnull().all():
                print("No deduplicated categories found, skipping the following code.")

            else:
                # Remove rows where 'deduplicated_category' is blank or NaN
                deduplicated_topic_map_df = deduplicated_topic_map_df.loc[
                    (
                        deduplicated_topic_map_df["deduplicated_category"].str.strip()
                        != ""
                    )
                    & ~(deduplicated_topic_map_df["deduplicated_category"].isnull()),
                    ["old_category", "deduplicated_category", "match_score"],
                ]

                reference_df = reference_df.merge(
                    deduplicated_topic_map_df, on="old_category", how="left"
                )

                reference_df.rename(
                    columns={"Subtopic": "Subtopic_old", "Sentiment": "Sentiment_old"},
                    inplace=True,
                )
                # Extract subtopic and sentiment from deduplicated_category
                reference_df["Subtopic"] = reference_df[
                    "deduplicated_category"
                ].str.extract(r"^(.*?) \|")[
                    0
                ]  # Extract subtopic
                reference_df["Sentiment"] = reference_df[
                    "deduplicated_category"
                ].str.extract(r"\| (.*)$")[
                    0
                ]  # Extract sentiment

                # Combine with old values to ensure no data is lost
                reference_df["Subtopic"] = reference_df[
                    "deduplicated_category"
                ].combine_first(reference_df["Subtopic_old"])
                reference_df["Sentiment"] = reference_df["Sentiment"].combine_first(
                    reference_df["Sentiment_old"]
                )

            reference_df = reference_df.rename(
                columns={"General Topic": "General topic"}, errors="ignore"
            )
            reference_df = reference_df[
                [
                    "Response References",
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                    "Summary",
                    "Start row of group",
                    "Group",
                ]
            ]

            if merge_general_topics == "Yes":
                # Replace General topic names for each Subtopic with that for the Subtopic with the most responses
                # Step 1: Count the number of occurrences for each General topic and Subtopic combination
                count_df = (
                    reference_df.groupby(["Subtopic", "General topic"])
                    .size()
                    .reset_index(name="Count")
                )

                # Step 2: Find the General topic with the maximum count for each Subtopic
                max_general_topic = count_df.loc[
                    count_df.groupby("Subtopic")["Count"].idxmax()
                ]

                # Step 3: Map the General topic back to the original DataFrame
                reference_df = reference_df.merge(
                    max_general_topic[["Subtopic", "General topic"]],
                    on="Subtopic",
                    suffixes=("", "_max"),
                    how="left",
                )

                reference_df["General topic"] = reference_df[
                    "General topic_max"
                ].combine_first(reference_df["General topic"])

            if merge_sentiment == "Yes":
                # Step 1: Count the number of occurrences for each General topic and Subtopic combination
                count_df = (
                    reference_df.groupby(["Subtopic", "Sentiment"])
                    .size()
                    .reset_index(name="Count")
                )

                # Step 2: Determine the number of unique Sentiment values for each Subtopic
                unique_sentiments = (
                    count_df.groupby("Subtopic")["Sentiment"]
                    .nunique()
                    .reset_index(name="UniqueCount")
                )

                # Step 3: Update Sentiment to 'Mixed' where there is more than one unique sentiment
                reference_df = reference_df.merge(
                    unique_sentiments, on="Subtopic", how="left"
                )
                reference_df["Sentiment"] = reference_df.apply(
                    lambda row: "Mixed" if row["UniqueCount"] > 1 else row["Sentiment"],
                    axis=1,
                )

                # Clean up the DataFrame by dropping the UniqueCount column
                reference_df.drop(columns=["UniqueCount"], inplace=True)

            # print("reference_df:", reference_df)
            reference_df = reference_df[
                [
                    "Response References",
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                    "Summary",
                    "Start row of group",
                    "Group",
                ]
            ]

        # Update reference summary column with all summaries
        reference_df["Summary"] = reference_df.groupby(
            ["Response References", "General topic", "Subtopic", "Sentiment"]
        )["Summary"].transform(" <br> ".join)

        # Check that we have not inadvertantly removed some data during the above process
        end_unique_references = len(reference_df["Response References"].unique())

        if initial_unique_references != end_unique_references:
            raise Exception(
                f"Number of unique references changed during processing: Initial={initial_unique_references}, Final={end_unique_references}"
            )

        # Drop duplicates in the reference table - each comment should only have the same topic referred to once
        reference_df.drop_duplicates(
            ["Response References", "General topic", "Subtopic", "Sentiment"],
            inplace=True,
        )

        # Before recreating topic_summary_df, check if input had Group information
        # If input topic_summary_df doesn't have Group or all have same Group, normalize reference_df Group
        input_has_group = "Group" in topic_summary_df.columns
        input_unique_groups = (
            topic_summary_df["Group"].nunique() if input_has_group else 0
        )

        # Check reference_df Group values before normalization
        ref_df_has_group = "Group" in reference_df.columns
        ref_df_unique_groups_before = (
            reference_df["Group"].nunique() if ref_df_has_group else 0
        )

        # If input didn't have meaningful Group distinction, normalize to single Group
        if not input_has_group or input_unique_groups <= 1:
            # Get the most common Group value from reference_df, or use "All" if not present
            if ref_df_has_group and not reference_df["Group"].empty:
                most_common_group = (
                    reference_df["Group"].mode()[0]
                    if len(reference_df["Group"].mode()) > 0
                    else "All"
                )
            else:
                most_common_group = "All"
            # Normalize all Groups to the most common one to prevent topic count increase
            reference_df["Group"] = most_common_group

            # Verify normalization worked
            ref_df_unique_groups_after = reference_df["Group"].nunique()
            if ref_df_unique_groups_before > 1 and ref_df_unique_groups_after > 1:
                print(
                    f"Warning: Group normalization may have failed. "
                    f"reference_df had {ref_df_unique_groups_before} unique Groups before normalization, "
                    f"and {ref_df_unique_groups_after} after. Forcing all to '{most_common_group}'."
                )
                reference_df["Group"] = most_common_group

        # Remake topic_summary_df based on new reference_df
        # Rebuild topic_summary_df from updated reference_df to ensure consistency
        # This ensures the topic count reflects the actual merged state
        if not reference_df.empty:
            # Check if Response References are present and not all empty
            if "Response References" in reference_df.columns:
                non_empty_refs = reference_df["Response References"].notna() & (
                    reference_df["Response References"].astype(str).str.strip() != ""
                )
                if not non_empty_refs.any():
                    print(
                        "Warning: All Response References are empty in reference_df. "
                        "Number of responses will be 0 for all topics."
                    )
            topic_summary_df = create_topic_summary_df_from_reference_table(
                reference_df, sentiment_checkbox=sentiment_checkbox
            )
        else:
            print(
                "Warning: reference_df is empty after deduplication. "
                "Cannot recreate topic_summary_df. Number of responses will be 0 for all topics."
            )
            # Create an empty topic_summary_df with the expected structure
            topic_summary_df = pd.DataFrame()

        # Then merge the topic numbers back to the original dataframe
        # Only merge if both dataframes are not empty
        if not reference_df.empty and not topic_summary_df.empty:
            reference_df = reference_df.merge(
                topic_summary_df[
                    ["General topic", "Subtopic", "Sentiment", "Group", "Topic number"]
                ],
                on=["General topic", "Subtopic", "Sentiment", "Group"],
                how="left",
            )

    else:
        print("Topics have not beeen deduplicated")

    reference_table_file_name_no_ext = get_file_name_no_ext(reference_table_file_name)
    unique_topics_table_file_name_no_ext = get_file_name_no_ext(
        unique_topics_table_file_name
    )

    if not file_data.empty:
        basic_response_data = get_basic_response_data(file_data, chosen_cols)
        reference_df_pivot = convert_reference_table_to_pivot_table(
            reference_df, basic_response_data
        )

        reference_pivot_file_path = (
            output_folder + reference_table_file_name_no_ext + "_pivot_dedup.csv"
        )
        if should_output_files == "True":
            reference_df_pivot.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                reference_pivot_file_path, index=None, encoding="utf-8-sig"
            )
            log_output_files.append(reference_pivot_file_path)

    reference_file_out_path = (
        output_folder + reference_table_file_name_no_ext + "_dedup.csv"
    )
    unique_topics_file_out_path = (
        output_folder + unique_topics_table_file_name_no_ext + "_dedup.csv"
    )

    if should_output_files == "True":
        reference_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
            reference_file_out_path, index=None, encoding="utf-8-sig"
        )
        topic_summary_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
            unique_topics_file_out_path, index=None, encoding="utf-8-sig"
        )

        output_files.append(reference_file_out_path)
        output_files.append(unique_topics_file_out_path)

    # Outputs for markdown table output
    topic_summary_df_revised_display = topic_summary_df.apply(
        lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
    )
    deduplicated_unique_table_markdown = topic_summary_df_revised_display.to_markdown(
        index=False
    )

    print("Deduplication task successfully completed")

    return (
        reference_df,
        topic_summary_df,
        output_files,
        log_output_files,
        deduplicated_unique_table_markdown,
    )


def deduplicate_topics_llm(
    reference_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    reference_table_file_name: str,
    unique_topics_table_file_name: str,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    model_source: str,
    local_model=None,
    tokenizer=None,
    assistant_model=None,
    in_excel_sheets: str = "",
    merge_sentiment: str = "No",
    merge_general_topics: str = "No",
    in_data_files: Union[List[str], pd.DataFrame] = list(),
    chosen_cols: List[str] = "",
    output_folder: str = OUTPUT_FOLDER,
    candidate_topics=None,
    azure_endpoint: str = "",
    output_debug_files: str = "False",
    api_url: str = None,
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    azure_api_key_textbox: str = "",
    model_name_map: dict = model_name_map,
    output_files: str = "True",
    sentiment_checkbox: str = "Negative or Positive",
):
    """
    Deduplicate topics using LLM semantic understanding to identify and merge similar topics.

    Args:
        reference_df (pd.DataFrame): DataFrame containing reference data with topics.
        topic_summary_df (pd.DataFrame): DataFrame summarizing unique topics.
        reference_table_file_name (str): Base file name for the output reference table.
        unique_topics_table_file_name (str): Base file name for the output unique topics table.
        model_choice (str): The LLM model to use for deduplication.
        in_api_key (str): Google API key for the LLM service (for Gemini models).
        temperature (float): Temperature setting for the LLM.
        model_source (str): Source of the model (AWS, Gemini, Local, etc.).
        local_model: Local model instance (if using local model).
        tokenizer: Tokenizer for local model.
        assistant_model: Assistant model for speculative decoding.
        in_excel_sheets (str, optional): Comma-separated list of Excel sheet names to load. Defaults to "".
        merge_sentiment (str, optional): Whether to merge topics regardless of sentiment ("Yes" or "No"). Defaults to "No".
        merge_general_topics (str, optional): Whether to merge topics across different general topics ("Yes" or "No"). Defaults to "No".
        in_data_files (Union[List[str], pd.DataFrame], optional): List of input data file paths or a pandas DataFrame. If a DataFrame is provided, it will be used directly without loading from file. Defaults to [].
        chosen_cols (List[str], optional): List of chosen columns from the input data files. Defaults to "".
        output_folder (str, optional): Folder path to save output files. Defaults to OUTPUT_FOLDER.
        candidate_topics (optional): Candidate topics file for zero-shot guidance. Defaults to None.
        azure_endpoint (str, optional): Azure endpoint for the LLM. Defaults to "".
        output_debug_files (str, optional): Whether to output debug files. Defaults to "False".
        api_url (str, optional): API URL for inference-server models. Defaults to None.
        aws_access_key_textbox (str, optional): AWS access key for Bedrock. Defaults to "".
        aws_secret_key_textbox (str, optional): AWS secret key for Bedrock. Defaults to "".
        aws_region_textbox (str, optional): AWS region for Bedrock. Defaults to "".
        azure_api_key_textbox (str, optional): Azure API key for Azure/OpenAI models. Defaults to "".
        model_name_map (dict, optional): Mapping of model names to their configurations. Defaults to model_name_map from config.
        output_files (str, optional): Whether to output files ("True" or "False"). Defaults to "True".
        sentiment_checkbox (str, optional): Sentiment analysis option ("Negative or Positive", "Negative, Neutral, or Positive", or "Do not assess sentiment"). Defaults to "Negative or Positive".
        output_files (str, optional): Whether to output files ("True" or "False"). Defaults to "True".
    """

    # Save the parameter value before it gets overwritten
    should_output_files = output_files
    output_files = list()  # This is now the list of output file paths
    log_output_files = list()
    file_data = pd.DataFrame()
    deduplicated_unique_table_markdown = ""

    if "Response References" in reference_df.columns:
        # Convert float or str to int
        reference_df["Response References"] = (
            reference_df["Response References"].astype(float).astype(int)
        )
    if "Start row of group" in reference_df.columns:
        # Convert float or str to int
        reference_df["Start row of group"] = (
            reference_df["Start row of group"].astype(float).astype(int)
        )

    # Check if data is too short for deduplication
    if (len(reference_df["Response References"].unique()) == 1) | (
        len(topic_summary_df["Topic number"].unique()) == 1
    ):
        print(
            "Data file outputs are too short for deduplicating. Returning original data."
        )

        # Get file name without extension and create proper output paths
        reference_table_file_name_no_ext = get_file_name_no_ext(
            reference_table_file_name
        )
        unique_topics_table_file_name_no_ext = get_file_name_no_ext(
            unique_topics_table_file_name
        )

        # Create output paths with _dedup suffix to match normal path
        reference_file_out_path = (
            output_folder + reference_table_file_name_no_ext + "_dedup.csv"
        )
        unique_topics_file_out_path = (
            output_folder + unique_topics_table_file_name_no_ext + "_dedup.csv"
        )

        if should_output_files == "True":
            # Save the DataFrames to CSV files
            reference_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                reference_file_out_path, index=None, encoding="utf-8-sig"
            )
            topic_summary_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                unique_topics_file_out_path, index=None, encoding="utf-8-sig"
            )

            output_files.append(reference_file_out_path)
            output_files.append(unique_topics_file_out_path)

        # Create markdown output for display
        topic_summary_df_revised_display = topic_summary_df.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )
        deduplicated_unique_table_markdown = (
            topic_summary_df_revised_display.to_markdown(index=False)
        )

        # Compare input vs output for early return case
        if not topic_summary_df.empty:
            input_unique_topics = topic_summary_df.drop_duplicates(
                subset=["General topic", "Subtopic"]
            ).shape[0]
            output_unique_topics = topic_summary_df.drop_duplicates(
                subset=["General topic", "Subtopic"]
            ).shape[0]
            print(
                f"Topic count comparison (early return): Input had {input_unique_topics} unique 'General topic' | 'Subtopic' combinations, "
                f"Output has {output_unique_topics} unique 'General topic' | 'Subtopic' combinations. "
                f"No deduplication performed (data too short)."
            )

        # Return with token counts set to 0 for early return
        return (
            reference_df,
            topic_summary_df,
            output_files,
            log_output_files,
            deduplicated_unique_table_markdown,
            0,  # input_tokens
            0,  # output_tokens
            0,  # number_of_calls
            0.0,  # estimated_time_taken
        )

    # For checking that data is not lost during the process
    initial_unique_references = len(reference_df["Response References"].unique())

    # Capture input topic count for comparison
    if not topic_summary_df.empty:
        input_unique_topics = topic_summary_df.drop_duplicates(
            subset=["General topic", "Subtopic"]
        ).shape[0]
    else:
        input_unique_topics = 0

    # Create topic summary if it doesn't exist
    if topic_summary_df.empty:
        topic_summary_df = create_topic_summary_df_from_reference_table(reference_df)
        # Update input count if we just created the topic summary
        input_unique_topics = topic_summary_df.drop_duplicates(
            subset=["General topic", "Subtopic"]
        ).shape[0]

        topic_summary_df.drop(
            ["1", "2", "3", "Response References"],
            axis=1,
            errors="ignore",
            inplace=True,
        )

        if "Topic number" not in reference_df.columns:
            # Merge topic numbers back to the original dataframe
            reference_df = reference_df.merge(
                topic_summary_df[
                    ["General topic", "Subtopic", "Sentiment", "Topic number"]
                ],
                on=["General topic", "Subtopic", "Sentiment"],
                how="left",
            )

    orig_unique_topics_file_out_path = (
        output_folder
        + get_file_name_no_ext(unique_topics_table_file_name)
        + "_orig_pre_dedup.csv"
    )
    if should_output_files == "True":
        topic_summary_df.drop(
            ["1", "2", "3", "Response References"], axis=1, errors="ignore"
        ).to_csv(orig_unique_topics_file_out_path, index=None, encoding="utf-8-sig")

    # Load data files if provided
    # Check if in_data_files is not empty (handles both DataFrame and list)
    has_data_files = (
        (isinstance(in_data_files, pd.DataFrame) and not in_data_files.empty)
        or (isinstance(in_data_files, list) and len(in_data_files) > 0)
        or (not isinstance(in_data_files, (pd.DataFrame, list)) and in_data_files)
    )
    if has_data_files and chosen_cols:
        # Check if in_data_files is already a DataFrame
        if isinstance(in_data_files, pd.DataFrame):
            # Use the DataFrame directly
            file_data = in_data_files.copy()
            # Filter to chosen columns if specified
            if chosen_cols and isinstance(chosen_cols, list):
                # Ensure all chosen columns exist in the DataFrame
                available_cols = [
                    col for col in chosen_cols if col in file_data.columns
                ]
                if available_cols:
                    file_data = file_data[available_cols]
                else:
                    print(
                        f"Warning: None of the chosen columns {chosen_cols} found in DataFrame. Using all columns."
                    )
            data_file_names_textbox = "dataframe"
            total_number_of_batches = 1
        else:
            # Load from file path(s) as before
            file_data, data_file_names_textbox, total_number_of_batches = (
                load_in_data_file(in_data_files, chosen_cols, 1, in_excel_sheets)
            )
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)

    # Process candidate topics if provided
    candidate_topics_table = ""
    if candidate_topics is not None:
        try:

            # Read and process candidate topics
            # Handle both string paths (CLI) and gr.FileData objects (Gradio)
            candidate_topics_path = (
                candidate_topics
                if isinstance(candidate_topics, str)
                else getattr(candidate_topics, "name", None)
            )
            if candidate_topics_path is None:
                raise ValueError(
                    "candidate_topics must be a file path string or a FileData object with a 'name' attribute"
                )
            candidate_topics_df = read_file(candidate_topics_path)
            candidate_topics_df = candidate_topics_df.fillna("")
            candidate_topics_df = candidate_topics_df.astype(str)

            # Generate zero-shot topics DataFrame
            zero_shot_topics_df = generate_zero_shot_topics_df(
                candidate_topics_df, "No", False
            )

            if not zero_shot_topics_df.empty:
                candidate_topics_table = zero_shot_topics_df[
                    ["General topic", "Subtopic"]
                ].to_markdown(index=False)
                print(
                    f"Found {len(zero_shot_topics_df)} candidate topics to consider during deduplication"
                )
        except Exception as e:
            print(f"Error processing candidate topics: {e}")
            candidate_topics_table = ""

    # Determine if sentiment should be included based on sentiment_checkbox
    include_sentiment = sentiment_checkbox != "Do not assess sentiment"

    # Normalize topic names for LLM comparison to avoid capitalization-only "merges"
    # Create a copy of topic_summary_df with normalized topic names for the LLM
    # This prevents the LLM from seeing capitalization differences as different topics
    # Use the shared normalize_topic_name_for_llm function from helper_functions
    # Create normalized version for LLM (but keep original for mapping back)
    topics_for_llm = topic_summary_df.copy()
    topics_for_llm["General topic"] = topics_for_llm["General topic"].apply(
        normalize_topic_name_for_llm
    )
    topics_for_llm["Subtopic"] = topics_for_llm["Subtopic"].apply(
        normalize_topic_name_for_llm
    )

    # Remove duplicates after normalization (these are the same topic with different capitalization)
    # This prevents the LLM from seeing them as separate topics
    if include_sentiment:
        topics_for_llm = topics_for_llm.drop_duplicates(
            subset=["General topic", "Subtopic", "Sentiment"]
        )
    else:
        topics_for_llm = topics_for_llm.drop_duplicates(
            subset=["General topic", "Subtopic"]
        )

    # Prepare topics table for LLM analysis (conditionally include sentiment)
    if include_sentiment:
        topics_table = topics_for_llm[
            ["General topic", "Subtopic", "Sentiment", "Number of responses"]
        ].to_markdown(index=False)
        sentiment_text = ", and Sentiment classifications"
        sentiment_columns = "\n3. 'Original Sentiment' - The current sentiment"
        merged_sentiment_columns = "\n4. 'Merged Sentiment' - The consolidated sentiment (use 'Mixed' if sentiments differ)"
    else:
        topics_table = topics_for_llm[
            ["General topic", "Subtopic", "Number of responses"]
        ].to_markdown(index=False)
        sentiment_text = ""
        sentiment_columns = ""
        merged_sentiment_columns = ""

    # Format the prompt with candidate topics if available
    if candidate_topics_table:
        formatted_prompt = llm_deduplication_prompt_with_candidates.format(
            topics_table=topics_table,
            candidate_topics_table=candidate_topics_table,
            max_number_of_topics=max_number_of_topics,
            sentiment_text=sentiment_text,
            sentiment_columns=sentiment_columns,
            merged_sentiment_columns=merged_sentiment_columns,
        )
    else:
        formatted_prompt = llm_deduplication_prompt.format(
            topics_table=topics_table,
            max_number_of_topics=max_number_of_topics,
            sentiment_text=sentiment_text,
            sentiment_columns=sentiment_columns,
            merged_sentiment_columns=merged_sentiment_columns,
        )

    # Initialise conversation history
    conversation_history = list()
    whole_conversation = list()
    whole_conversation_metadata = list()

    # Set up model clients based on model source
    if "Gemini" in model_source:
        print("Using Gemini model:", model_choice)
        client, config = construct_gemini_generative_model(
            in_api_key,
            temperature,
            model_choice,
            llm_deduplication_system_prompt,
            max_tokens,
            LLM_SEED,
        )
        bedrock_runtime = None
    elif "Azure/OpenAI" in model_source:
        print("Using Azure/OpenAI AI Inference model:", model_choice)
        if azure_api_key_textbox:
            os.environ["AZURE_INFERENCE_CREDENTIAL"] = azure_api_key_textbox
        client, config = construct_azure_client(
            in_api_key=azure_api_key_textbox, endpoint=azure_endpoint
        )
        bedrock_runtime = None
    elif "AWS" in model_source:
        print("Using AWS Bedrock model:", model_choice)
        bedrock_runtime = connect_to_bedrock_runtime(
            model_name_map,
            model_choice,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            aws_region_textbox,
        )
        client = None
        config = None
    elif "Local" in model_source:
        print("Using local model:", model_choice)
        client = None
        config = None
        bedrock_runtime = None
    elif "inference-server" in model_source:
        print("Using inference-server model:", model_choice)
        client = None
        config = None
        bedrock_runtime = None
        # api_url is already passed to call_llm_with_markdown_table_checks
        if api_url is None:
            raise ValueError(
                "api_url is required when model_source is 'inference-server'"
            )
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Call LLM to get deduplication suggestions
    print("Calling LLM for topic deduplication analysis...")

    # Use the existing call_llm_with_markdown_table_checks function
    (
        responses,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        response_text,
    ) = call_llm_with_markdown_table_checks(
        batch_prompts=[formatted_prompt],
        system_prompt=llm_deduplication_system_prompt,
        conversation_history=conversation_history,
        whole_conversation=whole_conversation,
        whole_conversation_metadata=whole_conversation_metadata,
        client=client,
        client_config=config,
        model_choice=model_choice,
        temperature=temperature,
        reported_batch_no=1,
        local_model=local_model,
        tokenizer=tokenizer,
        bedrock_runtime=bedrock_runtime,
        model_source=model_source,
        MAX_OUTPUT_VALIDATION_ATTEMPTS=3,
        assistant_prefill="",
        master=False,
        CHOSEN_LOCAL_MODEL_TYPE=CHOSEN_LOCAL_MODEL_TYPE,
        random_seed=LLM_SEED,
        api_url=api_url,
    )

    # Generate debug files if enabled
    if should_output_files == "True":
        try:
            # Create batch file path details for debug files
            batch_file_path_details = (
                get_file_name_no_ext(reference_table_file_name) + "_llm_dedup"
            )
            model_choice_clean_short = (
                model_choice.replace("/", "_").replace(":", "_").replace(".", "_")
            )

            # Create full prompt for debug output
            full_prompt = llm_deduplication_system_prompt + "\n" + formatted_prompt

            # Write debug files
            (
                current_prompt_content_logged,
                current_summary_content_logged,
                current_conversation_content_logged,
                current_metadata_content_logged,
            ) = process_debug_output_iteration(
                OUTPUT_DEBUG_FILES,
                output_folder,
                batch_file_path_details,
                model_choice_clean_short,
                full_prompt,
                response_text,
                whole_conversation,
                whole_conversation_metadata,
                log_output_files,
                task_type="llm_deduplication",
            )

            print("Debug files written for LLM deduplication analysis")

        except Exception as e:
            print(f"Error writing debug files for LLM deduplication: {e}")

    # Parse the LLM response to extract merge suggestions
    merge_suggestions_df = (
        pd.DataFrame()
    )  # Initialize empty DataFrame for analysis results
    num_merges_applied = 0
    # Track unique topic combinations being removed and added
    # This helps explain the discrepancy between merge operations and actual topic reductions
    unique_original_combinations = set()
    unique_merged_combinations = set()

    try:
        # Extract the markdown table from the response
        table_match = re.search(
            r"\|.*\|.*\n\|.*\|.*\n(\|.*\|.*\n)*", response_text, re.MULTILINE
        )
        if table_match:
            table_text = table_match.group(0)

            merge_suggestions_df = pd.read_csv(
                StringIO(table_text), sep="|", skipinitialspace=True
            )

            # Clean up the DataFrame
            # Remove empty columns (created from leading/trailing pipes in markdown)
            merge_suggestions_df = merge_suggestions_df.dropna(
                axis=1, how="all"
            )  # Remove empty columns
            merge_suggestions_df.columns = merge_suggestions_df.columns.str.strip()

            # Remove columns that are empty strings or just whitespace (from markdown table parsing)
            # This fixes issues where leading/trailing pipes create empty column names
            # Also remove columns with invalid names like 'end', '>' that can appear from parsing errors
            expected_column_keywords = [
                "Original General topic",
                "Original Subtopic",
                "Merged General topic",
                "Merged Subtopic",
                "Merge Reason",
                "Original Sentiment",
                "Merged Sentiment",
            ]

            valid_columns = []
            found_expected_column = False
            for col in merge_suggestions_df.columns:
                col_stripped = str(col).strip()
                # Check if this column matches an expected column name
                is_expected = any(
                    keyword.lower() in col_stripped.lower()
                    for keyword in expected_column_keywords
                )

                if is_expected:
                    found_expected_column = True
                    valid_columns.append(col)
                elif found_expected_column:
                    # Once we've found expected columns, keep subsequent columns too
                    # (in case there are additional valid columns)
                    if col_stripped and col_stripped not in ["", "nan", "None"]:
                        valid_columns.append(col)
                elif not found_expected_column:
                    # Before finding expected columns, filter out obvious artifacts
                    # Exclude single character columns and common parsing artifacts
                    if (
                        col_stripped
                        and col_stripped not in ["", "nan", "None", "end", ">"]
                        and len(col_stripped) > 1
                    ):
                        valid_columns.append(col)

            # If we found expected columns, use only valid columns; otherwise keep all non-empty
            if found_expected_column and valid_columns:
                merge_suggestions_df = merge_suggestions_df[valid_columns]
            elif not found_expected_column:
                # If no expected columns found, filter out obvious artifacts
                valid_columns = [
                    col
                    for col in merge_suggestions_df.columns
                    if str(col).strip() not in ["", "nan", "None", "end", ">"]
                    and len(str(col).strip()) > 1
                ]
                if valid_columns:
                    merge_suggestions_df = merge_suggestions_df[valid_columns]

            # Also remove any columns where all values are empty strings after stripping
            cols_to_drop = []
            for col in merge_suggestions_df.columns:
                if merge_suggestions_df[col].astype(str).str.strip().eq("").all():
                    cols_to_drop.append(col)
            if cols_to_drop:
                merge_suggestions_df = merge_suggestions_df.drop(columns=cols_to_drop)

            # Remove rows where all values are NaN
            merge_suggestions_df = merge_suggestions_df.dropna(how="all")

            # Convert all columns to string to avoid float/NaN issues when calling .strip()
            for col in merge_suggestions_df.columns:
                merge_suggestions_df[col] = (
                    merge_suggestions_df[col]
                    .astype(str)
                    .replace("nan", "")
                    .replace("NaN", "")
                    .replace("None", "")
                )

            # Filter out markdown table divider rows (rows that are primarily dashes/hyphens)
            # These are false positives from parsing the markdown table structure
            def is_divider_row(row):
                """Check if a row is a markdown table divider (contains mostly dashes/hyphens)."""
                row_str = " ".join(str(val) for val in row.values if pd.notna(val))
                # Remove whitespace and check if it's mostly dashes/hyphens
                row_clean = row_str.replace(" ", "").replace("|", "").strip()
                # If the row is empty or consists mostly of dashes/hyphens, it's a divider
                if not row_clean or len(row_clean) == 0:
                    return True
                # Check if more than 80% of characters are dashes/hyphens
                dash_count = sum(1 for c in row_clean if c in ["-", "_", "="])
                return dash_count > 0.8 * len(row_clean) if len(row_clean) > 0 else True

            # Remove divider rows
            merge_suggestions_df = merge_suggestions_df[
                ~merge_suggestions_df.apply(is_divider_row, axis=1)
            ]

            if not merge_suggestions_df.empty:
                print(
                    f"LLM identified {len(merge_suggestions_df)} potential topic merges"
                )

                # Deduplicate merge suggestions to avoid processing the same merge multiple times
                # Normalize the original topic names for deduplication
                def normalize_for_dedup(name):
                    """Normalize topic name for deduplication comparison."""
                    if pd.isna(name) or name is None:
                        return ""
                    normalized = str(name)
                    normalized = initial_clean(normalized)
                    normalized = normalized.strip()
                    normalized = normalized.replace("\n", " ")
                    normalized = normalized.replace("\r", " ")
                    normalized = normalized.replace("/", " or ")
                    normalized = normalized.replace("&", " and ")
                    normalized = normalized.replace(" s ", "s ")
                    normalized = normalized.lower()
                    return normalized

                # Create normalized columns for deduplication
                if "Original General topic" in merge_suggestions_df.columns:
                    merge_suggestions_df["_orig_gen_norm"] = merge_suggestions_df[
                        "Original General topic"
                    ].apply(normalize_for_dedup)
                else:
                    merge_suggestions_df["_orig_gen_norm"] = ""

                if "Original Subtopic" in merge_suggestions_df.columns:
                    merge_suggestions_df["_orig_sub_norm"] = merge_suggestions_df[
                        "Original Subtopic"
                    ].apply(normalize_for_dedup)
                else:
                    merge_suggestions_df["_orig_sub_norm"] = ""

                if include_sentiment:
                    # Deduplicate based on normalized original topics and sentiment
                    if "Original Sentiment" in merge_suggestions_df.columns:
                        merge_suggestions_df["_orig_sent"] = merge_suggestions_df[
                            "Original Sentiment"
                        ].astype(str)
                    else:
                        merge_suggestions_df["_orig_sent"] = ""
                    initial_count = len(merge_suggestions_df)
                    merge_suggestions_df = merge_suggestions_df.drop_duplicates(
                        subset=["_orig_gen_norm", "_orig_sub_norm", "_orig_sent"],
                        keep="first",
                    )
                else:
                    # Deduplicate based on normalized original topics only
                    initial_count = len(merge_suggestions_df)
                    merge_suggestions_df = merge_suggestions_df.drop_duplicates(
                        subset=["_orig_gen_norm", "_orig_sub_norm"], keep="first"
                    )

                # Remove the temporary normalization columns
                merge_suggestions_df = merge_suggestions_df.drop(
                    columns=["_orig_gen_norm", "_orig_sub_norm", "_orig_sent"],
                    errors="ignore",
                )

                genuine_merges_count = len(merge_suggestions_df)
                duplicates_removed = initial_count - genuine_merges_count

                if duplicates_removed > 0:
                    print(
                        f"Removed {duplicates_removed} duplicate merge suggestion(s). "
                        f"Processing {genuine_merges_count} genuine merge(s)."
                    )
                else:
                    print(f"Processing {genuine_merges_count} genuine merge(s).")

                # Apply the merges to the reference_df
                # Helper function to normalize topic names for comparison
                # Uses the same transformations applied to topic names in reference_df
                def normalize_topic_name(name):
                    """Normalize topic name using the same transformations as reference_df topics for comparison."""
                    if pd.isna(name) or name is None:
                        return ""
                    # Apply the same cleaning and transformations used in llm_api_call.py
                    normalized = str(name)
                    normalized = initial_clean(normalized)
                    normalized = normalized.strip()
                    normalized = normalized.replace("\n", " ")
                    normalized = normalized.replace("\r", " ")
                    normalized = normalized.replace("/", " or ")
                    normalized = normalized.replace("&", " and ")
                    normalized = normalized.replace(" s ", "s ")
                    normalized = normalized.lower()
                    return normalized

                # OPTIMIZATION: Pre-compute normalized topic mappings for efficient lookup
                # Instead of normalizing every row for every merge suggestion, normalize unique combinations once
                # Create a mapping from (actual_general, actual_subtopic, sentiment) -> normalized_general, normalized_subtopic
                # This allows us to quickly find matching rows without repeated normalization

                # Get unique topic combinations from reference_df
                if include_sentiment:
                    unique_topics = reference_df[
                        ["General topic", "Subtopic", "Sentiment"]
                    ].drop_duplicates()
                    # Create normalized lookup dictionaries
                    # Map: (actual_general, actual_subtopic, sentiment) -> (normalized_general, normalized_subtopic)
                    topic_normalization_map = {}
                    for _, row in unique_topics.iterrows():
                        key = (
                            str(row["General topic"]),
                            str(row["Subtopic"]),
                            str(row["Sentiment"]),
                        )
                        topic_normalization_map[key] = (
                            normalize_topic_name(row["General topic"]),
                            normalize_topic_name(row["Subtopic"]),
                        )

                    # Create reverse lookup: (normalized_general, normalized_subtopic, sentiment) -> set of (actual_general, actual_subtopic, sentiment)
                    normalized_to_actual_map = {}
                    for key, (norm_gen, norm_sub) in topic_normalization_map.items():
                        norm_key = (
                            norm_gen,
                            norm_sub,
                            key[2],
                        )  # (normalized_gen, normalized_sub, sentiment)
                        if norm_key not in normalized_to_actual_map:
                            normalized_to_actual_map[norm_key] = set()
                        normalized_to_actual_map[norm_key].add(key)

                    # Create a mask lookup: for each row index, store its normalized topic combination
                    # This allows us to quickly find all rows matching a normalized topic
                    row_normalized_map = {}
                    for idx, row in reference_df.iterrows():
                        key = (
                            str(row["General topic"]),
                            str(row["Subtopic"]),
                            str(row["Sentiment"]),
                        )
                        if key in topic_normalization_map:
                            norm_gen, norm_sub = topic_normalization_map[key]
                            norm_key = (norm_gen, norm_sub, key[2])
                            if norm_key not in row_normalized_map:
                                row_normalized_map[norm_key] = []
                            row_normalized_map[norm_key].append(idx)
                else:
                    unique_topics = reference_df[
                        ["General topic", "Subtopic"]
                    ].drop_duplicates()
                    # Create normalized lookup dictionaries
                    topic_normalization_map = {}
                    for _, row in unique_topics.iterrows():
                        key = (str(row["General topic"]), str(row["Subtopic"]))
                        topic_normalization_map[key] = (
                            normalize_topic_name(row["General topic"]),
                            normalize_topic_name(row["Subtopic"]),
                        )

                    # Create reverse lookup: (normalized_general, normalized_subtopic) -> set of (actual_general, actual_subtopic)
                    normalized_to_actual_map = {}
                    for key, (norm_gen, norm_sub) in topic_normalization_map.items():
                        norm_key = (norm_gen, norm_sub)
                        if norm_key not in normalized_to_actual_map:
                            normalized_to_actual_map[norm_key] = set()
                        normalized_to_actual_map[norm_key].add(key)

                    # Create a mask lookup: for each normalized topic, store list of row indices
                    row_normalized_map = {}
                    for idx, row in reference_df.iterrows():
                        key = (str(row["General topic"]), str(row["Subtopic"]))
                        if key in topic_normalization_map:
                            norm_gen, norm_sub = topic_normalization_map[key]
                            norm_key = (norm_gen, norm_sub)
                            if norm_key not in row_normalized_map:
                                row_normalized_map[norm_key] = []
                            row_normalized_map[norm_key].append(idx)

                for _, row in merge_suggestions_df.iterrows():
                    # Safely extract and convert values to strings, handling NaN/float values
                    def safe_get_strip(row, key, default=""):
                        value = row.get(key, default)
                        if (
                            pd.isna(value)
                            or value is None
                            or str(value).lower() in ["nan", "none", ""]
                        ):
                            return default
                        return str(value).strip()

                    original_general = safe_get_strip(row, "Original General topic", "")
                    original_subtopic = safe_get_strip(row, "Original Subtopic", "")
                    merged_general = safe_get_strip(row, "Merged General topic", "")
                    merged_subtopic = safe_get_strip(row, "Merged Subtopic", "")

                    # Conditionally handle sentiment based on include_sentiment flag
                    if include_sentiment:
                        original_sentiment = safe_get_strip(
                            row, "Original Sentiment", ""
                        )
                        merged_sentiment = safe_get_strip(row, "Merged Sentiment", "")

                        # Check all required fields including sentiment
                        if all(
                            [
                                original_general,
                                original_subtopic,
                                original_sentiment,
                                merged_general,
                                merged_subtopic,
                                merged_sentiment,
                            ]
                        ):
                            # Find matching rows using pre-computed normalized lookup
                            normalized_orig_gen = normalize_topic_name(original_general)
                            normalized_orig_sub = normalize_topic_name(
                                original_subtopic
                            )
                            lookup_key = (
                                normalized_orig_gen,
                                normalized_orig_sub,
                                original_sentiment,
                            )

                            # Get row indices that match this normalized topic combination
                            matching_indices = row_normalized_map.get(lookup_key, [])
                            # Create boolean mask using index.isin for efficiency
                            mask = reference_df.index.isin(matching_indices)

                            if mask.any():
                                # Normalize merged topics (original topics already normalized above)
                                normalized_merged_gen = normalize_topic_name(
                                    merged_general
                                )
                                normalized_merged_sub = normalize_topic_name(
                                    merged_subtopic
                                )

                                # Skip if normalized versions are identical (only capitalization changed)
                                if (
                                    normalized_orig_gen == normalized_merged_gen
                                    and normalized_orig_sub == normalized_merged_sub
                                ):
                                    # This is just a capitalization change or self-merge, skip it
                                    print(
                                        f"Skipped self-merge: '{original_general}' | '{original_subtopic}' | '{original_sentiment}' -> "
                                        f"'{merged_general}' | '{merged_subtopic}' | '{merged_sentiment}' "
                                        f"(normalized versions are identical)"
                                    )
                                    continue

                                # Check if the merged combination already exists in reference_df (before applying the merge)
                                # This indicates a real merge (consolidation) vs just a rename
                                # Use pre-computed lookup for efficiency
                                merged_lookup_key = (
                                    normalized_merged_gen,
                                    normalized_merged_sub,
                                    merged_sentiment,
                                )
                                merged_exists_before = (
                                    merged_lookup_key in row_normalized_map
                                    and len(row_normalized_map[merged_lookup_key]) > 0
                                )

                                # Check if the merged combo is different from the original (not just renaming to itself)
                                is_different_combo = (
                                    normalized_orig_gen != normalized_merged_gen
                                    or normalized_orig_sub != normalized_merged_sub
                                    or original_sentiment != merged_sentiment
                                )

                                # It's a real consolidation if the merged combo already exists and is different from original
                                is_real_merge = (
                                    merged_exists_before and is_different_combo
                                )

                                # Update the matching rows (including sentiment)
                                # Use merged values which already have whitespace stripped and preserve original case
                                reference_df.loc[mask, "General topic"] = merged_general
                                reference_df.loc[mask, "Subtopic"] = merged_subtopic
                                reference_df.loc[mask, "Sentiment"] = merged_sentiment
                                num_merges_applied += 1

                                # Track unique combinations for reporting
                                orig_combo = (
                                    normalized_orig_gen,
                                    normalized_orig_sub,
                                    original_sentiment,
                                )
                                merged_combo = (
                                    normalized_merged_gen,
                                    normalized_merged_sub,
                                    merged_sentiment,
                                )
                                unique_original_combinations.add(orig_combo)
                                unique_merged_combinations.add(merged_combo)

                                merge_type = (
                                    "consolidated" if is_real_merge else "renamed"
                                )
                                print(
                                    f"Merged ({merge_type}): {original_general} | {original_subtopic} | {original_sentiment} -> {merged_general} | {merged_subtopic} | {merged_sentiment}"
                                )
                            else:
                                # Debug: show why merge failed
                                # Use pre-computed normalized topics from lookup
                                # Get available topics from pre-computed map for debugging
                                available_gen_topics = {
                                    norm_gen
                                    for (
                                        norm_gen,
                                        norm_sub,
                                        _,
                                    ) in row_normalized_map.keys()
                                }
                                available_sub_topics = {
                                    norm_sub
                                    for (
                                        norm_gen,
                                        norm_sub,
                                        _,
                                    ) in row_normalized_map.keys()
                                }
                                # Check if the normalized general topic exists
                                gen_exists = normalized_orig_gen in available_gen_topics
                                sub_exists = normalized_orig_sub in available_sub_topics
                                print(
                                    f"Failed to merge: '{original_general}' | '{original_subtopic}' | '{original_sentiment}' "
                                    f"(normalized: '{normalized_orig_gen}' | '{normalized_orig_sub}'). "
                                    f"General topic exists: {gen_exists}, Subtopic exists: {sub_exists}. "
                                    f"No matching rows found in reference_df."
                                )

                    else:
                        # Check required fields without sentiment
                        if all(
                            [
                                original_general,
                                original_subtopic,
                                merged_general,
                                merged_subtopic,
                            ]
                        ):
                            # Find matching rows using pre-computed normalized lookup
                            normalized_orig_gen = normalize_topic_name(original_general)
                            normalized_orig_sub = normalize_topic_name(
                                original_subtopic
                            )
                            lookup_key = (normalized_orig_gen, normalized_orig_sub)

                            # Get row indices that match this normalized topic combination
                            matching_indices = row_normalized_map.get(lookup_key, [])
                            # Create boolean mask using index.isin for efficiency
                            mask = reference_df.index.isin(matching_indices)

                            if mask.any():
                                # Normalize merged topics (original topics already normalized above)
                                normalized_merged_gen = normalize_topic_name(
                                    merged_general
                                )
                                normalized_merged_sub = normalize_topic_name(
                                    merged_subtopic
                                )

                                # Skip if normalized versions are identical (only capitalization changed)
                                if (
                                    normalized_orig_gen == normalized_merged_gen
                                    and normalized_orig_sub == normalized_merged_sub
                                ):
                                    # This is just a capitalization change or self-merge, skip it
                                    print(
                                        f"Skipped self-merge: '{original_general}' | '{original_subtopic}' -> "
                                        f"'{merged_general}' | '{merged_subtopic}' "
                                        f"(normalized versions are identical)"
                                    )
                                    continue

                                # Check if the merged combination already exists in reference_df (before applying the merge)
                                # This indicates a real merge (consolidation) vs just a rename
                                # Use pre-computed lookup for efficiency
                                merged_lookup_key = (
                                    normalized_merged_gen,
                                    normalized_merged_sub,
                                )
                                merged_exists_before = (
                                    merged_lookup_key in row_normalized_map
                                    and len(row_normalized_map[merged_lookup_key]) > 0
                                )

                                # Check if the merged combo is different from the original (not just renaming to itself)
                                is_different_combo = (
                                    normalized_orig_gen != normalized_merged_gen
                                    or normalized_orig_sub != normalized_merged_sub
                                )

                                # It's a real consolidation if the merged combo already exists and is different from original
                                is_real_merge = (
                                    merged_exists_before and is_different_combo
                                )

                                # Update the matching rows (without sentiment)
                                # Use merged values which already have whitespace stripped and preserve original case
                                reference_df.loc[mask, "General topic"] = merged_general
                                reference_df.loc[mask, "Subtopic"] = merged_subtopic
                                num_merges_applied += 1

                                # Track unique combinations for reporting
                                orig_combo = (normalized_orig_gen, normalized_orig_sub)
                                merged_combo = (
                                    normalized_merged_gen,
                                    normalized_merged_sub,
                                )
                                unique_original_combinations.add(orig_combo)
                                unique_merged_combinations.add(merged_combo)

                                merge_type = (
                                    "consolidated" if is_real_merge else "renamed"
                                )
                                print(
                                    f"Merged ({merge_type}): {original_general} | {original_subtopic} -> {merged_general} | {merged_subtopic}"
                                )
                            else:
                                # Debug: show why merge failed
                                # Use pre-computed normalized topics from lookup
                                # Get available topics from pre-computed map for debugging
                                available_gen_topics = {
                                    norm_gen
                                    for (
                                        norm_gen,
                                        norm_sub,
                                    ) in row_normalized_map.keys()
                                }
                                available_sub_topics = {
                                    norm_sub
                                    for (
                                        norm_gen,
                                        norm_sub,
                                    ) in row_normalized_map.keys()
                                }
                                # Check if the normalized general topic exists
                                gen_exists = normalized_orig_gen in available_gen_topics
                                sub_exists = normalized_orig_sub in available_sub_topics
                                print(
                                    f"Failed to merge: '{original_general}' | '{original_subtopic}' "
                                    f"(normalized: '{normalized_orig_gen}' | '{normalized_orig_sub}'). "
                                    f"General topic exists: {gen_exists}, Subtopic exists: {sub_exists}. "
                                    f"No matching rows found in reference_df."
                                )
            else:
                print("No merge suggestions found in LLM response")
        else:
            print("No markdown table found in LLM response")

        if num_merges_applied == 0:
            print("No duplicate topics found to merge")
        else:
            # Report on unique topic combinations affected
            unique_originals_count = len(unique_original_combinations)
            unique_merged_count = len(unique_merged_combinations)
            unique_reduction = unique_originals_count - unique_merged_count

            print(
                f"Merge summary: {num_merges_applied} merge operation(s) processed, "
                f"affecting {unique_originals_count} unique original topic combination(s), "
                f"resulting in {unique_merged_count} unique merged topic combination(s). "
                f"Net reduction: {unique_reduction} unique topic combination(s)."
            )

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print("Continuing with original data...")

    # Update reference summary column with all summaries
    if include_sentiment:
        reference_df["Summary"] = reference_df.groupby(
            ["Response References", "General topic", "Subtopic", "Sentiment"]
        )["Summary"].transform(" <br> ".join)
    else:
        reference_df["Summary"] = reference_df.groupby(
            ["Response References", "General topic", "Subtopic"]
        )["Summary"].transform(" <br> ".join)

    # Check that we have not inadvertently removed some data during the process
    end_unique_references = len(reference_df["Response References"].unique())

    if initial_unique_references != end_unique_references:
        raise Exception(
            f"Number of unique references changed during processing: Initial={initial_unique_references}, Final={end_unique_references}"
        )

    # Drop duplicates in the reference table
    if include_sentiment:
        reference_df.drop_duplicates(
            ["Response References", "General topic", "Subtopic", "Sentiment"],
            inplace=True,
        )
    else:
        reference_df.drop_duplicates(
            ["Response References", "General topic", "Subtopic"], inplace=True
        )

    # Before recreating topic_summary_df, check if input had Group information
    # If input topic_summary_df doesn't have Group or all have same Group, normalize reference_df Group
    input_has_group = "Group" in topic_summary_df.columns
    input_unique_groups = topic_summary_df["Group"].nunique() if input_has_group else 0

    # If input didn't have meaningful Group distinction, normalize to single Group
    if not input_has_group or input_unique_groups <= 1:
        # Get the most common Group value from reference_df, or use "All" if not present
        if "Group" in reference_df.columns and not reference_df["Group"].empty:
            most_common_group = (
                reference_df["Group"].mode()[0]
                if len(reference_df["Group"].mode()) > 0
                else "All"
            )
        else:
            most_common_group = "All"
        # Normalize all Groups to the most common one to prevent topic count increase
        reference_df["Group"] = most_common_group

    # Remake topic_summary_df based on new reference_df
    topic_summary_df = create_topic_summary_df_from_reference_table(
        reference_df, sentiment_checkbox=sentiment_checkbox
    )

    # Normalize topic names in both dataframes to ensure consistent formatting
    # Use the same normalization function as defined earlier in this function
    # Apply normalization to topic_summary_df
    if "General topic" in topic_summary_df.columns:
        topic_summary_df["General topic"] = topic_summary_df["General topic"].apply(
            normalize_topic_name_for_llm
        )
    if "Subtopic" in topic_summary_df.columns:
        topic_summary_df["Subtopic"] = topic_summary_df["Subtopic"].apply(
            normalize_topic_name_for_llm
        )

    # Apply normalization to reference_df
    if "General topic" in reference_df.columns:
        reference_df["General topic"] = reference_df["General topic"].apply(
            normalize_topic_name_for_llm
        )
    if "Subtopic" in reference_df.columns:
        reference_df["Subtopic"] = reference_df["Subtopic"].apply(
            normalize_topic_name_for_llm
        )

    if "Topic number" not in reference_df.columns:

        # Merge the topic numbers back to the original dataframe
        reference_df = reference_df.merge(
            topic_summary_df[
                ["General topic", "Subtopic", "Sentiment", "Group", "Topic number"]
            ],
            on=["General topic", "Subtopic", "Sentiment", "Group"],
            how="left",
        )

    # Create pivot table if file data is available
    if not file_data.empty:
        basic_response_data = get_basic_response_data(file_data, chosen_cols)
        reference_df_pivot = convert_reference_table_to_pivot_table(
            reference_df, basic_response_data
        )

        reference_pivot_file_path = (
            output_folder
            + get_file_name_no_ext(reference_table_file_name)
            + "_pivot_dedup.csv"
        )
        if should_output_files == "True":
            reference_df_pivot.to_csv(
                reference_pivot_file_path, index=None, encoding="utf-8-sig"
            )
        log_output_files.append(reference_pivot_file_path)

    # Save analysis results CSV if merge suggestions were found
    if not merge_suggestions_df.empty:
        analysis_results_file_path = (
            output_folder
            + get_file_name_no_ext(reference_table_file_name)
            + "_dedup_llm_analysis_results.csv"
        )
        if should_output_files == "True":
            merge_suggestions_df.to_csv(
                analysis_results_file_path, index=None, encoding="utf-8-sig"
            )
            log_output_files.append(analysis_results_file_path)
            print(f"Analysis results saved to: {analysis_results_file_path}")

    # Save output files
    reference_file_out_path = (
        output_folder + get_file_name_no_ext(reference_table_file_name) + "_dedup.csv"
    )
    unique_topics_file_out_path = (
        output_folder
        + get_file_name_no_ext(unique_topics_table_file_name)
        + "_dedup.csv"
    )
    if should_output_files == "True":
        reference_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
            reference_file_out_path, index=None, encoding="utf-8-sig"
        )
        topic_summary_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
            unique_topics_file_out_path, index=None, encoding="utf-8-sig"
        )

        output_files.append(reference_file_out_path)
        output_files.append(unique_topics_file_out_path)

    # Outputs for markdown table output
    topic_summary_df_revised_display = topic_summary_df.apply(
        lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
    )
    deduplicated_unique_table_markdown = topic_summary_df_revised_display.to_markdown(
        index=False
    )

    # Calculate token usage and timing information for logging
    total_input_tokens = 0
    total_output_tokens = 0
    number_of_calls = 1  # Single LLM call for deduplication

    # Extract token usage from conversation metadata
    if whole_conversation_metadata:
        for metadata in whole_conversation_metadata:
            if "input_tokens:" in metadata and "output_tokens:" in metadata:
                try:
                    input_tokens = int(
                        metadata.split("input_tokens: ")[1].split(" ")[0]
                    )
                    output_tokens = int(
                        metadata.split("output_tokens: ")[1].split(" ")[0]
                    )
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                except (ValueError, IndexError):
                    pass

    # Calculate estimated time taken (rough estimate based on token usage)
    estimated_time_taken = (
        total_input_tokens + total_output_tokens
    ) / 1000  # Rough estimate in seconds

    print("LLM deduplication task successfully completed")

    # Compare input vs output unique topic combinations
    if not topic_summary_df.empty:
        output_unique_topics = topic_summary_df.drop_duplicates(
            subset=["General topic", "Subtopic"]
        ).shape[0]
        print(
            f"Topic count comparison: Input had {input_unique_topics} unique 'General topic' | 'Subtopic' combinations, "
            f"Output has {output_unique_topics} unique 'General topic' | 'Subtopic' combinations. "
            f"Reduction: {input_unique_topics - output_unique_topics} topics merged."
        )
    else:
        print(
            f"Topic count comparison: Input had {input_unique_topics} unique 'General topic' | 'Subtopic' combinations, "
            f"Output has 0 unique 'General topic' | 'Subtopic' combinations."
        )

    return (
        reference_df,
        topic_summary_df,
        output_files,
        log_output_files,
        deduplicated_unique_table_markdown,
        total_input_tokens,
        total_output_tokens,
        number_of_calls,
        estimated_time_taken,
    )  # , num_merges_applied


def sample_reference_table_summaries(
    reference_df: pd.DataFrame,
    random_seed: int,
    no_of_sampled_summaries: int = default_number_of_sampled_summaries,
    sample_reference_table_checkbox: bool = False,
):
    """
    Sample x number of summaries from which to produce summaries, so that the input token length is not too long.
    """

    if sample_reference_table_checkbox:

        all_summaries = pd.DataFrame(
            columns=[
                "General topic",
                "Subtopic",
                "Sentiment",
                "Group",
                "Response References",
                "Summary",
            ]
        )

        if "Group" not in reference_df.columns:
            reference_df["Group"] = "All"

        reference_df_grouped = reference_df.groupby(
            ["General topic", "Subtopic", "Sentiment", "Group"]
        )

        if "Revised summary" in reference_df.columns:
            out_message = "Summary has already been created for this file"
            print(out_message)
            raise Exception(out_message)

        for group_keys, reference_df_group in reference_df_grouped:
            if len(reference_df_group["General topic"]) > 1:

                filtered_reference_df = reference_df_group.reset_index()

                filtered_reference_df_unique = filtered_reference_df.drop_duplicates(
                    ["General topic", "Subtopic", "Sentiment", "Summary"]
                )

                # Sample n of the unique topic summaries PER GROUP. To limit the length of the text going into the summarisation tool
                # This ensures each group gets up to no_of_sampled_summaries summaries, not the total across all groups
                filtered_reference_df_unique_sampled = (
                    filtered_reference_df_unique.sample(
                        min(no_of_sampled_summaries, len(filtered_reference_df_unique)),
                        random_state=random_seed,
                    )
                )

                all_summaries = pd.concat(
                    [all_summaries, filtered_reference_df_unique_sampled]
                )

        # If no responses/topics qualify, just go ahead with the original reference dataframe
        if all_summaries.empty:
            sampled_reference_table_df = reference_df
            # Filter by sentiment only (Response References is a string in original df, not a count)
            sampled_reference_table_df = sampled_reference_table_df.loc[
                sampled_reference_table_df["Sentiment"] != "Not Mentioned"
            ]
        else:
            # FIXED: Preserve Group column in aggregation to maintain group-specific summaries
            sampled_reference_table_df = (
                all_summaries.groupby(
                    ["General topic", "Subtopic", "Sentiment", "Group"]
                )
                .agg(
                    {
                        "Response References": "size",  # Count the number of references
                        "Summary": lambda x: "\n".join(
                            [s.split(": ", 1)[1] for s in x if ": " in s]
                        ),  # Join substrings after ': '
                    }
                )
                .reset_index()
            )
            # Filter by sentiment and count (Response References is now a numeric count after aggregation)
            sampled_reference_table_df = sampled_reference_table_df.loc[
                (sampled_reference_table_df["Sentiment"] != "Not Mentioned")
                & (sampled_reference_table_df["Response References"] > 1)
            ]
    else:
        sampled_reference_table_df = reference_df

    summarised_references_markdown = sampled_reference_table_df.to_markdown(index=False)

    return sampled_reference_table_df, summarised_references_markdown


def count_tokens_in_text(text: str, tokenizer=None, model_source: str = "Local") -> int:
    """
    Count the number of tokens in the given text.

    Args:
        text (str): The text to count tokens for
        tokenizer (object, optional): Tokenizer object for local models. Defaults to None.
        model_source (str): Source of the model to determine tokenization method. Defaults to "Local".

    Returns:
        int: Number of tokens in the text
    """
    if not text:
        return 0

    try:
        if model_source == "Local" and tokenizer and len(tokenizer) > 0:
            # Use local tokenizer if available
            tokens = tokenizer[0].encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            # Fallback: rough estimation using word count (approximately 1.3 tokens per word)
            word_count = len(text.split())
            return int(word_count * 1.3)
    except Exception as e:
        print(f"Error counting tokens: {e}. Using word count estimation.")
        # Fallback: rough estimation using word count
        word_count = len(text.split())
        return int(word_count * 1.3)


def summarise_output_topics_query(
    model_choice: str,
    in_api_key: str,
    temperature: float,
    formatted_summary_prompt: str,
    summarise_topic_descriptions_system_prompt: str,
    model_source: str,
    bedrock_runtime: boto3.Session.client,
    local_model=list(),
    tokenizer=list(),
    assistant_model=list(),
    azure_endpoint: str = "",
    api_url: str = None,
):
    """
    Query an LLM to generate a summary of topics based on the provided prompts.

    Args:
        model_choice (str): The name/type of model to use for generation
        in_api_key (str): API key for accessing the model service
        temperature (float): Temperature parameter for controlling randomness in generation
        formatted_summary_prompt (str): The formatted prompt containing topics to summarize
        summarise_topic_descriptions_system_prompt (str): System prompt providing context and instructions
        model_source (str): Source of the model (e.g. "AWS", "Gemini", "Local")
        bedrock_runtime (boto3.Session.client): AWS Bedrock runtime client for AWS models
        local_model (object, optional): Local model object if using local inference. Defaults to empty list.
        tokenizer (object, optional): Tokenizer object if using local inference. Defaults to empty list.
    Returns:
        tuple: Contains:
            - response_text (str): The generated summary text
            - conversation_history (list): History of the conversation with the model
            - whole_conversation_metadata (list): Metadata about the conversation
    """
    conversation_history = list()
    whole_conversation_metadata = list()
    client = list()
    client_config = {}

    # Combine system prompt and user prompt for token counting
    full_input_text = (
        summarise_topic_descriptions_system_prompt + "\n" + formatted_summary_prompt[0]
        if isinstance(formatted_summary_prompt, list)
        else summarise_topic_descriptions_system_prompt
        + "\n"
        + formatted_summary_prompt
    )

    # Count tokens in the input text
    input_token_count = count_tokens_in_text(full_input_text, tokenizer, model_source)

    # Check if input exceeds context length
    if input_token_count > LLM_CONTEXT_LENGTH:
        error_message = f"Input text exceeds LLM context length. Input tokens: {input_token_count}, Max context length: {LLM_CONTEXT_LENGTH}. Please reduce the input text size."
        print(error_message)
        raise ValueError(error_message)

    print(f"Input token count: {input_token_count} (Max: {LLM_CONTEXT_LENGTH})")

    # Prepare Gemini models before query
    if "Gemini" in model_source:
        # print("Using Gemini model:", model_choice)
        client, config = construct_gemini_generative_model(
            in_api_key=in_api_key,
            temperature=temperature,
            model_choice=model_choice,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    elif "Azure/OpenAI" in model_source:
        client, config = construct_azure_client(
            in_api_key=os.environ.get("AZURE_INFERENCE_CREDENTIAL", ""),
            endpoint=azure_endpoint,
        )
    elif "Local" in model_source:
        pass
        # print("Using local model: ", model_choice)
    elif "AWS" in model_source:
        pass
        # print("Using AWS Bedrock model:", model_choice)

    whole_conversation = [summarise_topic_descriptions_system_prompt]

    # Process requests to large language model
    (
        responses,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        response_text,
    ) = process_requests(
        formatted_summary_prompt,
        system_prompt,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        client,
        client_config,
        model_choice,
        temperature,
        bedrock_runtime=bedrock_runtime,
        model_source=model_source,
        local_model=local_model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        assistant_prefill=summary_assistant_prefill,
        api_url=api_url,
    )

    summarised_output = re.sub(
        r"\n{2,}", "\n", response_text
    )  # Replace multiple line breaks with a single line break
    summarised_output = re.sub(
        r"^\n{1,}", "", summarised_output
    )  # Remove one or more line breaks at the start
    summarised_output = re.sub(
        r"\n", "<br>", summarised_output
    )  # Replace \n with more html friendly <br> tags
    summarised_output = summarised_output.strip()

    print("Finished summary query")

    # Ensure the system prompt is included in the conversation history
    try:
        if isinstance(conversation_history, list):
            has_system_prompt = False

            if conversation_history:
                first_entry = conversation_history[0]
                if isinstance(first_entry, dict):
                    role_is_system = first_entry.get("role") == "system"
                    parts = first_entry.get("parts")
                    content_matches = (
                        parts == summarise_topic_descriptions_system_prompt
                        or (
                            isinstance(parts, list)
                            and summarise_topic_descriptions_system_prompt in parts
                        )
                    )
                    has_system_prompt = role_is_system and content_matches
                elif isinstance(first_entry, str):
                    has_system_prompt = (
                        first_entry.strip().lower().startswith("system:")
                    )

            if not has_system_prompt:
                conversation_history.insert(
                    0,
                    {
                        "role": "system",
                        "parts": [summarise_topic_descriptions_system_prompt],
                    },
                )
    except Exception as _e:
        # Non-fatal: if anything goes wrong, return the original conversation history
        pass

    return (
        summarised_output,
        conversation_history,
        whole_conversation_metadata,
        response_text,
    )


def process_debug_output_iteration(
    output_debug_files: str,
    output_folder: str,
    batch_file_path_details: str,
    model_choice_clean_short: str,
    final_system_prompt: str,
    summarised_output: str,
    conversation_history: list,
    metadata: list,
    log_output_files: list,
    task_type: str,
) -> tuple[str, str, str, str]:
    """
    Writes debug files for summary generation if output_debug_files is "True",
    and returns the content of the prompt, summary, conversation, and metadata for the current iteration.

    Args:
        output_debug_files (str): Flag to indicate if debug files should be written.
        output_folder (str): The folder where output files are saved.
        batch_file_path_details (str): Details for the batch file path.
        model_choice_clean_short (str): Shortened cleaned model choice.
        final_system_prompt (str): The system prompt content.
        summarised_output (str): The summarised output content.
        conversation_history (list): The full conversation history.
        metadata (list): The metadata for the conversation.
        log_output_files (list): A list to append paths of written log files. This list is modified in-place.
        task_type (str): The type of task being performed.
    Returns:
        tuple[str, str, str, str]: A tuple containing the content of the prompt,
                                    summarised output, conversation history (as string),
                                    and metadata (as string) for the current iteration.
    """
    current_prompt_content = final_system_prompt
    current_summary_content = summarised_output

    if isinstance(conversation_history, list):

        # Handle both list of strings and list of dicts
        if conversation_history and isinstance(conversation_history[0], dict):
            # Convert list of dicts to list of strings
            conversation_strings = list()
            for entry in conversation_history:
                if "role" in entry and "parts" in entry:
                    role = entry["role"].capitalize()
                    message = (
                        " ".join(entry["parts"])
                        if isinstance(entry["parts"], list)
                        else str(entry["parts"])
                    )
                    conversation_strings.append(f"{role}: {message}")
                else:
                    # Fallback for unexpected dict format
                    conversation_strings.append(str(entry))
            current_conversation_content = "\n".join(conversation_strings)
        else:
            # Handle list of strings
            current_conversation_content = "\n".join(conversation_history)
    else:
        current_conversation_content = str(conversation_history)
    current_metadata_content = str(metadata)
    current_task_type = task_type

    if output_debug_files == "True":
        try:
            formatted_prompt_output_path = (
                output_folder
                + batch_file_path_details
                + "_full_prompt_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            final_table_output_path = (
                output_folder
                + batch_file_path_details
                + "_full_response_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            whole_conversation_path = (
                output_folder
                + batch_file_path_details
                + "_full_conversation_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            whole_conversation_path_meta = (
                output_folder
                + batch_file_path_details
                + "_metadata_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )

            with open(
                formatted_prompt_output_path,
                "w",
                encoding="utf-8-sig",
                errors="replace",
            ) as f:
                f.write(current_prompt_content)
            with open(
                final_table_output_path, "w", encoding="utf-8-sig", errors="replace"
            ) as f:
                f.write(current_summary_content)
            with open(
                whole_conversation_path, "w", encoding="utf-8-sig", errors="replace"
            ) as f:
                f.write(current_conversation_content)
            with open(
                whole_conversation_path_meta,
                "w",
                encoding="utf-8-sig",
                errors="replace",
            ) as f:
                f.write(current_metadata_content)

            log_output_files.append(formatted_prompt_output_path)
            log_output_files.append(final_table_output_path)
            log_output_files.append(whole_conversation_path)
            log_output_files.append(whole_conversation_path_meta)
        except Exception as e:
            print(f"Error in writing debug files for summary: {e}")

    # Return the content of the objects for the current iteration.
    # The caller can then append these to separate lists if accumulation is desired.
    return (
        current_prompt_content,
        current_summary_content,
        current_conversation_content,
        current_metadata_content,
    )


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def summarise_output_topics(
    sampled_reference_table_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    reference_table_df: pd.DataFrame,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    reference_data_file_name: str,
    summarised_outputs: list = list(),
    latest_summary_completed: int = 0,
    out_metadata_str: str = "",
    in_data_files: List[str] = list(),
    in_excel_sheets: str = "",
    chosen_cols: List[str] = list(),
    log_output_files: list[str] = list(),
    summarise_format_radio: str = "Return a summary up to two paragraphs long that includes as much detail as possible from the original text",
    output_folder: str = OUTPUT_FOLDER,
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    model_name_map: dict = model_name_map,
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    existing_logged_content: list = list(),
    additional_summary_instructions_provided: str = "",
    output_debug_files: str = "False",
    group_value: str = "All",
    reasoning_suffix: str = reasoning_suffix,
    local_model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    summarise_topic_descriptions_prompt: str = summarise_topic_descriptions_prompt,
    summarise_topic_descriptions_system_prompt: str = summarise_topic_descriptions_system_prompt,
    do_summaries: str = "Yes",
    api_url: str = None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Create improved summaries of topics by consolidating raw batch-level summaries from the initial model run. Works on a single group of summaries at a time (called from wrapper function summarise_output_topics_by_group).

    Args:
        sampled_reference_table_df (pd.DataFrame): DataFrame containing sampled reference data with summaries
        topic_summary_df (pd.DataFrame): DataFrame containing topic summary information
        reference_table_df (pd.DataFrame): DataFrame mapping response references to topics
        model_choice (str): Name of the LLM model to use
        in_api_key (str): API key for model access
        temperature (float): Temperature parameter for model generation
        reference_data_file_name (str): Name of the reference data file
        summarised_outputs (list, optional): List to store generated summaries. Defaults to empty list.
        latest_summary_completed (int, optional): Index of last completed summary. Defaults to 0.
        out_metadata_str (str, optional): String for metadata output. Defaults to empty string.
        in_data_files (List[str], optional): List of input data file paths. Defaults to empty list.
        in_excel_sheets (str, optional): Excel sheet names if using Excel files. Defaults to empty string.
        chosen_cols (List[str], optional): List of columns selected for analysis. Defaults to empty list.
        log_output_files (list[str], optional): List of log file paths. Defaults to empty list.
        summarise_format_radio (str, optional): Format instructions for summary generation. Defaults to two paragraph format.
        output_folder (str, optional): Folder path for outputs. Defaults to OUTPUT_FOLDER.
        context_textbox (str, optional): Additional context for summarization. Defaults to empty string.
        aws_access_key_textbox (str, optional): AWS access key. Defaults to empty string.
        aws_secret_key_textbox (str, optional): AWS secret key. Defaults to empty string.
        model_name_map (dict, optional): Dictionary mapping model choices to their properties. Defaults to model_name_map.
        hf_api_key_textbox (str, optional): Hugging Face API key. Defaults to empty string.
        azure_endpoint_textbox (str, optional): Azure endpoint. Defaults to empty string.
        additional_summary_instructions_provided (str, optional): Additional summary instructions provided by the user. Defaults to empty string.
        existing_logged_content (list, optional): List of existing logged content. Defaults to empty list.
        output_debug_files (str, optional): Flag to indicate if debug files should be written. Defaults to "False".
        group_value (str, optional): Value of the group to summarise. Defaults to "All".
        reasoning_suffix (str, optional): Suffix for reasoning. Defaults to reasoning_suffix.
        local_model (object, optional): Local model object if using local inference. Defaults to None.
        tokenizer (object, optional): Tokenizer object if using local inference. Defaults to None.
        assistant_model (object, optional): Assistant model object if using local inference. Defaults to None.
        summarise_topic_descriptions_prompt (str, optional): Prompt template for topic summarization.
        summarise_topic_descriptions_system_prompt (str, optional): System prompt for topic summarization.
        do_summaries (str, optional): Flag to control summary generation. Defaults to "Yes".
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to track_tqdm=True.

    Returns:
        Multiple outputs including summarized content, metadata, and file paths
    """
    out_metadata = list()
    summarised_output_markdown = ""
    output_files = list()
    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_number_of_calls = 0
    time_taken = 0
    out_metadata_str = (
        ""  # Output metadata is currently replaced on starting a summarisation task
    )
    out_message = list()
    task_type = "Topic summarisation"
    topic_summary_df_revised = pd.DataFrame()

    all_prompts_content = list()
    all_summaries_content = list()
    all_metadata_content = list()
    all_groups_content = list()
    all_batches_content = list()
    all_model_choice_content = list()
    all_validated_content = list()
    all_task_type_content = list()
    all_logged_content = list()
    all_file_names_content = list()

    tic = time.perf_counter()

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    model_choice_clean = clean_column_name(
        model_name_map[model_choice]["short_name"],
        max_length=20,
        front_characters=False,
    )

    if context_textbox and "The context of this analysis is" not in context_textbox:
        context_textbox = "The context of this analysis is '" + context_textbox + "'."

    if log_output_files is None:
        log_output_files = list()

    # Check for data for summarisations
    if not topic_summary_df.empty and not reference_table_df.empty:
        print("Unique table and reference table data found.")
    else:
        out_message = "Please upload a unique topic table and reference table file to continue with summarisation."
        print(out_message)
        raise Exception(out_message)

    if "Revised summary" in reference_table_df.columns:
        out_message = "Summary has already been created for this file"
        print(out_message)
        raise Exception(out_message)

    # Load in data file and chosen columns if exists to create pivot table later
    file_data = pd.DataFrame()
    if in_data_files and chosen_cols:
        file_data, data_file_names_textbox, total_number_of_batches = load_in_data_file(
            in_data_files, chosen_cols, 1, in_excel_sheets=in_excel_sheets
        )
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)
        # Use sys.stdout.write to avoid issues with progress bars
        # sys.stdout.write(out_message + "\n")
        # sys.stdout.flush()
        # Note: file_data will remain empty, pivot tables will not be created

    reference_table_df = reference_table_df.rename(
        columns={"General Topic": "General topic"}, errors="ignore"
    )
    topic_summary_df = topic_summary_df.rename(
        columns={"General Topic": "General topic"}, errors="ignore"
    )
    if "Group" not in reference_table_df.columns:
        reference_table_df["Group"] = "All"
    if "Group" not in topic_summary_df.columns:
        topic_summary_df["Group"] = "All"
    if "Group" not in sampled_reference_table_df.columns:
        sampled_reference_table_df["Group"] = "All"

    # Use the Summary column if it exists, otherwise use the Revised summary column
    if "Summary" in sampled_reference_table_df.columns:
        all_summaries = sampled_reference_table_df["Summary"].tolist()
    else:
        all_summaries = sampled_reference_table_df["Revised summary"].tolist()

    all_groups = sampled_reference_table_df["Group"].tolist()

    if not group_value:
        group_value = str(all_groups[0])
    else:
        group_value = str(group_value)

    length_all_summaries = len(all_summaries)

    model_source = model_name_map[model_choice]["source"]

    if (model_source == "Local") & (RUN_LOCAL_MODEL == "1") & (not local_model):
        progress(0.1, f"Using global model: {CHOSEN_LOCAL_MODEL_TYPE}")
        local_model = get_model()
        tokenizer = get_tokenizer()
        assistant_model = get_assistant_model()

    (
        "Revising topic-level summaries. "
        + str(latest_summary_completed)
        + " summaries completed so far."
    )
    summary_loop = progress.tqdm(
        range(latest_summary_completed, length_all_summaries),
        desc="Revising topic-level summaries",
        unit="summaries",
    )

    if do_summaries == "Yes":

        bedrock_runtime = connect_to_bedrock_runtime(
            model_name_map,
            model_choice,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            aws_region_textbox,
        )

        create_batch_file_path_details(reference_data_file_name)
        model_choice_clean_short = clean_column_name(
            model_choice_clean, max_length=20, front_characters=False
        )
        file_name_clean = f"{clean_column_name(reference_data_file_name, max_length=15)}_{clean_column_name(str(group_value), max_length=15).replace(' ','_')}"
        # file_name_clean = clean_column_name(reference_data_file_name, max_length=20, front_characters=True)
        in_column_cleaned = clean_column_name(chosen_cols, max_length=20)

        combined_summary_instructions = (
            summarise_format_radio + ". " + additional_summary_instructions_provided
        )

        for summary_no in summary_loop:
            print("Current summary number is:", summary_no)

            batch_file_path_details = f"{file_name_clean}_batch_{latest_summary_completed + 1}_size_1_col_{in_column_cleaned}"

            summary_text = all_summaries[summary_no]
            formatted_summary_prompt = [
                summarise_topic_descriptions_prompt.format(
                    summaries=summary_text, summary_format=combined_summary_instructions
                )
            ]

            formatted_summarise_topic_descriptions_system_prompt = (
                summarise_topic_descriptions_system_prompt.format(
                    column_name=chosen_cols, consultation_context=context_textbox
                )
            )

            # Apply reasoning suffix for GPT-OSS models (Local, inference-server, or AWS)
            is_gpt_oss_model = (
                "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower()
            )

            if is_gpt_oss_model:
                # Use default reasoning suffix if not set
                effective_reasoning_suffix = (
                    reasoning_suffix if reasoning_suffix else "Reasoning: low"
                )
                if effective_reasoning_suffix:
                    formatted_summarise_topic_descriptions_system_prompt = (
                        formatted_summarise_topic_descriptions_system_prompt
                        + "\n"
                        + effective_reasoning_suffix
                    )
            elif "Local" in model_source and reasoning_suffix:
                # For other local models, use reasoning_suffix if provided
                formatted_summarise_topic_descriptions_system_prompt = (
                    formatted_summarise_topic_descriptions_system_prompt
                    + "\n"
                    + reasoning_suffix
                )

            try:
                response, conversation_history, metadata, response_text = (
                    summarise_output_topics_query(
                        model_choice,
                        in_api_key,
                        temperature,
                        formatted_summary_prompt,
                        formatted_summarise_topic_descriptions_system_prompt,
                        model_source,
                        bedrock_runtime,
                        local_model,
                        tokenizer=tokenizer,
                        assistant_model=assistant_model,
                        azure_endpoint=azure_endpoint_textbox,
                        api_url=api_url,
                    )
                )
                summarised_output = response_text
            except Exception as e:
                print("Creating summary failed:", e)
                summarised_output = ""

            summarised_outputs.append(summarised_output)
            out_metadata.extend(metadata)
            out_metadata_str = ". ".join(out_metadata)

            # Call the new function to process and log debug outputs for the current iteration.
            # The returned values are the contents of the prompt, summary, conversation, and metadata

            full_prompt = (
                formatted_summarise_topic_descriptions_system_prompt
                + "\n"
                + formatted_summary_prompt[0]
            )

            # Coerce toggle to string expected by debug writer (accepts True/False or "True"/"False")
            output_debug_files_str = (
                "True"
                if (
                    (isinstance(output_debug_files, bool) and output_debug_files)
                    or (str(output_debug_files) == "True")
                )
                else "False"
            )

            (
                current_prompt_content_logged,
                current_summary_content_logged,
                current_conversation_content_logged,
                current_metadata_content_logged,
            ) = process_debug_output_iteration(
                output_debug_files_str,
                output_folder,
                batch_file_path_details,
                model_choice_clean_short,
                full_prompt,
                summarised_output,
                conversation_history,
                metadata,
                log_output_files,
                task_type=task_type,
            )

            all_prompts_content.append(current_prompt_content_logged)
            all_summaries_content.append(current_summary_content_logged)
            # all_conversation_content.append(current_conversation_content_logged)
            all_metadata_content.append(current_metadata_content_logged)
            all_groups_content.append(all_groups[summary_no])
            all_batches_content.append(f"{summary_no}:")
            all_model_choice_content.append(model_choice_clean_short)
            all_validated_content.append("No")
            all_task_type_content.append(task_type)
            all_file_names_content.append(reference_data_file_name)
            latest_summary_completed += 1

            toc = time.perf_counter()
            time_taken = toc - tic

            if time_taken > max_time_for_loop:
                print(
                    "Time taken for loop is greater than maximum time allowed. Exiting and restarting loop"
                )
                summary_loop.close()
                tqdm._instances.clear()
                break

    # If all summaries completed, make final outputs
    if latest_summary_completed >= length_all_summaries:
        print("All summaries completed. Creating outputs.")

        sampled_reference_table_df["Revised summary"] = summarised_outputs

        join_cols = ["General topic", "Subtopic", "Sentiment"]
        join_plus_summary_cols = [
            "General topic",
            "Subtopic",
            "Sentiment",
            "Revised summary",
        ]

        summarised_references_j = sampled_reference_table_df[
            join_plus_summary_cols
        ].drop_duplicates(join_plus_summary_cols)

        topic_summary_df_revised = topic_summary_df.merge(
            summarised_references_j, on=join_cols, how="left"
        )

        # If no new summary is available, keep the original
        # But prefer the version without "Rows X to Y" prefix to avoid duplication
        def clean_summary_text(text):
            if pd.isna(text):
                return text
            # Remove "Rows X to Y:" prefix if present (both at start and after <br> tags)
            import re

            # First remove from the beginning
            cleaned = re.sub(r"^Rows\s+\d+\s+to\s+\d+:\s*", "", str(text))
            # Then remove from after <br> tags
            cleaned = re.sub(r"<br>\s*Rows\s+\d+\s+to\s+\d+:\s*", "<br>", cleaned)
            return cleaned

        topic_summary_df_revised["Revised summary"] = topic_summary_df_revised[
            "Revised summary"
        ].combine_first(topic_summary_df_revised["Summary"])
        # Clean the revised summary to remove "Rows X to Y" prefixes
        topic_summary_df_revised["Revised summary"] = topic_summary_df_revised[
            "Revised summary"
        ].apply(clean_summary_text)
        topic_summary_df_revised = topic_summary_df_revised[
            [
                "General topic",
                "Subtopic",
                "Sentiment",
                "Group",
                "Number of responses",
                "Revised summary",
            ]
        ]

        # Note: "Rows X to Y:" prefixes are now cleaned by the clean_summary_text function above
        topic_summary_df_revised["Topic number"] = range(
            1, len(topic_summary_df_revised) + 1
        )

        # If no new summary is available, keep the original. Also join on topic number to ensure consistent topic number assignment
        reference_table_df_revised = reference_table_df.copy()
        reference_table_df_revised = reference_table_df_revised.drop(
            "Topic number", axis=1, errors="ignore"
        )

        # Ensure reference table has Topic number column
        if (
            "Topic number" not in reference_table_df_revised.columns
            or "Revised summary" not in reference_table_df_revised.columns
        ):
            if (
                "Topic number" in topic_summary_df_revised.columns
                and "Revised summary" in topic_summary_df_revised.columns
            ):
                reference_table_df_revised = reference_table_df_revised.merge(
                    topic_summary_df_revised[
                        [
                            "General topic",
                            "Subtopic",
                            "Sentiment",
                            "Group",
                            "Topic number",
                            "Revised summary",
                        ]
                    ],
                    on=["General topic", "Subtopic", "Sentiment", "Group"],
                    how="left",
                )

        reference_table_df_revised["Revised summary"] = reference_table_df_revised[
            "Revised summary"
        ].combine_first(reference_table_df_revised["Summary"])
        # Clean the revised summary to remove "Rows X to Y" prefixes
        reference_table_df_revised["Revised summary"] = reference_table_df_revised[
            "Revised summary"
        ].apply(clean_summary_text)
        reference_table_df_revised = reference_table_df_revised.drop(
            "Summary", axis=1, errors="ignore"
        )

        # Remove topics that are tagged as 'Not Mentioned'
        topic_summary_df_revised = topic_summary_df_revised.loc[
            topic_summary_df_revised["Sentiment"] != "Not Mentioned", :
        ]
        reference_table_df_revised = reference_table_df_revised.loc[
            reference_table_df_revised["Sentiment"] != "Not Mentioned", :
        ]

        # Combine the logged content into a list of dictionaries
        all_logged_content = [
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
                all_summaries_content,
                all_metadata_content,
                all_batches_content,
                all_model_choice_content,
                all_validated_content,
                all_groups_content,
                all_task_type_content,
                all_file_names_content,
            )
        ]

        if isinstance(existing_logged_content, pd.DataFrame):
            existing_logged_content = existing_logged_content.to_dict(orient="records")

        out_logged_content = existing_logged_content + all_logged_content

        ### Save output files

        if output_debug_files == "True":

            if not file_data.empty:
                basic_response_data = get_basic_response_data(file_data, chosen_cols)
                reference_table_df_revised_pivot = (
                    convert_reference_table_to_pivot_table(
                        reference_table_df_revised, basic_response_data
                    )
                )

                ### Save pivot file to log area
                reference_table_df_revised_pivot_path = (
                    output_folder
                    + file_name_clean
                    + "_summ_reference_table_pivot_"
                    + model_choice_clean
                    + ".csv"
                )
                reference_table_df_revised_pivot.drop(
                    ["1", "2", "3"], axis=1, errors="ignore"
                ).to_csv(
                    reference_table_df_revised_pivot_path,
                    index=None,
                    encoding="utf-8-sig",
                )
                log_output_files.append(reference_table_df_revised_pivot_path)

            # Save to file
            topic_summary_df_revised_path = (
                output_folder
                + file_name_clean
                + "_summ_unique_topics_table_"
                + model_choice_clean
                + ".csv"
            )
            topic_summary_df_revised.drop(
                ["1", "2", "3"], axis=1, errors="ignore"
            ).to_csv(topic_summary_df_revised_path, index=None, encoding="utf-8-sig")

            reference_table_df_revised_path = (
                output_folder
                + file_name_clean
                + "_summ_reference_table_"
                + model_choice_clean
                + ".csv"
            )
            reference_table_df_revised.drop(
                ["1", "2", "3"], axis=1, errors="ignore"
            ).to_csv(reference_table_df_revised_path, index=None, encoding="utf-8-sig")

            log_output_files.extend(
                [reference_table_df_revised_path, topic_summary_df_revised_path]
            )

        ###
        topic_summary_df_revised_display = topic_summary_df_revised.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )
        summarised_output_markdown = topic_summary_df_revised_display.to_markdown(
            index=False
        )

        # Ensure same file name not returned twice
        output_files = list(set(output_files))
        log_output_files = list(set(log_output_files))

        acc_input_tokens, acc_output_tokens, acc_number_of_calls = (
            calculate_tokens_from_metadata(
                out_metadata_str, model_choice, model_name_map
            )
        )

        toc = time.perf_counter()
        time_taken = toc - tic

        if isinstance(out_message, list):
            out_message = "\n".join(out_message)
        else:
            out_message = out_message

        out_message = (
            out_message
            + f"\nTopic summarisation finished processing. Total time: {round(float(time_taken), 1)}s"
        )
        print(out_message)

        return (
            sampled_reference_table_df,
            topic_summary_df_revised,
            reference_table_df_revised,
            output_files,
            summarised_outputs,
            latest_summary_completed,
            out_metadata_str,
            summarised_output_markdown,
            log_output_files,
            output_files,
            acc_input_tokens,
            acc_output_tokens,
            acc_number_of_calls,
            time_taken,
            out_message,
            out_logged_content,
        )


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def wrapper_summarise_output_topics_per_group(
    grouping_col: str,
    sampled_reference_table_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    reference_table_df: pd.DataFrame,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    reference_data_file_name: str,
    summarised_outputs: list = list(),
    latest_summary_completed: int = 0,
    out_metadata_str: str = "",
    in_data_files: List[str] = list(),
    in_excel_sheets: str = "",
    chosen_cols: List[str] = list(),
    log_output_files: list[str] = list(),
    summarise_format_radio: str = "Return a summary up to two paragraphs long that includes as much detail as possible from the original text",
    output_folder: str = OUTPUT_FOLDER,
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    model_name_map: dict = model_name_map,
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    existing_logged_content: list = list(),
    sample_reference_table: bool = False,
    no_of_sampled_summaries: int = default_number_of_sampled_summaries,
    random_seed: int = 42,
    api_url: str = None,
    additional_summary_instructions_provided: str = "",
    output_debug_files: str = OUTPUT_DEBUG_FILES,
    reasoning_suffix: str = reasoning_suffix,
    local_model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    summarise_topic_descriptions_prompt: str = summarise_topic_descriptions_prompt,
    summarise_topic_descriptions_system_prompt: str = summarise_topic_descriptions_system_prompt,
    do_summaries: str = "Yes",
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    List[str],
    int,
    str,
    str,
    List[str],
    List[str],
    int,
    int,
    int,
    float,
    str,
    List[dict],
]:
    """
    A wrapper function that iterates through unique values in a specified grouping column
    and calls the `summarise_output_topics` function for each group of summaries.
    It accumulates results from each call and returns a consolidated output.

    :param grouping_col: The name of the column to group the data by.
    :param sampled_reference_table_df: DataFrame containing sampled reference data with summaries
    :param topic_summary_df: DataFrame containing topic summary information
    :param reference_table_df: DataFrame mapping response references to topics
    :param model_choice: Name of the LLM model to use
    :param in_api_key: API key for model access
    :param temperature: Temperature parameter for model generation
    :param reference_data_file_name: Name of the reference data file
    :param summarised_outputs: List to store generated summaries
    :param latest_summary_completed: Index of last completed summary
    :param out_metadata_str: String for metadata output
    :param in_data_files: List of input data file paths
    :param in_excel_sheets: Excel sheet names if using Excel files
    :param chosen_cols: List of columns selected for analysis
    :param log_output_files: List of log file paths
    :param summarise_format_radio: Format instructions for summary generation
    :param output_folder: Folder path for outputs
    :param context_textbox: Additional context for summarization
    :param aws_access_key_textbox: AWS access key
    :param aws_secret_key_textbox: AWS secret key
    :param model_name_map: Dictionary mapping model choices to their properties
    :param hf_api_key_textbox: Hugging Face API key
    :param azure_endpoint_textbox: Azure endpoint
    :param existing_logged_content: List of existing logged content
    :param additional_summary_instructions_provided: Additional summary instructions
    :param output_debug_files: Flag to indicate if debug files should be written
    :param reasoning_suffix: Suffix for reasoning
    :param local_model: Local model object if using local inference
    :param tokenizer: Tokenizer object if using local inference
    :param assistant_model: Assistant model object if using local inference
    :param summarise_topic_descriptions_prompt: Prompt template for topic summarization
    :param summarise_topic_descriptions_system_prompt: System prompt for topic summarization
    :param do_summaries: Flag to control summary generation
    :param sample_reference_table: If True, sample the reference table at the top of the function
    :param no_of_sampled_summaries: Number of summaries to sample per group (default 100)
    :param random_seed: Random seed for reproducible sampling (default 42)
    :param progress: Gradio progress tracker
    :return: A tuple containing consolidated results, mimicking the return structure of `summarise_output_topics`
    """

    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_number_of_calls = 0
    out_message = list()

    # Logged content
    all_groups_logged_content = existing_logged_content

    # Check if we have data to process
    # Allow empty sampled_reference_table_df if sample_reference_table is True (it will be created from reference_table_df)
    if (
        (sampled_reference_table_df.empty and not sample_reference_table)
        or topic_summary_df.empty
        or reference_table_df.empty
    ):
        out_message = "Please upload reference table, topic summary, and sampled reference table files to continue with summarisation."
        print(out_message)
        raise Exception(out_message)

    # Ensure Group column exists
    if "Group" not in sampled_reference_table_df.columns:
        sampled_reference_table_df["Group"] = "All"
    if "Group" not in topic_summary_df.columns:
        topic_summary_df["Group"] = "All"
    if "Group" not in reference_table_df.columns:
        reference_table_df["Group"] = "All"

    # Sample reference table if requested
    if sample_reference_table:
        print(
            f"Sampling reference table with {no_of_sampled_summaries} summaries per group..."
        )
        sampled_reference_table_df, _ = sample_reference_table_summaries(
            reference_table_df,
            random_seed=random_seed,
            no_of_sampled_summaries=no_of_sampled_summaries,
            sample_reference_table_checkbox=sample_reference_table,
        )
        print(
            f"Sampling complete. {len(sampled_reference_table_df)} summaries selected."
        )

    # Get unique group values
    unique_values = sampled_reference_table_df["Group"].unique()

    if len(unique_values) > MAX_GROUPS:
        print(
            f"Warning: More than {MAX_GROUPS} unique values found in '{grouping_col}'. Processing only the first {MAX_GROUPS}."
        )
        unique_values = unique_values[:MAX_GROUPS]

    # Initialize accumulators for results across all groups
    acc_sampled_reference_table_df = pd.DataFrame()
    acc_topic_summary_df_revised = pd.DataFrame()
    acc_reference_table_df_revised = pd.DataFrame()
    acc_output_files = list()
    acc_log_output_files = list()
    acc_summarised_outputs = list()
    acc_latest_summary_completed = latest_summary_completed
    acc_out_metadata_str = out_metadata_str
    acc_summarised_output_markdown = ""
    acc_total_time_taken = 0.0
    acc_logged_content = list()

    if len(unique_values) == 1:
        # If only one unique value, no need for progress bar, iterate directly
        loop_object = unique_values
    else:
        # If multiple unique values, use tqdm progress bar
        loop_object = progress.tqdm(
            unique_values, desc="Summarising group", unit="groups"
        )

    for i, group_value in enumerate(loop_object):
        print(
            f"\nProcessing summary group: {grouping_col} = {group_value} ({i+1}/{len(unique_values)})"
        )

        # Filter data for current group
        filtered_sampled_reference_table_df = sampled_reference_table_df[
            sampled_reference_table_df["Group"] == group_value
        ].copy()
        filtered_topic_summary_df = topic_summary_df[
            topic_summary_df["Group"] == group_value
        ].copy()
        filtered_reference_table_df = reference_table_df[
            reference_table_df["Group"] == group_value
        ].copy()

        if filtered_sampled_reference_table_df.empty:
            print(f"No data for {grouping_col} = {group_value}. Skipping.")
            continue

        # Create unique file name for this group's outputs
        group_file_name = f"{reference_data_file_name}_{clean_column_name(str(group_value), max_length=15).replace(' ','_')}"

        # Call summarise_output_topics for the current group
        try:
            (
                seg_sampled_reference_table_df,
                seg_topic_summary_df_revised,
                seg_reference_table_df_revised,
                seg_output_files,
                seg_summarised_outputs,
                seg_latest_summary_completed,
                seg_out_metadata_str,
                seg_summarised_output_markdown,
                seg_log_output_files,
                seg_output_files_2,
                seg_acc_input_tokens,
                seg_acc_output_tokens,
                seg_acc_number_of_calls,
                seg_time_taken,
                seg_out_message,
                seg_logged_content,
            ) = summarise_output_topics(
                sampled_reference_table_df=filtered_sampled_reference_table_df,
                topic_summary_df=filtered_topic_summary_df,
                reference_table_df=filtered_reference_table_df,
                model_choice=model_choice,
                in_api_key=in_api_key,
                temperature=temperature,
                reference_data_file_name=group_file_name,
                summarised_outputs=list(),  # Fresh for each call
                latest_summary_completed=0,  # Reset for each group
                out_metadata_str="",  # Fresh for each call
                in_data_files=in_data_files,
                in_excel_sheets=in_excel_sheets,
                chosen_cols=chosen_cols,
                log_output_files=list(),  # Fresh for each call
                summarise_format_radio=summarise_format_radio,
                output_folder=output_folder,
                context_textbox=context_textbox,
                aws_access_key_textbox=aws_access_key_textbox,
                aws_secret_key_textbox=aws_secret_key_textbox,
                aws_region_textbox=aws_region_textbox,
                model_name_map=model_name_map,
                hf_api_key_textbox=hf_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                existing_logged_content=all_groups_logged_content,
                additional_summary_instructions_provided=additional_summary_instructions_provided,
                output_debug_files=output_debug_files,
                group_value=group_value,
                reasoning_suffix=reasoning_suffix,
                local_model=local_model,
                tokenizer=tokenizer,
                assistant_model=assistant_model,
                summarise_topic_descriptions_prompt=summarise_topic_descriptions_prompt,
                summarise_topic_descriptions_system_prompt=summarise_topic_descriptions_system_prompt,
                do_summaries=do_summaries,
                api_url=api_url,
            )

            # Aggregate results
            acc_sampled_reference_table_df = pd.concat(
                [acc_sampled_reference_table_df, seg_sampled_reference_table_df]
            )
            acc_topic_summary_df_revised = pd.concat(
                [acc_topic_summary_df_revised, seg_topic_summary_df_revised]
            )
            acc_reference_table_df_revised = pd.concat(
                [acc_reference_table_df_revised, seg_reference_table_df_revised]
            )

            # For lists, extend
            acc_output_files.extend(
                f for f in seg_output_files if f not in acc_output_files
            )
            acc_log_output_files.extend(
                f for f in seg_log_output_files if f not in acc_log_output_files
            )
            acc_summarised_outputs.extend(seg_summarised_outputs)

            acc_latest_summary_completed = seg_latest_summary_completed
            acc_out_metadata_str += (
                ("\n---\n" if acc_out_metadata_str else "")
                + f"Group {grouping_col}={group_value}:\n"
                + seg_out_metadata_str
            )
            acc_summarised_output_markdown = (
                seg_summarised_output_markdown  # Keep the latest markdown
            )
            acc_total_time_taken += float(seg_time_taken)
            acc_logged_content.extend(seg_logged_content)

            # Accumulate token counts
            acc_input_tokens += seg_acc_input_tokens
            acc_output_tokens += seg_acc_output_tokens
            acc_number_of_calls += seg_acc_number_of_calls

            print(
                f"Group {grouping_col} = {group_value} summarised. Time: {seg_time_taken:.2f}s"
            )

        except Exception as e:
            print(f"Error processing summary group {grouping_col} = {group_value}: {e}")
            # Optionally, decide if you want to continue with other groups or stop
            # For now, it will continue
            continue

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    # Create consolidated output files
    overall_file_name = clean_column_name(reference_data_file_name, max_length=20)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    # Save consolidated outputs
    if (
        not acc_topic_summary_df_revised.empty
        and not acc_reference_table_df_revised.empty
    ):
        # Sort the dataframes
        if "General topic" in acc_topic_summary_df_revised.columns:
            acc_topic_summary_df_revised["Number of responses"] = (
                acc_topic_summary_df_revised["Number of responses"].astype(int)
            )
            acc_topic_summary_df_revised.sort_values(
                [
                    "Group",
                    "Number of responses",
                    "General topic",
                    "Subtopic",
                    "Sentiment",
                ],
                ascending=[True, False, True, True, True],
                inplace=True,
            )
        elif "Main heading" in acc_topic_summary_df_revised.columns:
            acc_topic_summary_df_revised["Number of responses"] = (
                acc_topic_summary_df_revised["Number of responses"].astype(int)
            )
            acc_topic_summary_df_revised.sort_values(
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

        # Save consolidated files
        consolidated_topic_summary_path = (
            output_folder
            + overall_file_name
            + "_all_final_summ_unique_topics_"
            + model_choice_clean_short
            + ".csv"
        )
        consolidated_reference_table_path = (
            output_folder
            + overall_file_name
            + "_all_final_summ_reference_table_"
            + model_choice_clean_short
            + ".csv"
        )

        acc_topic_summary_df_revised.drop(
            ["1", "2", "3"], axis=1, errors="ignore"
        ).to_csv(consolidated_topic_summary_path, index=None, encoding="utf-8-sig")
        acc_reference_table_df_revised.drop(
            ["1", "2", "3"], axis=1, errors="ignore"
        ).to_csv(consolidated_reference_table_path, index=None, encoding="utf-8-sig")

        acc_output_files.extend(
            [consolidated_topic_summary_path, consolidated_reference_table_path]
        )

        # Create markdown output for display
        topic_summary_df_revised_display = acc_topic_summary_df_revised.apply(
            lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length))
        )
        acc_summarised_output_markdown = topic_summary_df_revised_display.to_markdown(
            index=False
        )

    out_message = "\n".join(out_message)
    out_message = (
        out_message
        + " "
        + f"Topic summarisation finished processing all groups. Total time: {acc_total_time_taken:.2f}s"
    )
    print(out_message)

    # The return signature should match summarise_output_topics
    return (
        acc_sampled_reference_table_df,
        acc_topic_summary_df_revised,
        acc_reference_table_df_revised,
        acc_output_files,
        acc_summarised_outputs,
        acc_latest_summary_completed,
        acc_out_metadata_str,
        acc_summarised_output_markdown,
        acc_log_output_files,
        acc_output_files,  # Duplicate for compatibility
        acc_input_tokens,
        acc_output_tokens,
        acc_number_of_calls,
        acc_total_time_taken,
        out_message,
        acc_logged_content,
    )


def convert_markdown_headers_to_excel_format(text: str) -> str:
    """
    Convert markdown headers to Excel-friendly format that preserves hierarchy.

    Converts:
    - # Header (H1) -> === HEADER === (most prominent)
    - ## Header (H2) -> --- Header --- (medium)
    - ### Header (H3) -> ── Header ── (less prominent)
    - #### Header (H4) -> • Header (with bullet)
    - ##### Header (H5) ->   • Header (indented)
    - ###### Header (H6) ->     • Header (more indented)

    Args:
        text (str): Text containing markdown headers

    Returns:
        str: Text with markdown headers converted to Excel-friendly format
    """
    if not text:
        return text

    lines = text.split("\n")
    converted_lines = []

    for line in lines:
        # Match markdown headers (# through ######)
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            header_level = len(header_match.group(1))  # Number of # characters
            header_text = header_match.group(2).strip()

            if header_level == 1:
                # H1: Most prominent - uppercase with double equals
                converted_line = f"=== {header_text.upper()} ==="
            elif header_level == 2:
                # H2: Medium prominence - title case with dashes
                converted_line = f"--- {header_text.title()} ---"
            elif header_level == 3:
                # H3: Less prominent - title case with single dashes
                converted_line = f"── {header_text.title()} ──"
            elif header_level == 4:
                # H4: Bullet with no indentation
                converted_line = f"• {header_text}"
            elif header_level == 5:
                # H5: Bullet with indentation
                converted_line = f"  • {header_text}"
            else:  # header_level == 6
                # H6: Bullet with more indentation
                converted_line = f"    • {header_text}"

            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    return "\n".join(converted_lines)


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def overall_summary(
    topic_summary_df: pd.DataFrame,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    reference_data_file_name: str,
    output_folder: str = OUTPUT_FOLDER,
    chosen_cols: List[str] = list(),
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    model_name_map: dict = model_name_map,
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    existing_logged_content: list = list(),
    api_url: str = None,
    output_debug_files: str = output_debug_files,
    log_output_files: list = list(),
    reasoning_suffix: str = reasoning_suffix,
    local_model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    summarise_everything_prompt: str = summarise_everything_prompt,
    comprehensive_summary_format_prompt: str = comprehensive_summary_format_prompt,
    comprehensive_summary_format_prompt_by_group: str = comprehensive_summary_format_prompt_by_group,
    summarise_everything_system_prompt: str = summarise_everything_system_prompt,
    do_summaries: str = "Yes",
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[
    List[str],
    List[str],
    int,
    str,
    List[str],
    List[str],
    int,
    int,
    int,
    float,
    List[dict],
]:
    """
    Create an overall summary of all responses based on a topic summary table.

    Args:
        topic_summary_df (pd.DataFrame): DataFrame containing topic summaries
        model_choice (str): Name of the LLM model to use
        in_api_key (str): API key for model access
        temperature (float): Temperature parameter for model generation
        reference_data_file_name (str): Name of reference data file
        output_folder (str, optional): Folder to save outputs. Defaults to OUTPUT_FOLDER.
        chosen_cols (List[str], optional): Columns to analyze. Defaults to empty list.
        context_textbox (str, optional): Additional context. Defaults to empty string.
        aws_access_key_textbox (str, optional): AWS access key. Defaults to empty string.
        aws_secret_key_textbox (str, optional): AWS secret key. Defaults to empty string.
        aws_region_textbox (str, optional): AWS region. Defaults to empty string.
        model_name_map (dict, optional): Mapping of model names. Defaults to model_name_map.
        hf_api_key_textbox (str, optional): Hugging Face API key. Defaults to empty string.
        existing_logged_content (list, optional): List of existing logged content. Defaults to empty list.
        output_debug_files (str, optional): Flag to indicate if debug files should be written. Defaults to "False".
        log_output_files (list, optional): List of existing logged content. Defaults to empty list.
        api_url (str, optional): API URL for inference-server models. Defaults to None.
        reasoning_suffix (str, optional): Suffix for reasoning. Defaults to reasoning_suffix.
        local_model (object, optional): Local model object. Defaults to None.
        tokenizer (object, optional): Tokenizer object. Defaults to None.
        assistant_model (object, optional): Assistant model object. Defaults to None.
        summarise_everything_prompt (str, optional): Prompt for overall summary
        comprehensive_summary_format_prompt (str, optional): Prompt for comprehensive summary format
        comprehensive_summary_format_prompt_by_group (str, optional): Prompt for group summary format
        summarise_everything_system_prompt (str, optional): System prompt for overall summary
        do_summaries (str, optional): Whether to generate summaries. Defaults to "Yes".
        progress (gr.Progress, optional): Progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        Tuple containing:
            List[str]: Output files
            List[str]: Text summarized outputs
            int: Latest summary completed
            str: Output metadata
            List[str]: Summarized outputs
            List[str]: Summarized outputs for DataFrame
            int: Number of input tokens
            int: Number of output tokens
            int: Number of API calls
            float: Time taken
            List[dict]: List of logged content
    """

    out_metadata = list()
    latest_summary_completed = 0
    output_files = list()
    txt_summarised_outputs = list()
    summarised_outputs = list()
    summarised_outputs_for_df = list()
    input_tokens_num = 0
    output_tokens_num = 0
    number_of_calls_num = 0
    time_taken = 0
    out_message = list()
    all_logged_content = list()
    all_prompts_content = list()
    all_summaries_content = list()
    all_metadata_content = list()
    all_groups_content = list()
    all_batches_content = list()
    all_model_choice_content = list()
    all_validated_content = list()
    task_type = "Overall summary"
    all_task_type_content = list()
    log_output_files = list()
    all_logged_content = list()
    all_file_names_content = list()
    tic = time.perf_counter()

    if "Group" not in topic_summary_df.columns:
        topic_summary_df["Group"] = "All"

    topic_summary_df = topic_summary_df.sort_values(
        by=["Group", "Number of responses"], ascending=[True, False]
    )

    unique_groups = sorted(topic_summary_df["Group"].unique())

    length_groups = len(unique_groups)

    if context_textbox and "The context of this analysis is" not in context_textbox:
        context_textbox = "The context of this analysis is '" + context_textbox + "'."

    if length_groups > 1:
        comprehensive_summary_format_prompt = (
            comprehensive_summary_format_prompt_by_group
        )
    else:
        comprehensive_summary_format_prompt = comprehensive_summary_format_prompt

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    batch_file_path_details = create_batch_file_path_details(reference_data_file_name)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    tic = time.perf_counter()

    if (
        (model_choice == CHOSEN_LOCAL_MODEL_TYPE)
        & (RUN_LOCAL_MODEL == "1")
        & (not local_model)
    ):
        progress(0.1, f"Using model: {CHOSEN_LOCAL_MODEL_TYPE}")
        local_model = get_model()
        tokenizer = get_tokenizer()
        assistant_model = get_assistant_model()

    summary_loop = tqdm(
        unique_groups, desc="Creating overall summary for groups", unit="groups"
    )

    if do_summaries == "Yes":
        model_source = model_name_map[model_choice]["source"]
        bedrock_runtime = connect_to_bedrock_runtime(
            model_name_map,
            model_choice,
            aws_access_key_textbox,
            aws_secret_key_textbox,
            aws_region_textbox,
        )

        for summary_group in summary_loop:

            print("Creating overall summary for group:", summary_group)

            # Get the group-specific DataFrame
            group_df = topic_summary_df.loc[
                topic_summary_df["Group"] == summary_group
            ].copy()

            # Prepare the system prompt first (needed for token counting)
            formatted_summarise_everything_system_prompt = (
                summarise_everything_system_prompt.format(
                    column_name=chosen_cols, consultation_context=context_textbox
                )
            )

            # Apply reasoning suffix for GPT-OSS models (Local, inference-server, or AWS)
            is_gpt_oss_model = (
                "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower()
            )

            if is_gpt_oss_model:
                # Use default reasoning suffix if not set
                effective_reasoning_suffix = (
                    reasoning_suffix if reasoning_suffix else "Reasoning: low"
                )
                if effective_reasoning_suffix:
                    formatted_summarise_everything_system_prompt = (
                        formatted_summarise_everything_system_prompt
                        + "\n"
                        + effective_reasoning_suffix
                    )
            elif "Local" in model_source and reasoning_suffix:
                # For other local models, use reasoning_suffix if provided
                formatted_summarise_everything_system_prompt = (
                    formatted_summarise_everything_system_prompt
                    + "\n"
                    + reasoning_suffix
                )

            # Create a test prompt with empty table to get base token count
            test_summary_text = ""
            test_formatted_summary_prompt = [
                summarise_everything_prompt.format(
                    topic_summary_table=test_summary_text,
                    summary_format=comprehensive_summary_format_prompt,
                )
            ]

            # Calculate base token count (system prompt + prompt template without table)
            full_test_text = (
                formatted_summarise_everything_system_prompt
                + "\n"
                + test_formatted_summary_prompt[0]
            )
            base_token_count = count_tokens_in_text(
                full_test_text, tokenizer, model_source
            )

            # Calculate available tokens for the summary table
            available_tokens = LLM_CONTEXT_LENGTH - base_token_count

            # Truncate DataFrame rows if needed to fit within context limit
            if len(group_df) > 0:
                # Start with all rows and check if they fit
                current_summary_text = group_df.to_markdown(index=False)
                current_token_count = count_tokens_in_text(
                    current_summary_text, tokenizer, model_source
                )

                # If the full table exceeds available tokens, truncate rows
                if current_token_count > available_tokens:
                    print(
                        f"Warning: Summary table for group '{summary_group}' exceeds context limit. "
                        f"Truncating rows. Table tokens: {current_token_count}, Available: {available_tokens}"
                    )

                    # Binary search approach: find the maximum number of rows that fit
                    # Start with all rows and reduce until we fit
                    num_rows = len(group_df)
                    min_rows = 0
                    max_rows = num_rows
                    best_df = group_df.iloc[:0]  # Empty DataFrame as fallback

                    # Try to find the maximum number of rows that fit
                    while min_rows < max_rows:
                        mid_rows = (min_rows + max_rows + 1) // 2
                        test_df = group_df.iloc[:mid_rows]
                        test_summary = test_df.to_markdown(index=False)
                        test_token_count = count_tokens_in_text(
                            test_summary, tokenizer, model_source
                        )

                        if test_token_count <= available_tokens:
                            best_df = test_df
                            min_rows = mid_rows
                        else:
                            max_rows = mid_rows - 1

                    # Use the best fitting DataFrame
                    group_df = best_df
                    print(
                        f"Truncated to {len(group_df)} rows (from {num_rows} original rows) "
                        f"to fit within context limit."
                    )

            # Create summary_text from (possibly truncated) DataFrame
            summary_text = group_df.to_markdown(index=False)

            formatted_summary_prompt = [
                summarise_everything_prompt.format(
                    topic_summary_table=summary_text,
                    summary_format=comprehensive_summary_format_prompt,
                )
            ]

            try:
                response, conversation_history, metadata, response_text = (
                    summarise_output_topics_query(
                        model_choice,
                        in_api_key,
                        temperature,
                        formatted_summary_prompt,
                        formatted_summarise_everything_system_prompt,
                        model_source,
                        bedrock_runtime,
                        local_model,
                        tokenizer=tokenizer,
                        assistant_model=assistant_model,
                        azure_endpoint=azure_endpoint_textbox,
                        api_url=api_url,
                    )
                )
                summarised_output_for_df = response_text
                summarised_output = response
            except Exception as e:
                print(
                    "Cannot create overall summary for group:",
                    summary_group,
                    "due to:",
                    e,
                )
                summarised_output = ""
                summarised_output_for_df = ""

            # Remove multiple consecutive line breaks (2 or more) and replace with single line break
            if summarised_output_for_df:
                summarised_output_for_df = re.sub(
                    r"\n{2,}", "\n", summarised_output_for_df
                )
                # Convert markdown headers to Excel-friendly format
                summarised_output_for_df = convert_markdown_headers_to_excel_format(
                    summarised_output_for_df
                )
            if summarised_output:
                summarised_output = re.sub(r"\n{2,}", "\n", summarised_output)

            summarised_outputs_for_df.append(summarised_output_for_df)
            summarised_outputs.append(summarised_output)
            txt_summarised_outputs.append(
                f"""Group name: {summary_group}\n""" + summarised_output
            )

            out_metadata.extend(metadata)
            out_metadata_str = ". ".join(out_metadata)

            full_prompt = (
                formatted_summarise_everything_system_prompt
                + "\n"
                + formatted_summary_prompt[0]
            )

            (
                current_prompt_content_logged,
                current_summary_content_logged,
                current_conversation_content_logged,
                current_metadata_content_logged,
            ) = process_debug_output_iteration(
                output_debug_files,
                output_folder,
                batch_file_path_details,
                model_choice_clean_short,
                full_prompt,
                summarised_output,
                conversation_history,
                metadata,
                log_output_files,
                task_type=task_type,
            )

            all_prompts_content.append(current_prompt_content_logged)
            all_summaries_content.append(current_summary_content_logged)
            # all_conversation_content.append(current_conversation_content_logged)
            all_metadata_content.append(current_metadata_content_logged)
            all_groups_content.append(summary_group)
            all_batches_content.append("1")
            all_model_choice_content.append(model_choice_clean_short)
            all_validated_content.append("No")
            all_task_type_content.append(task_type)
            all_file_names_content.append(reference_data_file_name)
            latest_summary_completed += 1
            clean_column_name(summary_group)

        # Write overall outputs to csv
        overall_summary_output_csv_path = (
            output_folder
            + batch_file_path_details
            + "_overall_summary_"
            + model_choice_clean_short
            + ".csv"
        )
        summarised_outputs_df = pd.DataFrame(
            data={"Group": unique_groups, "Summary": summarised_outputs_for_df}
        )
        summarised_outputs_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
            overall_summary_output_csv_path, index=None, encoding="utf-8-sig"
        )
        output_files.append(overall_summary_output_csv_path)

        summarised_outputs_df_for_display = pd.DataFrame(
            data={"Group": unique_groups, "Summary": summarised_outputs}
        )
        summarised_outputs_df_for_display["Summary"] = (
            summarised_outputs_df_for_display["Summary"]
            .apply(lambda x: markdown.markdown(x) if isinstance(x, str) else x)
            .str.replace(r"\n", "<br>", regex=False)
            .str.replace(r"(<br>\s*){2,}", "<br>", regex=True)
        )
        html_output_table = summarised_outputs_df_for_display.to_html(
            index=False, escape=False
        )

        output_files = list(set(output_files))

        input_tokens_num, output_tokens_num, number_of_calls_num = (
            calculate_tokens_from_metadata(
                out_metadata_str, model_choice, model_name_map
            )
        )

        # Check if beyond max time allowed for processing and break if necessary
        toc = time.perf_counter()
        time_taken = toc - tic

        out_message = "\n".join(out_message)
        out_message = (
            out_message
            + " "
            + f"Overall summary finished processing. Total time: {time_taken:.2f}s"
        )
        print(out_message)

        # Combine the logged content into a list of dictionaries
        all_logged_content = [
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
                all_summaries_content,
                all_metadata_content,
                all_batches_content,
                all_model_choice_content,
                all_validated_content,
                all_groups_content,
                all_task_type_content,
                all_file_names_content,
            )
        ]

        if isinstance(existing_logged_content, pd.DataFrame):
            existing_logged_content = existing_logged_content.to_dict(orient="records")

        out_logged_content = existing_logged_content + all_logged_content

    return (
        output_files,
        html_output_table,
        summarised_outputs_df,
        out_metadata_str,
        input_tokens_num,
        output_tokens_num,
        number_of_calls_num,
        time_taken,
        out_message,
        out_logged_content,
    )
