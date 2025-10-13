import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Tuple
import re
import spaces
import gradio as gr
import time
import markdown
import boto3
from tqdm import tqdm
import os
from tools.llm_api_call import generate_zero_shot_topics_df
from tools.prompts import summarise_topic_descriptions_prompt, summarise_topic_descriptions_system_prompt, system_prompt, summarise_everything_prompt, comprehensive_summary_format_prompt, summarise_everything_system_prompt, comprehensive_summary_format_prompt_by_group, summary_assistant_prefill, llm_deduplication_system_prompt, llm_deduplication_prompt, llm_deduplication_prompt_with_candidates
from tools.llm_funcs import construct_gemini_generative_model, process_requests, calculate_tokens_from_metadata, construct_azure_client, get_model, get_tokenizer, get_assistant_model, construct_gemini_generative_model, construct_azure_client, call_llm_with_markdown_table_checks
from tools.helper_functions import create_topic_summary_df_from_reference_table, load_in_data_file, get_basic_response_data, convert_reference_table_to_pivot_table, wrap_text, clean_column_name, get_file_name_no_ext, create_batch_file_path_details, read_file
from tools.aws_functions import connect_to_bedrock_runtime
from tools.config import OUTPUT_FOLDER, RUN_LOCAL_MODEL, MAX_COMMENT_CHARS, LLM_MAX_NEW_TOKENS, LLM_SEED, TIMEOUT_WAIT, NUMBER_OF_RETRY_ATTEMPTS, MAX_TIME_FOR_LOOP, BATCH_SIZE_DEFAULT, DEDUPLICATION_THRESHOLD, model_name_map, CHOSEN_LOCAL_MODEL_TYPE, REASONING_SUFFIX, MAX_SPACES_GPU_RUN_TIME, OUTPUT_DEBUG_FILES

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS
reasoning_suffix = REASONING_SUFFIX
output_debug_files = OUTPUT_DEBUG_FILES
max_text_length = 500

# DEDUPLICATION/SUMMARISATION FUNCTIONS
def deduplicate_categories(category_series: pd.Series, join_series: pd.Series, reference_df: pd.DataFrame, general_topic_series: pd.Series = None, merge_general_topics = "No", merge_sentiment:str="No", threshold: float = 90) -> pd.DataFrame:
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
    category_counts = reference_df['Subtopic'].value_counts().to_dict()

    # Initialize dictionaries for both category mapping and scores
    deduplication_map = {}
    match_scores = {}  # New dictionary to store match scores

    # First pass: Handle exact matches
    for category in category_series.unique():
        if category in deduplication_map:
            continue
            
        # Find all exact matches
        exact_matches = category_series[category_series.str.lower() == category.lower()].index.tolist()
        if len(exact_matches) > 1:
            # Find the variant with the highest count
            match_counts = {match: category_counts.get(category_series[match], 0) for match in exact_matches}
            most_common = max(match_counts.items(), key=lambda x: x[1])[0]
            most_common_category = category_series[most_common]
            
            # Map all exact matches to the most common variant and store score
            for match in exact_matches:
                deduplication_map[category_series[match]] = most_common_category
                match_scores[category_series[match]] = 100  # Exact matches get score of 100

    # Second pass: Handle fuzzy matches for remaining categories
    # Create a DataFrame to maintain the relationship between categories and general topics
    categories_df = pd.DataFrame({
        'category': category_series,
        'general_topic': general_topic_series
    }).drop_duplicates()

    for _, row in categories_df.iterrows():
        category = row['category']
        if category in deduplication_map:
            continue

        current_general_topic = row['general_topic']

        # Filter potential matches to only those within the same General topic if relevant
        if merge_general_topics == "No":
            potential_matches = categories_df[
                (categories_df['category'] != category) & 
                (categories_df['general_topic'] == current_general_topic)
            ]['category'].tolist()
        else:
            potential_matches = categories_df[
                (categories_df['category'] != category)
            ]['category'].tolist()

        matches = process.extract(category, 
                                potential_matches,
                                scorer=fuzz.WRatio,
                                score_cutoff=threshold)

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
    result_df = pd.DataFrame({
        'old_category': category_series + " | " + join_series,
        'deduplicated_category': category_series.map(lambda x: deduplication_map.get(x, x)),
        'match_score': category_series.map(lambda x: match_scores.get(x, 100))  # Add scores column
    })

    #print(result_df)

    return result_df

def deduplicate_topics(reference_df:pd.DataFrame,
                       topic_summary_df:pd.DataFrame,
                       reference_table_file_name:str,
                       unique_topics_table_file_name:str,
                       in_excel_sheets:str="",
                       merge_sentiment:str= "No",
                       merge_general_topics:str="No",
                       score_threshold:int=90,
                       in_data_files:List[str]=list(),
                       chosen_cols:List[str]="",
                       output_folder:str=OUTPUT_FOLDER,
                       deduplicate_topics:str="Yes"                       
                       ):
    '''
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
        in_data_files (List[str], optional): List of input data file paths. Defaults to [].
        chosen_cols (List[str], optional): List of chosen columns from the input data files. Defaults to "".
        output_folder (str, optional): Folder path to save output files. Defaults to OUTPUT_FOLDER.
        deduplicate_topics (str, optional): Whether to perform topic deduplication ("Yes" or "No"). Defaults to "Yes".
    '''
    output_files = list()
    log_output_files = list()
    file_data = pd.DataFrame()
    deduplicated_unique_table_markdown = ""


    if (len(reference_df["Response References"].unique()) == 1) | (len(topic_summary_df["Topic number"].unique()) == 1):
        print("Data file outputs are too short for deduplicating. Returning original data.")

        reference_file_out_path = output_folder + reference_table_file_name 
        unique_topics_file_out_path = output_folder + unique_topics_table_file_name

        output_files.append(reference_file_out_path)
        output_files.append(unique_topics_file_out_path)
        return reference_df, topic_summary_df, output_files, log_output_files, deduplicated_unique_table_markdown

    # For checking that data is not lost during the process
    initial_unique_references = len(reference_df["Response References"].unique())

    if topic_summary_df.empty:
        topic_summary_df = create_topic_summary_df_from_reference_table(reference_df)

        # Then merge the topic numbers back to the original dataframe
        reference_df = reference_df.merge(
            topic_summary_df[['General topic', 'Subtopic', 'Sentiment', 'Topic number']],
            on=['General topic', 'Subtopic', 'Sentiment'],
            how='left'
        )     

    if in_data_files and chosen_cols:
        file_data, data_file_names_textbox, total_number_of_batches = load_in_data_file(in_data_files, chosen_cols, 1, in_excel_sheets)
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)
        #raise Exception(out_message)

    # Run through this x times to try to get all duplicate topics
    if deduplicate_topics == "Yes":
        if "Group" not in reference_df.columns:
            reference_df["Group"] = "All"
        for i in range(0, 8):
            if merge_sentiment == "No":
                if merge_general_topics == "No":
                    reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    # Create an empty list to store results from each group
                    results = []
                    # Iterate over each group instead of using .apply()
                    for name, group in reference_df_unique.groupby(["General topic", "Sentiment", "Group"]):
                        # Run your function on the 'group' DataFrame
                        result = deduplicate_categories(
                            group["Subtopic"], 
                            group["Sentiment"], 
                            reference_df, 
                            general_topic_series=group["General topic"],
                            merge_general_topics="No",
                            threshold=score_threshold
                        )
                        results.append(result)
                    
                    # Concatenate all the results into a single DataFrame
                    deduplicated_topic_map_df = pd.concat(results).reset_index(drop=True)
                    # --- MODIFIED SECTION END ---

                else:
                    # This case should allow cross-topic matching but is still grouping by Sentiment
                    reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    reference_df_unique = reference_df.drop_duplicates("old_category")
                    
                    results = []
                    for name, group in reference_df_unique.groupby("Sentiment"):
                        result = deduplicate_categories(
                            group["Subtopic"], 
                            group["Sentiment"], 
                            reference_df, 
                            general_topic_series=None,
                            merge_general_topics="Yes",
                            threshold=score_threshold
                        )
                        results.append(result)
                    deduplicated_topic_map_df = pd.concat(results).reset_index(drop=True)

            else:
                if merge_general_topics == "No":
                    reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    reference_df_unique = reference_df.drop_duplicates("old_category")
                    
                    results = []
                    for name, group in reference_df_unique.groupby("General topic"):
                        result = deduplicate_categories(
                            group["Subtopic"], 
                            group["Sentiment"], 
                            reference_df, 
                            general_topic_series=group["General topic"],
                            merge_general_topics="No",
                            merge_sentiment=merge_sentiment, 
                            threshold=score_threshold
                        )
                        results.append(result)
                    deduplicated_topic_map_df = pd.concat(results).reset_index(drop=True)

                else:                    
                    reference_df["old_category"] = reference_df["Subtopic"] + " | " + reference_df["Sentiment"]
                    reference_df_unique = reference_df.drop_duplicates("old_category")

                    deduplicated_topic_map_df = deduplicate_categories(
                        reference_df_unique["Subtopic"], 
                        reference_df_unique["Sentiment"], 
                        reference_df, 
                        general_topic_series=None,
                        merge_general_topics="Yes",
                        merge_sentiment=merge_sentiment,
                        threshold=score_threshold
                    ).reset_index(drop=True)
        
            if deduplicated_topic_map_df['deduplicated_category'].isnull().all():
                print("No deduplicated categories found, skipping the following code.")

            else:
                # Remove rows where 'deduplicated_category' is blank or NaN
                deduplicated_topic_map_df = deduplicated_topic_map_df.loc[(deduplicated_topic_map_df['deduplicated_category'].str.strip() != '') & ~(deduplicated_topic_map_df['deduplicated_category'].isnull()), ['old_category','deduplicated_category', 'match_score']]

                reference_df = reference_df.merge(deduplicated_topic_map_df, on="old_category", how="left")

                reference_df.rename(columns={"Subtopic": "Subtopic_old", "Sentiment": "Sentiment_old"}, inplace=True)
                # Extract subtopic and sentiment from deduplicated_category
                reference_df["Subtopic"] = reference_df["deduplicated_category"].str.extract(r'^(.*?) \|')[0]  # Extract subtopic
                reference_df["Sentiment"] = reference_df["deduplicated_category"].str.extract(r'\| (.*)$')[0]  # Extract sentiment

                # Combine with old values to ensure no data is lost
                reference_df["Subtopic"] = reference_df["deduplicated_category"].combine_first(reference_df["Subtopic_old"])
                reference_df["Sentiment"] = reference_df["Sentiment"].combine_first(reference_df["Sentiment_old"])

            reference_df = reference_df.rename(columns={"General Topic":"General topic"}, errors="ignore")
            reference_df = reference_df[["Response References", "General topic", "Subtopic", "Sentiment", "Summary", "Start row of group", "Group"]]

            if merge_general_topics == "Yes":
                # Replace General topic names for each Subtopic with that for the Subtopic with the most responses
                # Step 1: Count the number of occurrences for each General topic and Subtopic combination
                count_df = reference_df.groupby(['Subtopic', 'General topic']).size().reset_index(name='Count')

                # Step 2: Find the General topic with the maximum count for each Subtopic
                max_general_topic = count_df.loc[count_df.groupby('Subtopic')['Count'].idxmax()]

                # Step 3: Map the General topic back to the original DataFrame
                reference_df = reference_df.merge(max_general_topic[['Subtopic', 'General topic']], on='Subtopic', suffixes=('', '_max'), how='left')

                reference_df['General topic'] = reference_df["General topic_max"].combine_first(reference_df["General topic"])        

            if merge_sentiment == "Yes":
                # Step 1: Count the number of occurrences for each General topic and Subtopic combination
                count_df = reference_df.groupby(['Subtopic', 'Sentiment']).size().reset_index(name='Count')

                # Step 2: Determine the number of unique Sentiment values for each Subtopic
                unique_sentiments = count_df.groupby('Subtopic')['Sentiment'].nunique().reset_index(name='UniqueCount')

                # Step 3: Update Sentiment to 'Mixed' where there is more than one unique sentiment
                reference_df = reference_df.merge(unique_sentiments, on='Subtopic', how='left')
                reference_df['Sentiment'] = reference_df.apply(
                    lambda row: 'Mixed' if row['UniqueCount'] > 1 else row['Sentiment'],
                    axis=1
                )

                # Clean up the DataFrame by dropping the UniqueCount column
                reference_df.drop(columns=['UniqueCount'], inplace=True)

            #print("reference_df:", reference_df)
            reference_df = reference_df[["Response References", "General topic", "Subtopic", "Sentiment", "Summary", "Start row of group", "Group"]]
            #reference_df.drop(['old_category', 'deduplicated_category', "Subtopic_old", "Sentiment_old"], axis=1, inplace=True, errors="ignore")        
        
        # Update reference summary column with all summaries
        reference_df["Summary"] = reference_df.groupby(
        ["Response References", "General topic", "Subtopic", "Sentiment"]
        )["Summary"].transform(' <br> '.join)

        # Check that we have not inadvertantly removed some data during the above process
        end_unique_references = len(reference_df["Response References"].unique())

        if initial_unique_references != end_unique_references:
            raise Exception(f"Number of unique references changed during processing: Initial={initial_unique_references}, Final={end_unique_references}")
        
        # Drop duplicates in the reference table - each comment should only have the same topic referred to once
        reference_df.drop_duplicates(['Response References', 'General topic', 'Subtopic', 'Sentiment'], inplace=True)

        # Remake topic_summary_df based on new reference_df
        topic_summary_df = create_topic_summary_df_from_reference_table(reference_df)

        # Then merge the topic numbers back to the original dataframe
        reference_df = reference_df.merge(
            topic_summary_df[['General topic', 'Subtopic', 'Sentiment', 'Group', 'Topic number']],
            on=['General topic', 'Subtopic', 'Sentiment', 'Group'],
            how='left'
        )       

    else: print("Topics have not beeen deduplicated")

    reference_table_file_name_no_ext = get_file_name_no_ext(reference_table_file_name)
    unique_topics_table_file_name_no_ext = get_file_name_no_ext(unique_topics_table_file_name)

    if not file_data.empty:
        basic_response_data = get_basic_response_data(file_data, chosen_cols)            
        reference_df_pivot = convert_reference_table_to_pivot_table(reference_df, basic_response_data)

        reference_pivot_file_path = output_folder + reference_table_file_name_no_ext + "_pivot_dedup.csv"
        reference_df_pivot.to_csv(reference_pivot_file_path, index=None, encoding='utf-8-sig')
        log_output_files.append(reference_pivot_file_path)

    

    reference_file_out_path = output_folder + reference_table_file_name_no_ext + "_dedup.csv"
    unique_topics_file_out_path = output_folder + unique_topics_table_file_name_no_ext + "_dedup.csv"
    reference_df.to_csv(reference_file_out_path, index = None, encoding='utf-8-sig')
    topic_summary_df.to_csv(unique_topics_file_out_path, index=None, encoding='utf-8-sig')

    output_files.append(reference_file_out_path)
    output_files.append(unique_topics_file_out_path)

    # Outputs for markdown table output
    topic_summary_df_revised_display = topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length)))
    deduplicated_unique_table_markdown = topic_summary_df_revised_display.to_markdown(index=False)

    return reference_df, topic_summary_df, output_files, log_output_files, deduplicated_unique_table_markdown

def deduplicate_topics_llm(reference_df:pd.DataFrame,
                          topic_summary_df:pd.DataFrame,
                          reference_table_file_name:str,
                          unique_topics_table_file_name:str,
                          model_choice:str,
                          in_api_key:str,
                          temperature:float,
                          model_source:str,
                          bedrock_runtime=None,
                          local_model=None,
                          tokenizer=None,
                          assistant_model=None,
                          in_excel_sheets:str="",
                          merge_sentiment:str="No",
                          merge_general_topics:str="No",
                          in_data_files:List[str]=list(),
                          chosen_cols:List[str]="",
                          output_folder:str=OUTPUT_FOLDER,
                          candidate_topics=None,
                          azure_endpoint:str=""
                          ):
    '''
    Deduplicate topics using LLM semantic understanding to identify and merge similar topics.

    Args:
        reference_df (pd.DataFrame): DataFrame containing reference data with topics.
        topic_summary_df (pd.DataFrame): DataFrame summarizing unique topics.
        reference_table_file_name (str): Base file name for the output reference table.
        unique_topics_table_file_name (str): Base file name for the output unique topics table.
        model_choice (str): The LLM model to use for deduplication.
        in_api_key (str): API key for the LLM service.
        temperature (float): Temperature setting for the LLM.
        model_source (str): Source of the model (AWS, Gemini, Local, etc.).
        bedrock_runtime: AWS Bedrock runtime client (if using AWS).
        local_model: Local model instance (if using local model).
        tokenizer: Tokenizer for local model.
        assistant_model: Assistant model for speculative decoding.
        in_excel_sheets (str, optional): Comma-separated list of Excel sheet names to load. Defaults to "".
        merge_sentiment (str, optional): Whether to merge topics regardless of sentiment ("Yes" or "No"). Defaults to "No".
        merge_general_topics (str, optional): Whether to merge topics across different general topics ("Yes" or "No"). Defaults to "No".
        in_data_files (List[str], optional): List of input data file paths. Defaults to [].
        chosen_cols (List[str], optional): List of chosen columns from the input data files. Defaults to "".
        output_folder (str, optional): Folder path to save output files. Defaults to OUTPUT_FOLDER.
        candidate_topics (optional): Candidate topics file for zero-shot guidance. Defaults to None.
    '''

    
    output_files = list()
    log_output_files = list()
    file_data = pd.DataFrame()
    deduplicated_unique_table_markdown = ""

    # Check if data is too short for deduplication
    if (len(reference_df["Response References"].unique()) == 1) | (len(topic_summary_df["Topic number"].unique()) == 1):
        print("Data file outputs are too short for deduplicating. Returning original data.")

        #print("reference_df:", reference_df)
        #print("topic_summary_df:", topic_summary_df)
        
        reference_file_out_path = output_folder + reference_table_file_name
        unique_topics_file_out_path = output_folder + unique_topics_table_file_name
        
        output_files.append(reference_file_out_path)
        output_files.append(unique_topics_file_out_path)
        return reference_df, topic_summary_df, output_files, log_output_files, deduplicated_unique_table_markdown

    # For checking that data is not lost during the process
    initial_unique_references = len(reference_df["Response References"].unique())

    # Create topic summary if it doesn't exist
    if topic_summary_df.empty:
        topic_summary_df = create_topic_summary_df_from_reference_table(reference_df)
        
        # Merge topic numbers back to the original dataframe
        reference_df = reference_df.merge(
            topic_summary_df[['General topic', 'Subtopic', 'Sentiment', 'Topic number']],
            on=['General topic', 'Subtopic', 'Sentiment'],
            how='left'
        )

    # Load data files if provided
    if in_data_files and chosen_cols:
        file_data, data_file_names_textbox, total_number_of_batches = load_in_data_file(in_data_files, chosen_cols, 1, in_excel_sheets)
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)

    # Process candidate topics if provided
    candidate_topics_table = ""
    if candidate_topics is not None:
        try:
            
            
            # Read and process candidate topics
            candidate_topics_df = read_file(candidate_topics.name)
            candidate_topics_df = candidate_topics_df.fillna("")
            candidate_topics_df = candidate_topics_df.astype(str)
            
            # Generate zero-shot topics DataFrame
            zero_shot_topics_df = generate_zero_shot_topics_df(candidate_topics_df, "No", False)
            
            if not zero_shot_topics_df.empty:
                candidate_topics_table = zero_shot_topics_df[['General topic', 'Subtopic']].to_markdown(index=False)
                print(f"Found {len(zero_shot_topics_df)} candidate topics to consider during deduplication")
        except Exception as e:
            print(f"Error processing candidate topics: {e}")
            candidate_topics_table = ""

    # Prepare topics table for LLM analysis
    topics_table = topic_summary_df[['General topic', 'Subtopic', 'Sentiment', 'Number of responses']].to_markdown(index=False)
    
    # Format the prompt with candidate topics if available
    if candidate_topics_table:
        formatted_prompt = llm_deduplication_prompt_with_candidates.format(
            topics_table=topics_table, 
            candidate_topics_table=candidate_topics_table
        )
    else:
        formatted_prompt = llm_deduplication_prompt.format(topics_table=topics_table)
    
    # Initialise conversation history
    conversation_history = []
    whole_conversation = []
    whole_conversation_metadata = []
    
    # Set up model clients based on model source
    if "Gemini" in model_source:
        client, config = construct_gemini_generative_model(
            in_api_key, temperature, model_choice, llm_deduplication_system_prompt, 
            max_tokens, LLM_SEED
        )
        bedrock_runtime = None
    elif "AWS" in model_source:
        if not bedrock_runtime:
            bedrock_runtime = boto3.client('bedrock-runtime')
        client = None
        config = None
    elif "Azure/OpenAI" in model_source:
        client, config = construct_azure_client(in_api_key, azure_endpoint)
        bedrock_runtime = None
    elif "Local" in model_source:
        client = None
        config = None
        bedrock_runtime = None
    else:
        raise ValueError(f"Unsupported model source: {model_source}")

    # Call LLM to get deduplication suggestions
    print("Calling LLM for topic deduplication analysis...")
    
    # Use the existing call_llm_with_markdown_table_checks function
    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(
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
        random_seed=LLM_SEED
    )

    # Generate debug files if enabled
    if OUTPUT_DEBUG_FILES == "True":
        try:
            # Create batch file path details for debug files
            batch_file_path_details = get_file_name_no_ext(reference_table_file_name) + "_llm_dedup"
            model_choice_clean_short = model_choice.replace("/", "_").replace(":", "_").replace(".", "_")
            
            # Create full prompt for debug output
            full_prompt = llm_deduplication_system_prompt + "\n" + formatted_prompt
            
            # Write debug files
            current_prompt_content_logged, current_summary_content_logged, current_conversation_content_logged, current_metadata_content_logged = process_debug_output_iteration(
                OUTPUT_DEBUG_FILES, 
                output_folder, 
                batch_file_path_details, 
                model_choice_clean_short, 
                full_prompt, 
                response_text, 
                whole_conversation, 
                whole_conversation_metadata, 
                log_output_files, 
                task_type="llm_deduplication"
            )
            
            print(f"Debug files written for LLM deduplication analysis")
            
        except Exception as e:
            print(f"Error writing debug files for LLM deduplication: {e}")

    # Parse the LLM response to extract merge suggestions
    merge_suggestions_df = pd.DataFrame()  # Initialize empty DataFrame for analysis results
    num_merges_applied = 0
    
    try:
        # Extract the markdown table from the response
        table_match = re.search(r'\|.*\|.*\n\|.*\|.*\n(\|.*\|.*\n)*', response_text, re.MULTILINE)
        if table_match:
            table_text = table_match.group(0)
            
            # Convert markdown table to DataFrame
            from io import StringIO
            merge_suggestions_df = pd.read_csv(StringIO(table_text), sep='|', skipinitialspace=True)
            
            # Clean up the DataFrame
            merge_suggestions_df = merge_suggestions_df.dropna(axis=1, how='all')  # Remove empty columns
            merge_suggestions_df.columns = merge_suggestions_df.columns.str.strip()
            
            # Remove rows where all values are NaN
            merge_suggestions_df = merge_suggestions_df.dropna(how='all')
            
            if not merge_suggestions_df.empty:
                print(f"LLM identified {len(merge_suggestions_df)} potential topic merges")
                
                # Apply the merges to the reference_df
                for _, row in merge_suggestions_df.iterrows():
                    original_general = row.get('Original General topic', '').strip()
                    original_subtopic = row.get('Original Subtopic', '').strip()
                    original_sentiment = row.get('Original Sentiment', '').strip()
                    merged_general = row.get('Merged General topic', '').strip()
                    merged_subtopic = row.get('Merged Subtopic', '').strip()
                    merged_sentiment = row.get('Merged Sentiment', '').strip()
                    
                    if all([original_general, original_subtopic, original_sentiment, 
                           merged_general, merged_subtopic, merged_sentiment]):
                        
                        # Find matching rows in reference_df
                        mask = (
                            (reference_df['General topic'] == original_general) &
                            (reference_df['Subtopic'] == original_subtopic) &
                            (reference_df['Sentiment'] == original_sentiment)
                        )
                        
                        if mask.any():
                            # Update the matching rows
                            reference_df.loc[mask, 'General topic'] = merged_general
                            reference_df.loc[mask, 'Subtopic'] = merged_subtopic
                            reference_df.loc[mask, 'Sentiment'] = merged_sentiment
                            num_merges_applied += 1
                            print(f"Merged: {original_general} | {original_subtopic} | {original_sentiment} -> {merged_general} | {merged_subtopic} | {merged_sentiment}")
            else:
                print("No merge suggestions found in LLM response")
        else:
            print("No markdown table found in LLM response")
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print("Continuing with original data...")

    # Update reference summary column with all summaries
    reference_df["Summary"] = reference_df.groupby(
        ["Response References", "General topic", "Subtopic", "Sentiment"]
    )["Summary"].transform(' <br> '.join)

    # Check that we have not inadvertently removed some data during the process
    end_unique_references = len(reference_df["Response References"].unique())

    if initial_unique_references != end_unique_references:
        raise Exception(f"Number of unique references changed during processing: Initial={initial_unique_references}, Final={end_unique_references}")
    
    # Drop duplicates in the reference table
    reference_df.drop_duplicates(['Response References', 'General topic', 'Subtopic', 'Sentiment'], inplace=True)

    # Remake topic_summary_df based on new reference_df
    topic_summary_df = create_topic_summary_df_from_reference_table(reference_df)

    # Merge the topic numbers back to the original dataframe
    reference_df = reference_df.merge(
        topic_summary_df[['General topic', 'Subtopic', 'Sentiment', 'Group', 'Topic number']],
        on=['General topic', 'Subtopic', 'Sentiment', 'Group'],
        how='left'
    )

    # Create pivot table if file data is available
    if not file_data.empty:
        basic_response_data = get_basic_response_data(file_data, chosen_cols)            
        reference_df_pivot = convert_reference_table_to_pivot_table(reference_df, basic_response_data)

        reference_pivot_file_path = output_folder + get_file_name_no_ext(reference_table_file_name) + "_pivot_dedup.csv"
        reference_df_pivot.to_csv(reference_pivot_file_path, index=None, encoding='utf-8-sig')
        log_output_files.append(reference_pivot_file_path)

    # Save analysis results CSV if merge suggestions were found
    if not merge_suggestions_df.empty:
        analysis_results_file_path = output_folder + get_file_name_no_ext(reference_table_file_name) + "_dedup_llm_analysis_results.csv"
        merge_suggestions_df.to_csv(analysis_results_file_path, index=None, encoding='utf-8-sig')
        log_output_files.append(analysis_results_file_path)
        print(f"Analysis results saved to: {analysis_results_file_path}")

    # Save output files
    reference_file_out_path = output_folder + get_file_name_no_ext(reference_table_file_name) + "_dedup.csv"
    unique_topics_file_out_path = output_folder + get_file_name_no_ext(unique_topics_table_file_name) + "_dedup.csv"
    reference_df.to_csv(reference_file_out_path, index=None, encoding='utf-8-sig')
    topic_summary_df.to_csv(unique_topics_file_out_path, index=None, encoding='utf-8-sig')

    output_files.append(reference_file_out_path)
    output_files.append(unique_topics_file_out_path)

    # Outputs for markdown table output
    topic_summary_df_revised_display = topic_summary_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length)))
    deduplicated_unique_table_markdown = topic_summary_df_revised_display.to_markdown(index=False)

    # Calculate token usage and timing information for logging
    total_input_tokens = 0
    total_output_tokens = 0
    number_of_calls = 1  # Single LLM call for deduplication
    
    # Extract token usage from conversation metadata
    if whole_conversation_metadata:
        for metadata in whole_conversation_metadata:
            if "input_tokens:" in metadata and "output_tokens:" in metadata:
                try:
                    input_tokens = int(metadata.split("input_tokens: ")[1].split(" ")[0])
                    output_tokens = int(metadata.split("output_tokens: ")[1].split(" ")[0])
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                except (ValueError, IndexError):
                    pass

    # Calculate estimated time taken (rough estimate based on token usage)
    estimated_time_taken = (total_input_tokens + total_output_tokens) / 1000  # Rough estimate in seconds

    return reference_df, topic_summary_df, output_files, log_output_files, deduplicated_unique_table_markdown, total_input_tokens, total_output_tokens, number_of_calls, estimated_time_taken#, num_merges_applied

def sample_reference_table_summaries(reference_df:pd.DataFrame,
                                     random_seed:int,
                                     no_of_sampled_summaries:int=150):
    
    '''
    Sample x number of summaries from which to produce summaries, so that the input token length is not too long.
    '''
    
    all_summaries = pd.DataFrame(columns=["General topic", "Subtopic", "Sentiment", "Group", "Response References", "Summary"])
    output_files = list()

    if "Group" not in reference_df.columns:
        reference_df["Group"] = "All"

    reference_df_grouped = reference_df.groupby(["General topic", "Subtopic", "Sentiment", "Group"])

    if 'Revised summary' in reference_df.columns:
        out_message = "Summary has already been created for this file"
        print(out_message)
        raise Exception(out_message)

    for group_keys, reference_df_group in reference_df_grouped:
        if len(reference_df_group["General topic"]) > 1:

            filtered_reference_df = reference_df_group.reset_index()

            filtered_reference_df_unique = filtered_reference_df.drop_duplicates(["General topic", "Subtopic", "Sentiment", "Summary"])

            # Sample n of the unique topic summaries. To limit the length of the text going into the summarisation tool
            filtered_reference_df_unique_sampled = filtered_reference_df_unique.sample(min(no_of_sampled_summaries, len(filtered_reference_df_unique)), random_state=random_seed)

            all_summaries = pd.concat([all_summaries, filtered_reference_df_unique_sampled])

    # If no responses/topics qualify, just go ahead with the original reference dataframe
    if all_summaries.empty:
        sampled_reference_table_df = reference_df
    else:
        sampled_reference_table_df = all_summaries.groupby(["General topic", "Subtopic", "Sentiment"]).agg({
        'Response References': 'size',  # Count the number of references
        'Summary': lambda x: '\n'.join([s.split(': ', 1)[1] for s in x if ': ' in s])  # Join substrings after ': '
        }).reset_index()

    sampled_reference_table_df = sampled_reference_table_df.loc[(sampled_reference_table_df["Sentiment"] != "Not Mentioned") & (sampled_reference_table_df["Response References"] > 1)]

    summarised_references_markdown = sampled_reference_table_df.to_markdown(index=False)

    return sampled_reference_table_df, summarised_references_markdown#, reference_df, topic_summary_df

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

def summarise_output_topics_query(model_choice:str, in_api_key:str, temperature:float, formatted_summary_prompt:str, summarise_topic_descriptions_system_prompt:str, model_source:str, bedrock_runtime:boto3.Session.client, local_model=list(), tokenizer=list(), assistant_model=list(), azure_endpoint:str=""):
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

    # Check if the input text exceeds the LLM context length
    from tools.config import LLM_CONTEXT_LENGTH
    
    # Combine system prompt and user prompt for token counting
    full_input_text = summarise_topic_descriptions_system_prompt + "\n" + formatted_summary_prompt[0] if isinstance(formatted_summary_prompt, list) else summarise_topic_descriptions_system_prompt + "\n" + formatted_summary_prompt
    
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
        #print("Using Gemini model:", model_choice)
        client, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
    elif "Azure/OpenAI" in model_source:
        client, config = construct_azure_client(in_api_key=os.environ.get("AZURE_INFERENCE_CREDENTIAL", ""), endpoint=azure_endpoint)
    elif "Local" in model_source:
        pass
        #print("Using local model: ", model_choice)
    elif "AWS" in model_source:
        pass
        #print("Using AWS Bedrock model:", model_choice)

    whole_conversation = [summarise_topic_descriptions_system_prompt] 

    # Process requests to large language model
    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(formatted_summary_prompt, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, client, client_config, model_choice, temperature, bedrock_runtime=bedrock_runtime, model_source=model_source, local_model=local_model, tokenizer=tokenizer, assistant_model=assistant_model, assistant_prefill=summary_assistant_prefill)

    summarised_output = re.sub(r'\n{2,}', '\n', response_text)  # Replace multiple line breaks with a single line break
    summarised_output = re.sub(r'^\n{1,}', '', summarised_output)  # Remove one or more line breaks at the start
    summarised_output = re.sub(r'\n', '<br>', summarised_output)  # Replace \n with more html friendly <br> tags
    summarised_output = summarised_output.strip()
    
    print("Finished summary query")

    return summarised_output, conversation_history, whole_conversation_metadata, response_text

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
    task_type: str
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
            conversation_strings = []
            for entry in conversation_history:
                if 'role' in entry and 'parts' in entry:
                    role = entry['role'].capitalize()
                    message = ' '.join(entry['parts']) if isinstance(entry['parts'], list) else str(entry['parts'])
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
            formatted_prompt_output_path = output_folder + batch_file_path_details +  "_full_prompt_" + model_choice_clean_short + "_" + current_task_type + ".txt"
            final_table_output_path = output_folder + batch_file_path_details + "_full_response_" + model_choice_clean_short + "_" + current_task_type + ".txt"
            whole_conversation_path = output_folder + batch_file_path_details + "_full_conversation_" + model_choice_clean_short + "_" + current_task_type + ".txt"
            whole_conversation_path_meta = output_folder + batch_file_path_details + "_metadata_" + model_choice_clean_short + "_" + current_task_type + ".txt"

            with open(formatted_prompt_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                f.write(current_prompt_content)
            with open(final_table_output_path, "w", encoding='utf-8-sig', errors='replace') as f:
                f.write(current_summary_content)
            with open(whole_conversation_path, "w", encoding='utf-8-sig', errors='replace') as f: 
                f.write(current_conversation_content)
            with open(whole_conversation_path_meta, "w", encoding='utf-8-sig', errors='replace') as f: 
                f.write(current_metadata_content)

            log_output_files.append(formatted_prompt_output_path)
            log_output_files.append(final_table_output_path)
            log_output_files.append(whole_conversation_path)
            log_output_files.append(whole_conversation_path_meta)
        except Exception as e: 
            print(f"Error in writing debug files for summary: {e}")
    
    # Return the content of the objects for the current iteration.
    # The caller can then append these to separate lists if accumulation is desired.
    return current_prompt_content, current_summary_content, current_conversation_content, current_metadata_content

@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def summarise_output_topics(sampled_reference_table_df:pd.DataFrame,
                            topic_summary_df:pd.DataFrame,
                            reference_table_df:pd.DataFrame,
                            model_choice:str,
                            in_api_key:str,
                            temperature:float,
                            reference_data_file_name:str,
                            summarised_outputs:list = list(),  
                            latest_summary_completed:int = 0,
                            out_metadata_str:str = "",
                            in_data_files:List[str]=list(),
                            in_excel_sheets:str="",
                            chosen_cols:List[str]=list(),
                            log_output_files:list[str]=list(),
                            summarise_format_radio:str="Return a summary up to two paragraphs long that includes as much detail as possible from the original text",
                            output_folder:str=OUTPUT_FOLDER,
                            context_textbox:str="",    
                            aws_access_key_textbox:str='',
                            aws_secret_key_textbox:str='',
                            model_name_map:dict=model_name_map,
                            hf_api_key_textbox:str='',
                            azure_endpoint_textbox:str='',
                            existing_logged_content:list=list(),
                            output_debug_files:str=output_debug_files,
                            reasoning_suffix:str=reasoning_suffix,
                            local_model:object=None, 
                            tokenizer:object=None,
                            assistant_model:object=None,                            
                            summarise_topic_descriptions_prompt:str=summarise_topic_descriptions_prompt, 
                            summarise_topic_descriptions_system_prompt:str=summarise_topic_descriptions_system_prompt,                            
                            do_summaries:str="Yes",                            
                            progress=gr.Progress(track_tqdm=True)):
    '''
    Create improved summaries of topics by consolidating raw batch-level summaries from the initial model run.

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
        existing_logged_content (list, optional): List of existing logged content. Defaults to empty list.
        output_debug_files (str, optional): Flag to indicate if debug files should be written. Defaults to "False".
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
    '''
    out_metadata = list()
    summarised_output_markdown = ""
    output_files = list()
    acc_input_tokens = 0
    acc_output_tokens = 0
    acc_number_of_calls = 0
    time_taken = 0
    out_metadata_str = "" # Output metadata is currently replaced on starting a summarisation task
    out_message = list()
    task_type = "Topic summarisation"
    topic_summary_df_revised = pd.DataFrame()

    all_prompts_content = list()
    all_summaries_content = list()
    all_conversation_content = list()
    all_metadata_content = list()
    all_groups_content = list()
    all_batches_content = list()
    all_model_choice_content = list()
    all_validated_content = list()
    all_task_type_content = list()
    all_logged_content = list()
    all_file_names_content = list()

    tic = time.perf_counter()

    model_choice_clean = clean_column_name(model_name_map[model_choice]["short_name"], max_length=20, front_characters=False)

    if log_output_files is None: log_output_files = list()   

    # Check for data for summarisations
    if not topic_summary_df.empty and not reference_table_df.empty:
        print("Unique table and reference table data found.")
    else:
        out_message = "Please upload a unique topic table and reference table file to continue with summarisation."
        print(out_message)
        raise Exception(out_message)
    
    if 'Revised summary' in reference_table_df.columns:
        out_message = "Summary has already been created for this file"
        print(out_message)
        raise Exception(out_message)
    
    # Load in data file and chosen columns if exists to create pivot table later
    if in_data_files and chosen_cols:
        file_data, data_file_names_textbox, total_number_of_batches = load_in_data_file(in_data_files, chosen_cols, 1, in_excel_sheets=in_excel_sheets)
    else:
        out_message = "No file data found, pivot table output will not be created."
        print(out_message)
        raise Exception(out_message)
    
    reference_table_df = reference_table_df.rename(columns={"General Topic":"General topic"}, errors="ignore")
    topic_summary_df = topic_summary_df.rename(columns={"General Topic":"General topic"}, errors="ignore")
    if "Group" not in reference_table_df.columns: reference_table_df["Group"] = "All"
    if "Group" not in topic_summary_df.columns: topic_summary_df["Group"] = "All"
    if "Group" not in sampled_reference_table_df.columns: sampled_reference_table_df["Group"] = "All"
   
    try: 
        all_summaries = sampled_reference_table_df["Summary"].tolist()
        all_groups = sampled_reference_table_df["Group"].tolist()
    except: 
        all_summaries = sampled_reference_table_df["Revised summary"].tolist()
        all_groups = sampled_reference_table_df["Group"].tolist()

    length_all_summaries = len(all_summaries)

    model_source = model_name_map[model_choice]["source"]

    if (model_source == "Local") & (RUN_LOCAL_MODEL == "1") & (not local_model):
        progress(0.1, f"Using global model: {CHOSEN_LOCAL_MODEL_TYPE}")
        local_model = get_model()
        tokenizer = get_tokenizer()
        assistant_model = get_assistant_model()

    summary_loop_description = "Revising topic-level summaries. " + str(latest_summary_completed) + " summaries completed so far."
    summary_loop = tqdm(range(latest_summary_completed, length_all_summaries), desc="Revising topic-level summaries", unit="summaries")   

    if do_summaries == "Yes":
        
        bedrock_runtime = connect_to_bedrock_runtime(model_name_map, model_choice, aws_access_key_textbox, aws_secret_key_textbox)

        batch_file_path_details = create_batch_file_path_details(reference_data_file_name)
        model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)

        for summary_no in summary_loop:
            print("Current summary number is:", summary_no)

            summary_text = all_summaries[summary_no]
            formatted_summary_prompt = [summarise_topic_descriptions_prompt.format(summaries=summary_text, summary_format=summarise_format_radio)]

            formatted_summarise_topic_descriptions_system_prompt = summarise_topic_descriptions_system_prompt.format(column_name=chosen_cols,consultation_context=context_textbox)

            if "Local" in model_source and reasoning_suffix: formatted_summarise_topic_descriptions_system_prompt = formatted_summarise_topic_descriptions_system_prompt + "\n" + reasoning_suffix

            try:
                response, conversation_history, metadata, response_text = summarise_output_topics_query(model_choice, in_api_key, temperature, formatted_summary_prompt, formatted_summarise_topic_descriptions_system_prompt, model_source, bedrock_runtime, local_model, tokenizer=tokenizer, assistant_model=assistant_model, azure_endpoint=azure_endpoint_textbox)
                summarised_output = response_text
            except Exception as e:
                print("Creating summary failed:", e)
                summarised_output = ""

            summarised_outputs.append(summarised_output)
            out_metadata.extend(metadata)
            out_metadata_str = '. '.join(out_metadata)            

            # Call the new function to process and log debug outputs for the current iteration.
            # The returned values are the contents of the prompt, summary, conversation, and metadata

            full_prompt = formatted_summarise_topic_descriptions_system_prompt + "\n" + formatted_summary_prompt[0]

            current_prompt_content_logged, current_summary_content_logged, current_conversation_content_logged, current_metadata_content_logged = \
                process_debug_output_iteration(
                    output_debug_files, 
                    output_folder, 
                    batch_file_path_details, 
                    model_choice_clean_short, 
                    full_prompt, 
                    summarised_output, 
                    conversation_history, 
                    metadata, 
                    log_output_files,
                    task_type=task_type
                )
            
            all_prompts_content.append(current_prompt_content_logged)
            all_summaries_content.append(current_summary_content_logged)
            #all_conversation_content.append(current_conversation_content_logged)
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
                print("Time taken for loop is greater than maximum time allowed. Exiting and restarting loop")
                summary_loop.close()
                tqdm._instances.clear()
                break

    # If all summaries completed, make final outputs
    if latest_summary_completed >= length_all_summaries:
        print("All summaries completed. Creating outputs.")       

        sampled_reference_table_df["Revised summary"] = summarised_outputs           

        join_cols = ["General topic", "Subtopic", "Sentiment"]
        join_plus_summary_cols = ["General topic", "Subtopic", "Sentiment", "Revised summary"]

        summarised_references_j = sampled_reference_table_df[join_plus_summary_cols].drop_duplicates(join_plus_summary_cols)

        topic_summary_df_revised = topic_summary_df.merge(summarised_references_j, on = join_cols, how = "left")

        # If no new summary is available, keep the original
        topic_summary_df_revised["Revised summary"] = topic_summary_df_revised["Revised summary"].combine_first(topic_summary_df_revised["Summary"])
        topic_summary_df_revised = topic_summary_df_revised[["General topic", "Subtopic", "Sentiment", "Group", "Number of responses", "Revised summary"]]

        # Ensure consistent Topic number assignment by recreating topic_summary_df from reference_df
        # if not reference_table_df_revised.empty:
        #     topic_summary_df_revised = create_topic_summary_df_from_reference_table(reference_table_df_revised)
        #     # Keep the revised summary column
        #     topic_summary_df_revised["Revised summary"] = topic_summary_df_revised["Revised summary"].combine_first(topic_summary_df_revised["Summary"])
        #     topic_summary_df_revised = topic_summary_df_revised[["General topic", "Subtopic", "Sentiment", "Group", "Number of responses", "Topic number", "Revised summary"]]

        # Replace all instances of 'Rows X to Y:' that remain on some topics that have not had additional summaries
        topic_summary_df_revised["Revised summary"] = topic_summary_df_revised["Revised summary"].str.replace("^Rows\s+\d+\s+to\s+\d+:\s*", "", regex=True) 
        topic_summary_df_revised["Topic number"] = range(1, len(topic_summary_df_revised) + 1)
        
        # If no new summary is available, keep the original. Also join on topic number to ensure consistent topic number assignment
        reference_table_df_revised = reference_table_df.copy()        
        reference_table_df_revised = reference_table_df_revised.drop("Topic number", axis=1, errors="ignore")

        # Ensure reference table has Topic number column
        if "Topic number" not in reference_table_df_revised.columns or "Revised summary" not in reference_table_df_revised.columns:
            if "Topic number" in topic_summary_df_revised.columns and "Revised summary" in topic_summary_df_revised.columns:
                reference_table_df_revised = reference_table_df_revised.merge(
                    topic_summary_df_revised[['General topic', 'Subtopic', 'Sentiment', 'Group', 'Topic number', 'Revised summary']],
                    on=['General topic', 'Subtopic', 'Sentiment', 'Group'],
                    how='left'
                )
        
        reference_table_df_revised["Revised summary"] = reference_table_df_revised["Revised summary"].combine_first(reference_table_df_revised["Summary"])
        reference_table_df_revised = reference_table_df_revised.drop("Summary", axis=1, errors="ignore")

        # Remove topics that are tagged as 'Not Mentioned'
        topic_summary_df_revised = topic_summary_df_revised.loc[topic_summary_df_revised["Sentiment"] != "Not Mentioned", :]
        reference_table_df_revised = reference_table_df_revised.loc[reference_table_df_revised["Sentiment"] != "Not Mentioned", :]

        # Combine the logged content into a list of dictionaries         
        all_logged_content = [
            {"prompt": prompt, "response": summary, "metadata": metadata, "batch": batch, "model_choice": model_choice, "validated": validated, "group": group, "task_type": task_type, "file_name": file_name}
            for prompt, summary, metadata, batch, model_choice, validated, group, task_type, file_name in zip(all_prompts_content, all_summaries_content, all_metadata_content, all_batches_content, all_model_choice_content, all_validated_content, all_groups_content, all_task_type_content, all_file_names_content)
        ]

        if isinstance(existing_logged_content, pd.DataFrame):
            existing_logged_content = existing_logged_content.to_dict(orient="records")

        out_logged_content = existing_logged_content + all_logged_content

        ### Save output files

        if not file_data.empty:
            basic_response_data = get_basic_response_data(file_data, chosen_cols)
            reference_table_df_revised_pivot = convert_reference_table_to_pivot_table(reference_table_df_revised, basic_response_data)

            ### Save pivot file to log area
            reference_table_df_revised_pivot_path = output_folder + batch_file_path_details + "_summarised_reference_table_pivot_" + model_choice_clean + ".csv"
            reference_table_df_revised_pivot.to_csv(reference_table_df_revised_pivot_path, index=None, encoding='utf-8-sig')
            log_output_files.append(reference_table_df_revised_pivot_path)

        # Save to file
        topic_summary_df_revised_path = output_folder + batch_file_path_details + "_summarised_unique_topics_table_" + model_choice_clean + ".csv"
        topic_summary_df_revised.to_csv(topic_summary_df_revised_path, index = None, encoding='utf-8-sig')

        reference_table_df_revised_path = output_folder + batch_file_path_details + "_summarised_reference_table_" + model_choice_clean + ".csv"
        reference_table_df_revised.to_csv(reference_table_df_revised_path, index = None, encoding='utf-8-sig')

        output_files.extend([reference_table_df_revised_path, topic_summary_df_revised_path])

        ###
        topic_summary_df_revised_display = topic_summary_df_revised.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=max_text_length)))
        summarised_output_markdown = topic_summary_df_revised_display.to_markdown(index=False)

        # Ensure same file name not returned twice
        output_files = list(set(output_files))
        log_output_files = list(set(log_output_files))

        acc_input_tokens, acc_output_tokens, acc_number_of_calls = calculate_tokens_from_metadata(out_metadata_str, model_choice, model_name_map)

        toc = time.perf_counter()
        time_taken = toc - tic

        out_message = '\n'.join(out_message)
        out_message = out_message + " " + f"Topic summarisation finished processing. Total time: {time_taken:.2f}s"
        print(out_message)

        return sampled_reference_table_df, topic_summary_df_revised, reference_table_df_revised, output_files, summarised_outputs, latest_summary_completed, out_metadata_str, summarised_output_markdown, log_output_files, output_files, acc_input_tokens, acc_output_tokens, acc_number_of_calls, time_taken, out_message, out_logged_content

@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def overall_summary(topic_summary_df:pd.DataFrame,
                    model_choice:str,
                    in_api_key:str,
                    temperature:float,
                    reference_data_file_name:str,
                    output_folder:str=OUTPUT_FOLDER,
                    chosen_cols:List[str]=list(),
                    context_textbox:str="",
                    aws_access_key_textbox:str='',
                    aws_secret_key_textbox:str='',
                    model_name_map:dict=model_name_map,
                    hf_api_key_textbox:str='',
                    azure_endpoint_textbox:str='',
                    existing_logged_content:list=list(),
                    output_debug_files:str=output_debug_files,
                    log_output_files:list=list(),                    
                    reasoning_suffix:str=reasoning_suffix,                    
                    local_model:object=None,
                    tokenizer:object=None,
                    assistant_model:object=None,
                    summarise_everything_prompt:str=summarise_everything_prompt,
                    comprehensive_summary_format_prompt:str=comprehensive_summary_format_prompt,
                    comprehensive_summary_format_prompt_by_group:str=comprehensive_summary_format_prompt_by_group,
                    summarise_everything_system_prompt:str=summarise_everything_system_prompt,
                    do_summaries:str="Yes",                            
                    progress=gr.Progress(track_tqdm=True)) -> Tuple[List[str], List[str], int, str, List[str], List[str], int, int, int, float, List[dict]]:
    '''
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
        model_name_map (dict, optional): Mapping of model names. Defaults to model_name_map.
        hf_api_key_textbox (str, optional): Hugging Face API key. Defaults to empty string.
        existing_logged_content (list, optional): List of existing logged content. Defaults to empty list.
        output_debug_files (str, optional): Flag to indicate if debug files should be written. Defaults to "False".
        log_output_files (list, optional): List of existing logged content. Defaults to empty list.        
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
    '''

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
    all_conversation_content = list()
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

    topic_summary_df = topic_summary_df.sort_values(by=["Group", "Number of responses"], ascending=[True, False])

    unique_groups = sorted(topic_summary_df["Group"].unique())

    length_groups = len(unique_groups)

    if length_groups > 1:
        comprehensive_summary_format_prompt = comprehensive_summary_format_prompt_by_group        
    else:
        comprehensive_summary_format_prompt = comprehensive_summary_format_prompt

    batch_file_path_details = create_batch_file_path_details(reference_data_file_name)
    model_choice_clean = model_name_map[model_choice]["short_name"]
    model_choice_clean_short = clean_column_name(model_choice_clean, max_length=20, front_characters=False)

    tic = time.perf_counter()

    if (model_choice == CHOSEN_LOCAL_MODEL_TYPE) & (RUN_LOCAL_MODEL == "1") & (not local_model):
        progress(0.1, f"Using model: {CHOSEN_LOCAL_MODEL_TYPE}")
        local_model = get_model()
        tokenizer = get_tokenizer()
        assistant_model = get_assistant_model()

    summary_loop = tqdm(unique_groups, desc="Creating overall summary for groups", unit="groups")   

    if do_summaries == "Yes":
        model_source = model_name_map[model_choice]["source"]
        bedrock_runtime = connect_to_bedrock_runtime(model_name_map, model_choice, aws_access_key_textbox, aws_secret_key_textbox)

        for summary_group in summary_loop:

            print("Creating overall summary for group:", summary_group)

            summary_text = topic_summary_df.loc[topic_summary_df["Group"]==summary_group].to_markdown(index=False)
            
            formatted_summary_prompt = [summarise_everything_prompt.format(topic_summary_table=summary_text, summary_format=comprehensive_summary_format_prompt)]

            formatted_summarise_everything_system_prompt = summarise_everything_system_prompt.format(column_name=chosen_cols,consultation_context=context_textbox)

            if "Local" in model_source and reasoning_suffix: formatted_summarise_everything_system_prompt = formatted_summarise_everything_system_prompt + "\n" + reasoning_suffix
            
            try:
                response, conversation_history, metadata, response_text = summarise_output_topics_query(model_choice, in_api_key, temperature, formatted_summary_prompt, formatted_summarise_everything_system_prompt, model_source, bedrock_runtime, local_model, tokenizer=tokenizer, assistant_model=assistant_model, azure_endpoint=azure_endpoint_textbox)
                summarised_output_for_df = response_text
                summarised_output = response
            except Exception as e:
                print("Cannot create overall summary for group:", summary_group, "due to:", e)
                summarised_output = ""
                summarised_output_for_df = ""

            summarised_outputs_for_df.append(summarised_output_for_df)
            summarised_outputs.append(summarised_output)
            txt_summarised_outputs.append(f"""Group name: {summary_group}\n""" + summarised_output)

            out_metadata.extend(metadata)
            out_metadata_str = '. '.join(out_metadata)

            full_prompt = formatted_summarise_everything_system_prompt + "\n" + formatted_summary_prompt[0]

            current_prompt_content_logged, current_summary_content_logged, current_conversation_content_logged, current_metadata_content_logged = process_debug_output_iteration(output_debug_files, output_folder, batch_file_path_details, model_choice_clean_short, full_prompt, summarised_output, conversation_history, metadata, log_output_files, task_type=task_type)            

            all_prompts_content.append(current_prompt_content_logged)
            all_summaries_content.append(current_summary_content_logged)
            #all_conversation_content.append(current_conversation_content_logged)
            all_metadata_content.append(current_metadata_content_logged)
            all_groups_content.append(summary_group)
            all_batches_content.append("1")
            all_model_choice_content.append(model_choice_clean_short)
            all_validated_content.append("No")
            all_task_type_content.append(task_type)
            all_file_names_content.append(reference_data_file_name)
            latest_summary_completed += 1
            summary_group_short = clean_column_name(summary_group)

        # Write overall outputs to csv
        overall_summary_output_csv_path = output_folder + batch_file_path_details + "_overall_summary_" + model_choice_clean_short + ".csv" 
        summarised_outputs_df = pd.DataFrame(data={"Group":unique_groups, "Summary":summarised_outputs_for_df})
        summarised_outputs_df.to_csv(overall_summary_output_csv_path, index=None, encoding='utf-8-sig')
        output_files.append(overall_summary_output_csv_path)

        summarised_outputs_df_for_display = pd.DataFrame(data={"Group":unique_groups, "Summary":summarised_outputs})
        summarised_outputs_df_for_display['Summary'] = summarised_outputs_df_for_display['Summary'].apply(
            lambda x: markdown.markdown(x) if isinstance(x, str) else x
        ).str.replace(r"\n", "<br>", regex=False)
        html_output_table = summarised_outputs_df_for_display.to_html(index=False, escape=False)

        output_files = list(set(output_files))

        input_tokens_num, output_tokens_num, number_of_calls_num = calculate_tokens_from_metadata(out_metadata_str, model_choice, model_name_map)

        # Check if beyond max time allowed for processing and break if necessary
        toc = time.perf_counter()
        time_taken = toc - tic

        out_message = '\n'.join(out_message)
        out_message = out_message + " " + f"Overall summary finished processing. Total time: {time_taken:.2f}s"
        print(out_message)

        # Combine the logged content into a list of dictionaries         
        all_logged_content = [
            {"prompt": prompt, "response": summary, "metadata": metadata, "batch": batch, "model_choice": model_choice, "validated": validated, "group": group, "task_type": task_type, "file_name": file_name}
            for prompt, summary, metadata, batch, model_choice, validated, group, task_type, file_name in zip(all_prompts_content, all_summaries_content, all_metadata_content, all_batches_content, all_model_choice_content, all_validated_content, all_groups_content, all_task_type_content, all_file_names_content)
        ]

        if isinstance(existing_logged_content, pd.DataFrame):
            existing_logged_content = existing_logged_content.to_dict(orient="records")

        out_logged_content = existing_logged_content + all_logged_content

    return output_files, html_output_table, summarised_outputs_df, out_metadata_str, input_tokens_num, output_tokens_num, number_of_calls_num, time_taken, out_message, out_logged_content