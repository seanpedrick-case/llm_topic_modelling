import os
import google.generativeai as ai
import pandas as pd
import numpy as np
import gradio as gr
import markdown
import time
import boto3
import json
import math
import string
import re
import spaces
from rapidfuzz import process, fuzz
from tqdm import tqdm
from gradio import Progress
from typing import List, Tuple
from io import StringIO

from tools.prompts import initial_table_prompt, prompt2, prompt3, system_prompt,add_existing_topics_system_prompt, add_existing_topics_prompt
from tools.helper_functions import put_columns_in_df, wrap_text
from tools.llm_funcs import load_model, construct_gemini_generative_model
from tools.llm_api_call import load_in_data_file, get_basic_response_data, data_file_to_markdown_table, clean_column_name,  convert_response_text_to_markdown_table, call_llm_with_markdown_table_checks,  ResponseObject, max_tokens, max_time_for_loop, batch_size_default,  GradioFileData
from tools.config import MAX_OUTPUT_VALIDATION_ATTEMPTS,  RUN_LOCAL_MODEL, model_name_map, OUTPUT_FOLDER, CHOSEN_LOCAL_MODEL_TYPE, LOCAL_REPO_ID, LOCAL_MODEL_FILE, LOCAL_MODEL_FOLDER

def write_llm_output_and_logs_verify(responses: List[ResponseObject],
                              whole_conversation: List[str],
                              whole_conversation_metadata: List[str],
                              file_name: str,
                              latest_batch_completed: int,
                              start_row:int,
                              end_row:int,
                              model_choice_clean: str,
                              temperature: float,
                              log_files_output_paths: List[str],
                              existing_reference_df:pd.DataFrame,
                              existing_topics_df:pd.DataFrame,
                              batch_size_number:int,
                              in_column:str,                              
                              first_run: bool = False,
                              output_folder:str=OUTPUT_FOLDER) -> None:
    """
    Writes the output of the large language model requests and logs to files.

    Parameters:
    - responses (List[ResponseObject]): A list of ResponseObject instances containing the text and usage metadata of the responses.
    - whole_conversation (List[str]): A list of strings representing the complete conversation including prompts and responses.
    - whole_conversation_metadata (List[str]): A list of strings representing metadata about the whole conversation.
    - file_name (str): The base part of the output file name.
    - latest_batch_completed (int): The index of the current batch.
    - start_row (int): Start row of the current batch.
    - end_row (int): End row of the current batch.
    - model_choice_clean (str): The cleaned model choice string.
    - temperature (float): The temperature parameter used in the model.
    - log_files_output_paths (List[str]): A list of paths to the log files.
    - existing_reference_df (pd.DataFrame): The existing reference dataframe mapping response numbers to topics.
    - existing_topics_df (pd.DataFrame): The existing unique topics dataframe 
    - first_run (bool): A boolean indicating if this is the first run through this function in this process. Defaults to False.
    - output_folder (str): A string indicating the folder to output to
    """
    unique_topics_df_out_path = []
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    unique_topics_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame()
    markdown_table = ""
    out_reference_df = pd.DataFrame()
    out_unique_topics_df = pd.DataFrame()
    batch_file_path_details = "error"

    # If there was an error in parsing, return boolean saying error
    is_error = False

    # Convert conversation to string and add to log outputs
    whole_conversation_str = '\n'.join(whole_conversation)
    whole_conversation_metadata_str = '\n'.join(whole_conversation_metadata)

    start_row_reported = start_row + 1

    # Example usage
    in_column_cleaned = clean_column_name(in_column, max_length=20)

    # Need to reduce output file names as full length files may be too long
    file_name = clean_column_name(file_name, max_length=30)    

    # Save outputs for each batch. If master file created, label file as master
    batch_file_path_details = f"{file_name}_batch_{latest_batch_completed + 1}_size_{batch_size_number}_col_{in_column_cleaned}"
    row_number_string_start = f"Rows {start_row_reported} to {end_row}: "

    whole_conversation_path = output_folder + batch_file_path_details + "_full_conversation_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"
    whole_conversation_path_meta = output_folder + batch_file_path_details + "_metadata_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

    with open(whole_conversation_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_str)

    with open(whole_conversation_path_meta, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_metadata_str)

    #log_files_output_paths.append(whole_conversation_path)
    log_files_output_paths.append(whole_conversation_path_meta)
    
    if isinstance(responses[-1], ResponseObject): response_text =  responses[-1].text
    elif "choices" in responses[-1]: response_text =  responses[-1]["choices"][0]['text']
    else: response_text =  responses[-1].text

    # Convert response text to a markdown table
    try:
        topic_with_response_df, is_error = convert_response_text_to_markdown_table(response_text, table_type="Verify titles table")
    except Exception as e:
        print("Error in parsing markdown table from response text:", e)
        return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_file_path_details, is_error   

    # Rename columns to ensure consistent use of data frames later in code
    topic_with_response_df.columns = ["Response References", "Is this a suitable title", "Explanation", "Alternative title"]


    # # Table to map references to topics
    reference_data = []

    # Iterate through each row in the original DataFrame
    for index, row in topic_with_response_df.iterrows():
        #references = re.split(r',\s*|\s+', str(row.iloc[4])) if pd.notna(row.iloc[4]) else ""
        references = re.findall(r'\d+', str(row.iloc[0])) if pd.notna(row.iloc[0]) else []
        topic = row.iloc[1] if pd.notna(row.iloc[1]) else ""
        summary = row.iloc[2] if pd.notna(row.iloc[2]) else ""
        suggested_title = row.iloc[3] if pd.notna(row.iloc[3]) else ""

        #summary = row_number_string_start + summary

        # Create a new entry for each reference number
        for ref in references:
            # Add start_row back onto reference_number
            try:
                response_ref_no =  str(int(ref) + int(start_row))
            except ValueError:
                print("Reference is not a number")
                continue

            row_data = {
                'Response References': response_ref_no,
                'Is this a suitable title': topic,
                'Explanation': summary,
                "Start row of group": start_row_reported,
                "Suggested title": suggested_title
            }

            reference_data.append(row_data)

    # Create a new DataFrame from the reference data
    new_reference_df = pd.DataFrame(reference_data)

    print("new_reference_df:", new_reference_df)

    # Append on old reference data
    out_reference_df = pd.concat([new_reference_df, existing_reference_df]).dropna(how='all')

    # # Remove duplicate Response References for the same topic
    # out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    # Try converting response references column to int, keep as string if fails
    try:
        out_reference_df["Response References"] = out_reference_df["Response References"].astype(int)
    except Exception as e:
        print("Could not convert Response References column to integer due to", e)
        print("out_reference_df['Response References']:", out_reference_df["Response References"].head())

    out_reference_df.sort_values(["Start row of group", "Response References"], inplace=True)

    # # Each topic should only be associated with each individual response once
    # out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    # # Save the new DataFrame to CSV
    # reference_table_out_path = output_folder + batch_file_path_details + "_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"    

    # # Table of all unique topics with descriptions
    # #print("topic_with_response_df:", topic_with_response_df)
    # new_unique_topics_df = topic_with_response_df[["General Topic", "Subtopic", "Sentiment"]]

    # new_unique_topics_df = new_unique_topics_df.rename(columns={new_unique_topics_df.columns[0]: "General Topic", new_unique_topics_df.columns[1]: "Subtopic", new_unique_topics_df.columns[2]: "Sentiment"})
    
    # # Join existing and new unique topics
    # out_unique_topics_df = pd.concat([new_unique_topics_df, existing_topics_df]).dropna(how='all')

    # out_unique_topics_df = out_unique_topics_df.rename(columns={out_unique_topics_df.columns[0]: "General Topic", out_unique_topics_df.columns[1]: "Subtopic", out_unique_topics_df.columns[2]: "Sentiment"})

    # out_unique_topics_df = out_unique_topics_df.drop_duplicates(["General Topic", "Subtopic", "Sentiment"]).\
    #         drop(["Response References", "Summary"], axis = 1, errors="ignore") 

    # # Get count of rows that refer to particular topics
    # reference_counts = out_reference_df.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    # 'Response References': 'size',  # Count the number of references
    # 'Summary': ' <br> '.join
    # }).reset_index()

    # # Join the counts to existing_unique_topics_df
    # out_unique_topics_df = out_unique_topics_df.merge(reference_counts, how='left', on=["General Topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    #out_reference_df = topic_with_response_df
    out_unique_topics_df = topic_with_response_df

    topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
    unique_topics_df_out_path = output_folder + batch_file_path_details + "_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
    reference_table_out_path = output_folder + batch_file_path_details + "_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv" 

    return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_file_path_details, is_error

@spaces.GPU
def verify_titles(in_data_file,
              file_data:pd.DataFrame,
              existing_topics_table:pd.DataFrame,
              existing_reference_df:pd.DataFrame,
              existing_unique_topics_df:pd.DataFrame,
              unique_table_df_display_table_markdown:str,
              file_name:str,
              num_batches:int,
              in_api_key:str,
              temperature:float,
              chosen_cols:List[str],
              model_choice:str,
              candidate_topics: GradioFileData = None,
              latest_batch_completed:int=0,
              out_message:List=[],
              out_file_paths:List = [],
              log_files_output_paths:List = [],
              first_loop_state:bool=False,
              whole_conversation_metadata_str:str="",
              initial_table_prompt:str=initial_table_prompt,
              prompt2:str=prompt2,
              prompt3:str=prompt3,
              system_prompt:str=system_prompt,
              add_existing_topics_system_prompt:str=add_existing_topics_system_prompt,
              add_existing_topics_prompt:str=add_existing_topics_prompt,
              number_of_prompts_used:int=1,
              batch_size:int=50,
              context_textbox:str="",
              time_taken:float = 0,
              sentiment_checkbox:str = "Negative, Neutral, or Positive",
              force_zero_shot_radio:str = "No",
              in_excel_sheets:List[str] = [],
              output_folder:str=OUTPUT_FOLDER,
              max_tokens:int=max_tokens,
              model_name_map:dict=model_name_map,              
              max_time_for_loop:int=max_time_for_loop,              
              progress=Progress(track_tqdm=True)):

    '''
    Query an LLM (local, (Gemma 2B Instruct, Gemini or Anthropic-based on AWS) with up to three prompts about a table of open text data. Up to 'batch_size' rows will be queried at a time.

    Parameters:
    - in_data_file (gr.File): Gradio file object containing input data
    - file_data (pd.DataFrame): Pandas dataframe containing the consultation response data.
    - existing_topics_table (pd.DataFrame): Pandas dataframe containing the latest master topic table that has been iterated through batches.
    - existing_reference_df (pd.DataFrame): Pandas dataframe containing the list of Response reference numbers alongside the derived topics and subtopics.
    - existing_unique_topics_df (pd.DataFrame): Pandas dataframe containing the unique list of topics, subtopics, sentiment and summaries until this point.
    - unique_table_df_display_table_markdown (str): Table for display in markdown format.
    - file_name (str): File name of the data file.
    - num_batches (int): Number of batches required to go through all the response rows.
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model.
    - chosen_cols (List[str]): A list of chosen columns to process.
    - candidate_topics (gr.FileData): A Gradio FileData object of existing candidate topics submitted by the user.
    - model_choice (str): The choice of model to use.
    - latest_batch_completed (int): The index of the latest file completed.
    - out_message (list): A list to store output messages.
    - out_file_paths (list): A list to store output file paths.
    - log_files_output_paths (list): A list to store log file output paths.
    - first_loop_state (bool): A flag indicating the first loop state.
    - whole_conversation_metadata_str (str): A string to store whole conversation metadata.
    - initial_table_prompt (str): The first prompt for the model.
    - prompt2 (str): The second prompt for the model.
    - prompt3 (str): The third prompt for the model.
    - system_prompt (str): The system prompt for the model.
    - add_existing_topics_system_prompt (str): The system prompt for the summary part of the model.
    - add_existing_topics_prompt (str): The prompt for the model summary.
    - number of requests (int): The number of prompts to send to the model.
    - batch_size (int): The number of data rows to consider in each request.
    - context_textbox (str, optional): A string giving some context to the consultation/task.
    - time_taken (float, optional): The amount of time taken to process the responses up until this point.
    - sentiment_checkbox (str, optional): What type of sentiment analysis should the topic modeller do?
    - force_zero_shot_radio (str, optional): Should responses be forced into a zero shot topic or not.
    - in_excel_sheets (List[str], optional): List of excel sheets to load from input file.
    - output_folder (str): The output folder where files will be saved.
    - max_tokens (int): The maximum number of tokens for the model.
    - model_name_map (dict, optional): A dictionary mapping full model name to shortened.
    - max_time_for_loop (int, optional): The number of seconds maximum that the function should run for before breaking (to run again, this is to avoid timeouts with some AWS services if deployed there).
    - progress (Progress): A progress tracker.
    '''

    tic = time.perf_counter()
    model = ""
    config = ""
    final_time = 0.0
    whole_conversation_metadata = []
    is_error = False
    create_revised_general_topics = False
    local_model = []
    tokenizer = []
    zero_shot_topics_df = pd.DataFrame()
    #llama_system_prefix = "<|start_header_id|>system<|end_header_id|>\n" #"<start_of_turn>user\n"
    #llama_system_suffix = "<|eot_id|>" #"<end_of_turn>\n<start_of_turn>model\n"
    #llama_cpp_prefix = "<|start_header_id|>system<|end_header_id|>\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n" #"<start_of_turn>user\n"
    #llama_cpp_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n" #"<end_of_turn>\n<start_of_turn>model\n"
    #llama_cpp_prefix = "<|user|>\n" # This is for phi 3.5
    #llama_cpp_suffix = "<|end|>\n<|assistant|>" # This is for phi 3.5
    llama_cpp_prefix = "<start_of_turn>user\n"
    llama_cpp_suffix = "<end_of_turn>\n<start_of_turn>model\n"

    # If you have a file input but no file data it hasn't yet been loaded. Load it here.
    if file_data.empty:
        print("No data table found, loading from file")
        try:
            #print("in_data_file:", in_data_file)
            in_colnames_drop, in_excel_sheets, file_name = put_columns_in_df(in_data_file)
            #print("in_colnames:", in_colnames_drop)
            file_data, file_name, num_batches = load_in_data_file(in_data_file, chosen_cols, batch_size_default, in_excel_sheets)
            #print("file_data loaded in:", file_data)
        except:
            # Check if files and text exist
            out_message = "Please enter a data file to summarise."
            print(out_message)
            raise Exception(out_message)


    #model_choice_clean = replace_punctuation_with_underscore(model_choice)
    model_choice_clean = model_name_map[model_choice]    

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("This is the first time through the loop, resetting latest_batch_completed to 0")
        if (latest_batch_completed == 999) | (latest_batch_completed == 0):
            latest_batch_completed = 0
            out_message = []
            out_file_paths = []
            #print("model_choice_clean:", model_choice_clean)

            if (model_choice == CHOSEN_LOCAL_MODEL_TYPE) & (RUN_LOCAL_MODEL == "1"):
                progress(0.1, f"Loading in local model: {CHOSEN_LOCAL_MODEL_TYPE}")
                local_model, tokenizer = load_model(local_model_type=CHOSEN_LOCAL_MODEL_TYPE, repo_id=LOCAL_REPO_ID, model_filename=LOCAL_MODEL_FILE, model_dir=LOCAL_MODEL_FOLDER)
                #print("Local model loaded:", local_model)

    if num_batches > 0:
        progress_measure = round(latest_batch_completed / num_batches, 1)
        progress(progress_measure, desc="Querying large language model")
    else:
        progress(0.1, desc="Querying large language model")

    if latest_batch_completed < num_batches:

        # Load file
        # If out message or out_file_paths are blank, change to a list so it can be appended to
        if isinstance(out_message, str):
            out_message = [out_message]

        if not out_file_paths:
            out_file_paths = []
    
        
        if model_choice == "anthropic.claude-3-sonnet-20240229-v1:0" and file_data.shape[1] > 300:
            out_message = "Your data has more than 300 rows, using the Sonnet model will be too expensive. Please choose the Haiku model instead."
            print(out_message)
            raise Exception(out_message)
            
        
        if sentiment_checkbox == "Negative, Neutral, or Positive": sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative, Neutral, or Positive"
        elif sentiment_checkbox == "Negative or Positive": sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative or Positive"
        elif sentiment_checkbox == "Do not assess sentiment": sentiment_prompt = "Create a third column containing only the text 'Not assessed'"
        else: sentiment_prompt = "In the third column, write the sentiment of the Subtopic: Negative, Neutral, or Positive"
        
        topics_loop_description = "Extracting topics from response batches (each batch of " + str(batch_size) + " responses)."
        topics_loop = tqdm(range(latest_batch_completed, num_batches), desc = topics_loop_description, unit="batches remaining")

        for i in topics_loop:       
            #for latest_batch_completed in range(num_batches):
            reported_batch_no = latest_batch_completed + 1  
            print("Running query batch", str(reported_batch_no))

            print("batch_size:", batch_size)

            # Call the function to prepare the input table
            simplified_csv_table_path, normalised_simple_markdown_table, start_row, end_row, batch_basic_response_df = data_file_to_markdown_table(file_data, file_name, chosen_cols, output_folder, latest_batch_completed, batch_size, verify_titles=True)
            #log_files_output_paths.append(simplified_csv_table_path)

            # Conversation history
            conversation_history = []

            print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

            # If the latest batch of responses contains at least one instance of text
            if not batch_basic_response_df.empty:

                # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
                if latest_batch_completed >= 1 or candidate_topics is not None:

                    # Prepare Gemini models before query       
                    if model_choice in ["gemini-2.0-flash", "gemini-1.5-pro-002"]:
                        print("Using Gemini model:", model_choice)
                        model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=add_existing_topics_system_prompt, max_tokens=max_tokens)
                    elif model_choice in ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"]:
                        print("Using AWS Bedrock model:", model_choice)
                    else:
                        print("Using local model:", model_choice)

                    
                    # Format the summary prompt with the response table and topics
                    formatted_system_prompt = add_existing_topics_system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols[0])
                    formatted_summary_prompt = add_existing_topics_prompt.format(response_table=normalised_simple_markdown_table)

                    print("formatted_summary_prompt:", formatted_summary_prompt)
                    

                    if model_choice == "gemma_2b_it_local":
                        formatted_summary_prompt = llama_cpp_prefix + formatted_system_prompt + "\n" + formatted_summary_prompt + llama_cpp_suffix
                        full_prompt = formatted_summary_prompt
                    else:
                        full_prompt = formatted_system_prompt + formatted_summary_prompt
                        
                    #latest_batch_number_string = "batch_" + str(latest_batch_completed - 1)

                    # Define the output file path for the formatted prompt
                    formatted_prompt_output_path = output_folder + file_name + "_" + str(reported_batch_no) +  "_full_prompt_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                    # Write the formatted prompt to the specified file
                    try:
                        with open(formatted_prompt_output_path, "w", encoding='utf-8', errors='replace') as f:
                            f.write(full_prompt)
                    except Exception as e:
                        print(f"Error writing prompt to file {formatted_prompt_output_path}: {e}")

                    if model_choice == "gemma_2b_it_local":
                        summary_prompt_list = [full_prompt] # Includes system prompt
                    else:
                        summary_prompt_list = [formatted_summary_prompt]


                    # print("master_summary_prompt_list:", summary_prompt_list[0])

                    summary_conversation_history = []
                    summary_whole_conversation = []

                    # Process requests to large language model
                    # responses, summary_conversation_history, whole_conversation, whole_conversation_metadata, response_text = process_requests(summary_prompt_list, add_existing_topics_system_prompt, summary_conversation_history, summary_whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, master = True)

                    responses, summary_conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(summary_prompt_list, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, MAX_OUTPUT_VALIDATION_ATTEMPTS, master = True)

                    # print("responses:", responses[-1].text)
                    # print("Whole conversation metadata:", whole_conversation_metadata)

                    topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, new_topic_df, new_markdown_table, new_reference_df, new_unique_topics_df, master_batch_out_file_part, is_error =  write_llm_output_and_logs_verify(responses, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, batch_size, chosen_cols, first_run=False)

                    # Write final output to text file for logging purposes
                    try:
                        final_table_output_path = output_folder + master_batch_out_file_part + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                        if isinstance(responses[-1], ResponseObject):
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                        elif "choices" in responses[-1]:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1]["choices"][0]['text'])
                        else:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)

                    except Exception as e:
                        print("Error in returning model response:", e)                    

                    # If error in table parsing, leave function
                    if is_error == True:
                        final_message_out = "Could not complete summary, error in LLM output."
                        raise Exception(final_message_out)
                        #return unique_table_df_display_table_markdown, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths#, final_message_out

                    # Write outputs to csv
                    ## Topics with references
                    new_topic_df.to_csv(topic_table_out_path, index=None)
                    log_files_output_paths.append(topic_table_out_path)

                    ## Reference table mapping response numbers to topics
                    new_reference_df.to_csv(reference_table_out_path, index=None)
                    out_file_paths.append(reference_table_out_path)

                    ## Unique topic list
                    new_unique_topics_df = pd.concat([new_unique_topics_df, existing_unique_topics_df]) #.drop_duplicates('Subtopic')

                    new_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
                    out_file_paths.append(unique_topics_df_out_path)
                    
                    # Outputs for markdown table output
                    unique_table_df_display_table = new_unique_topics_df.apply(lambda col: col.map(lambda x: wrap_text(x, max_text_length=500)))
                    unique_table_df_display_table_markdown = unique_table_df_display_table.to_markdown(index=False)

                    #whole_conversation_metadata.append(whole_conversation_metadata_str)
                    whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)
                    

                    #out_file_paths = [col for col in out_file_paths if latest_batch_number_string in col]
                    #log_files_output_paths = [col for col in log_files_output_paths if latest_batch_number_string in col]

                    out_file_paths = [col for col in out_file_paths if str(reported_batch_no) in col]
                    log_files_output_paths = [col for col in out_file_paths if str(reported_batch_no) in col]

                    #print("out_file_paths at end of loop:", out_file_paths)

                # If this is the first batch, run this
                else:
                    #system_prompt = system_prompt + normalised_simple_markdown_table

                    # Prepare Gemini models before query       
                    if model_choice in ["gemini-2.0-flash", "gemini-1.5-pro-002"]:
                        print("Using Gemini model:", model_choice)
                        model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
                    elif model_choice in ["gemma_2b_it_local"]:
                        print("Using local Gemma 2b model")
                    else:
                        print("Using AWS Bedrock model:", model_choice)

                    formatted_initial_table_system_prompt = system_prompt.format(consultation_context=context_textbox, column_name=chosen_cols)

                    formatted_initial_table_prompt = initial_table_prompt.format(response_table=normalised_simple_markdown_table)

                    if prompt2: formatted_prompt2 = prompt2.format(response_table=normalised_simple_markdown_table)
                    else: formatted_prompt2 = prompt2
                    
                    if prompt3: formatted_prompt3 = prompt3.format(response_table=normalised_simple_markdown_table)
                    else: formatted_prompt3 = prompt3

                    if model_choice == "gemma_2b_it_local":
                        formatted_initial_table_prompt = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_initial_table_prompt + llama_cpp_suffix
                        formatted_prompt2 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt2 + llama_cpp_suffix
                        formatted_prompt3 = llama_cpp_prefix + formatted_initial_table_system_prompt + "\n" + formatted_prompt3 + llama_cpp_suffix

                    batch_prompts = [formatted_initial_table_prompt, formatted_prompt2, formatted_prompt3][:number_of_prompts_used]  # Adjust this list to send fewer requests 
                    
                    whole_conversation = [formatted_initial_table_system_prompt] 

                    responses, conversation_history, whole_conversation, whole_conversation_metadata, response_text = call_llm_with_markdown_table_checks(batch_prompts, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, local_model, MAX_OUTPUT_VALIDATION_ATTEMPTS)


                    topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_table_df, markdown_table, reference_df, new_unique_topics_df, batch_file_path_details, is_error =  write_llm_output_and_logs_verify(responses, whole_conversation, whole_conversation_metadata, file_name, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, batch_size, chosen_cols, first_run=True)

                    # If error in table parsing, leave function
                    if is_error == True:
                        raise Exception("Error in output table parsing")
                        # unique_table_df_display_table_markdown, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths#, final_message_out
                    
                    
                    #all_topic_tables_df.append(topic_table_df)

                    topic_table_df.to_csv(topic_table_out_path, index=None)
                    out_file_paths.append(topic_table_out_path)

                    reference_df.to_csv(reference_table_out_path, index=None)
                    out_file_paths.append(reference_table_out_path)

                    ## Unique topic list

                    new_unique_topics_df = pd.concat([new_unique_topics_df, existing_unique_topics_df])

                    new_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
                    out_file_paths.append(unique_topics_df_out_path)
                    
                    #all_markdown_topic_tables.append(markdown_table)

                    whole_conversation_metadata.append(whole_conversation_metadata_str)
                    whole_conversation_metadata_str = '. '.join(whole_conversation_metadata)
                    
                    # Write final output to text file also
                    try:
                        final_table_output_path = output_folder + batch_file_path_details + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

                        if isinstance(responses[-1], ResponseObject):
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                            unique_table_df_display_table_markdown = responses[-1].text
                        elif "choices" in responses[-1]:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1]["choices"][0]['text'])
                            unique_table_df_display_table_markdown =responses[-1]["choices"][0]['text']
                        else:
                            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                                f.write(responses[-1].text)
                            unique_table_df_display_table_markdown = responses[-1].text

                        log_files_output_paths.append(final_table_output_path)

                    except Exception as e:
                        print("Error in returning model response:", e)
                    
                    new_topic_df = topic_table_df
                    new_reference_df = reference_df

            else:
                print("Current batch of responses contains no text, moving onto next. Batch number:", str(latest_batch_completed + 1), ". Start row:", start_row, ". End row:", end_row)

            # Increase latest file completed count unless we are over the last batch number
            if latest_batch_completed <= num_batches:
                print("Completed batch number:", str(reported_batch_no))
                latest_batch_completed += 1 

            toc = time.perf_counter()
            final_time = toc - tic

            if final_time > max_time_for_loop:
                print("Max time reached, breaking loop.")
                topics_loop.close()
                tqdm._instances.clear()
                break

            # Overwrite 'existing' elements to add new tables
            existing_reference_df = new_reference_df.dropna(how='all')
            existing_unique_topics_df = new_unique_topics_df.dropna(how='all')
            existing_topics_table = new_topic_df.dropna(how='all')

            # The topic table that can be modified does not need the summary column
            modifiable_unique_topics_df = existing_unique_topics_df#.drop("Summary", axis=1)

        out_time = f"{final_time:0.1f} seconds."
        
        out_message.append('All queries successfully completed in')

        final_message_out = '\n'.join(out_message)
        final_message_out = final_message_out + " " + out_time  

        print(final_message_out)

    # If we have extracted topics from the last batch, return the input out_message and file list to the relevant components
    if latest_batch_completed >= num_batches:
        print("Last batch reached, returning batch:", str(latest_batch_completed))
        # Set to a very high number so as not to mess with subsequent file processing by the user
        #latest_batch_completed = 999

        toc = time.perf_counter()
        final_time = (toc - tic) + time_taken
        out_time = f"Everything finished in {round(final_time,1)} seconds."
        print(out_time)

        print("All summaries completed. Creating outputs.")

        model_choice_clean = model_name_map[model_choice]   
        # Example usage
        in_column_cleaned = clean_column_name(chosen_cols, max_length=20)

        # Need to reduce output file names as full length files may be too long
        file_name = clean_column_name(file_name, max_length=30)    

        # Save outputs for each batch. If master file created, label file as master
        file_path_details = f"{file_name}_col_{in_column_cleaned}"

        # Create a pivoted reference table
        #existing_reference_df_pivot = convert_reference_table_to_pivot_table(existing_reference_df)

        # Save the new DataFrame to CSV
        #topic_table_out_path = output_folder + batch_file_path_details + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        #reference_table_out_pivot_path = output_folder + file_path_details + "_final_reference_table_pivot_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        reference_table_out_path = output_folder + file_path_details + "_final_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv" 
        unique_topics_df_out_path = output_folder + file_path_details + "_final_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        basic_response_data_out_path = output_folder + file_path_details + "_simplified_data_file_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

        ## Reference table mapping response numbers to topics
        existing_reference_df.to_csv(reference_table_out_path, index=None)
        out_file_paths.append(reference_table_out_path)

        # Create final unique topics table from reference table to ensure consistent numbers
        final_out_unique_topics_df = existing_unique_topics_df #create_topic_summary_df_from_reference_table(existing_reference_df)

        ## Unique topic list
        final_out_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
        out_file_paths.append(unique_topics_df_out_path)

        # Ensure that we are only returning the final results to outputs
        out_file_paths = [x for x in out_file_paths if '_final_' in x]

        ## Reference table mapping response numbers to topics
        #existing_reference_df_pivot.to_csv(reference_table_out_pivot_path, index = None)
        #log_files_output_paths.append(reference_table_out_pivot_path)

        ## Create a dataframe for missing response references:
        # Assuming existing_reference_df and file_data are already defined
        # Simplify table to just responses column and the Response reference number        

        basic_response_data = get_basic_response_data(file_data, chosen_cols, verify_titles=True)

        # Save simplified file data to log outputs
        pd.DataFrame(basic_response_data).to_csv(basic_response_data_out_path, index=None)
        log_files_output_paths.append(basic_response_data_out_path)

        # Step 1: Identify missing references
        missing_references = basic_response_data[~basic_response_data['Reference'].astype(str).isin(existing_reference_df['Response References'].astype(str).unique())]

        # Step 2: Create a new DataFrame with the same columns as existing_reference_df
        missing_df = pd.DataFrame(columns=existing_reference_df.columns)

        # Step 3: Populate the new DataFrame
        missing_df['Response References'] = missing_references['Reference']
        missing_df = missing_df.fillna(np.nan) #.infer_objects(copy=False)  # Fill other columns with NA

        # Display the new DataFrame
        #print("missing_df:", missing_df)

        missing_df_out_path = output_folder + file_path_details + "_missing_references_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
        missing_df.to_csv(missing_df_out_path, index=None)
        log_files_output_paths.append(missing_df_out_path)

        out_file_paths = list(set(out_file_paths))
        log_files_output_paths = list(set(log_files_output_paths))        

        final_out_file_paths = [file_path for file_path in out_file_paths if "final_" in file_path]
 
        # The topic table that can be modified does not need the summary column
        modifiable_unique_topics_df = final_out_unique_topics_df#.drop("Summary", axis=1)

        print("latest_batch_completed at end of batch iterations to return is", latest_batch_completed)

        return unique_table_df_display_table_markdown, existing_topics_table, final_out_unique_topics_df, existing_reference_df, final_out_file_paths, final_out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_out_file_paths, final_out_file_paths, gr.Dataframe(value=modifiable_unique_topics_df, headers=None, col_count=(modifiable_unique_topics_df.shape[1], "fixed"), row_count = (modifiable_unique_topics_df.shape[0], "fixed"), visible=True, type="pandas"), final_out_file_paths


    return unique_table_df_display_table_markdown, existing_topics_table, existing_unique_topics_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_file_paths, out_file_paths, gr.Dataframe(value=modifiable_unique_topics_df, headers=None, col_count=(modifiable_unique_topics_df.shape[1], "fixed"), row_count = (modifiable_unique_topics_df.shape[0], "fixed"), visible=True, type="pandas"), out_file_paths