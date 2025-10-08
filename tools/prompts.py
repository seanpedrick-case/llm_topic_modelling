###
# System prompt
###

generic_system_prompt = """You are a researcher analysing responses from an open text dataset. You are analysing a single column from this dataset."""

system_prompt = """You are a researcher analysing responses from an open text dataset. You are analysing a single column from this dataset called '{column_name}'. {consultation_context}"""

markdown_additional_prompt = """ You will be given a request for a markdown table. You must respond with ONLY the markdown table. Do not include any introduction, explanation, or concluding text."""

###
# Initial topic table prompt
###
initial_table_system_prompt = system_prompt + markdown_additional_prompt

initial_table_assistant_prefill = "|"

default_response_reference_format = "In the next column named 'Response References', list each specific Response reference number that is relevant to the Subtopic, separated by commas. Do no write any other text in this column."

single_response_reference_format = "In the next column named 'Placeholder', write the number 1 alongside each subtopic and no other text." # Deprecated. Instead now, no prompt is provided, and column is filled automatically with '1'

initial_table_prompt = """Your task is to create one new markdown table based on open text responses in the reponse table below.
In the first column named 'General topic', identify general topics relevant to responses. Create as many general topics as you can.
In the second column named 'Subtopic', list subtopics relevant to responses. Make the subtopics as specific as possible and make sure they cover every issue mentioned. The subtopic should never be empty.
{sentiment_choices}
{response_reference_format}
In the final column named 'Summary', write a summary of the subtopic based on relevant responses - highlight specific issues that appear. {add_existing_topics_summary_format}
Do not add any other columns. Do not add any other text to your response. Only mention topics that are relevant to at least one response.

Response table: 
{response_table}

New table:"""

# Return only one table in markdown format containing all relevant topics. Do not repeat Subtopics with the same Sentiment. 

###
# Adding existing topics to consultation responses
###

add_existing_topics_system_prompt = system_prompt + markdown_additional_prompt

add_existing_topics_assistant_prefill = "|"

force_existing_topics_prompt = """Create a new markdown table. In the first column named 'Placeholder', write 'Not assessed'. In the second column named 'Subtopics', assign Topics from the above table to Responses. Assign topics only if they are very relevant to the text of the Response. The assigned Subtopics should be chosen from the topics table above, exactly as written. Do not add any new topics, or modify existing topic names."""

allow_new_topics_prompt = """Create a new markdown table. In the first column named 'General topic', and the second column named 'Subtopic', assign General Topics and Subtopics to Responses. Assign topics from the Topics table above only if they are very relevant to the text of the Response. Fill in the General topic, Subtopic, or Sentiment for the Topic if they do not already exist. If you find a new topic that does not exist in the Topics table, add a new row to the new table. Make the General topic and Subtopic as specific as possible. The subtopic should never be blank or empty."""

force_single_topic_prompt = """ Assign each response to one single topic only."""

add_existing_topics_prompt = """Your task is to create one new markdown table, assigning responses from the Response table below to topics.
{topic_assignment}{force_single_topic}
{sentiment_choices}
{response_reference_format}
In the final column named 'Summary', write a summary of the Subtopic based on relevant responses - highlight specific issues that appear. {add_existing_topics_summary_format}
Do not add any other columns. Do not add any other text to your response. Only mention topics that are relevant to at least one response.

Responses are shown in the following Response table: 
{response_table}

Topics known to be relevant to this dataset are shown in the following Topics table: 
{topics}

New table:"""

###
# SENTIMENT CHOICES
###

negative_neutral_positive_sentiment_prompt = "In the third column named 'Sentiment', write the sentiment of the Subtopic: Negative, Neutral, or Positive"
negative_or_positive_sentiment_prompt = "In the third column named 'Sentiment', write the sentiment of the Subtopic: Negative or Positive"
do_not_assess_sentiment_prompt = "In the third column named 'Sentiment', write the text 'Not assessed'" # Not used anymore. Instead, the column is filled in automatically with 'Not assessed'
default_sentiment_prompt = "In the third column named 'Sentiment', write the sentiment of the Subtopic: Negative, Neutral, or Positive"

###
# STRUCTURE SUMMARY PROMPT
###

structured_summary_prompt = """Your task is to write a structured summary for open text responses.  

Create a new markdown table based on the response table below with the headings 'Main heading', 'Subheading' and 'Summary'.

For each of the responses in the Response table, you will create a row for each summary associated with each of the Main headings and Subheadings from the Headings table. If there is no Headings table, created your own headings. In the first and second columns, write a Main heading and Subheading from the Headings table.  Then in Summary, write a detailed and comprehensive summary that covers all information relevant to the Main heading and Subheading on the same row.
{summary_format}

Do not add any other columns. Do not add any other text to your response.

Responses are shown in the following Response table: 
{response_table}

Headings to structure the summary are in the following table: 
{topics}

New table:"""

###
# SUMMARISE TOPICS PROMPT
###

summary_assistant_prefill = ""

summarise_topic_descriptions_system_prompt = system_prompt

summarise_topic_descriptions_prompt = """Your task is to make a consolidated summary of the text below. {summary_format}. Return only the summary and no other text:

{summaries}

Summary:"""

single_para_summary_format_prompt = "Return a concise summary up to one paragraph long that summarises only the most important themes from the original text"

two_para_summary_format_prompt = "Return a summary up to two paragraphs long that includes as much detail as possible from the original text"

###
# OVERALL SUMMARY PROMPTS
###

summarise_everything_system_prompt = system_prompt

summarise_everything_prompt = """Below is a table that gives an overview of the main topics from a dataset of open text responses along with a description of each topic, and the number of responses that mentioned each topic:

'{topic_summary_table}'

Your task is to summarise the above table. {summary_format}. Return only the summary and no other text.

Summary:"""

comprehensive_summary_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary."

comprehensive_summary_format_prompt_by_group = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary. Compare and contrast differences between the topics and themes from each Group."




###
# VERIFY EXISTING DESCRIPTIONS/TITLES
###

verify_assistant_prefill = "|"

verify_titles_system_prompt = system_prompt

verify_titles_prompt = """Response numbers alongside the Response text and assigned descriptions are shown in the table below: 
{response_table}

The criteria for a suitable description for these responses is that they should be readable, concise, and fully encapsulate the main subject of the response.

Create a markdown table with four columns.
The first column is 'Response References', and should contain just the response number under consideration.
The second column is 'Is this a suitable description', answer the question with 'Yes' or 'No', with no other text.
The third column is 'Explanation', give a short explanation for your response in the second column.
The fourth column is 'Alternative description', suggest an alternative description for the response that meet the criteria stated above.
Do not add any other text to your response.

Output markdown table:"""


## The following didn't work well in testing and so is not currently used

create_general_topics_system_prompt = system_prompt

create_general_topics_prompt = """Subtopics known to be relevant to this dataset are shown in the following Topics table: 
{topics}

Your task is to create a General topic name for each Subtopic. The new Topics table should have the columns 'General topic' and 'Subtopic' only. Write a 'General topic' text label relevant to the Subtopic next to it in the new table. The text label should describe the general theme of the Subtopic. Do not add any other text, thoughts, or notes to your response.

New Topics table:"""

# example_instruction_prompt_llama3 = """<|start_header_id|>system<|end_header_id|>\n
# You are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n
# Summarise the following text in less than {length} words: "{text}"\n
# Summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""

# example_instruction_prompt_phi3 = """<|user|>\n
# Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.\n
# CONTENT: {summaries}\n
# QUESTION: {question}\n
# Answer:<|end|>\n
# <|assistant|>"""

# example_instruction_prompt_gemma = """<start_of_turn>user
# Categorise the following text into only one of the following categories that seems most relevant: 'cat1', 'cat2', 'cat3', 'cat4'. Answer only with the choice of category. Do not add any other text. Do not explain your choice.
# Text: {text}<end_of_turn>
# <start_of_turn>model
# Category:"""