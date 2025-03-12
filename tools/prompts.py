system_prompt = """You are a researcher analysing responses from an open text dataset. You are analysing a single column from this dataset that is full of open text responses called '{column_name}'. The context of this analysis is '{consultation_context}'."""

initial_table_prompt = """The open text data is shown in the following table that contains two columns, Reference and Response. Response table: 
{response_table}

Your task is to create one new markdown table with the headings 'General Topic', 'Subtopic', 'Sentiment', 'Response references', and 'Summary'.
In the first column identify general topics relevant to responses. Create as many general topics as you can.
In the second column list subtopics relevant to responses. Make the subtopics as specific as possible and make sure they cover every issue mentioned. The subtopic should never be blank or empty.
{sentiment_choices}.
In the fourth column list each specific Response reference number that is relevant to the Subtopic, separated by commas. Do no write any other text in this column.
In the fifth column, write a short summary of the subtopic based on relevant responses - highlight specific issues that appear.
Do not add any other columns. Do not add any other text to your response.

New table:"""

# Return only one table in markdown format containing all relevant topics. Do not repeat Subtopics with the same Sentiment. 

prompt2 = ""

prompt3 = ""

## Adding existing topics to consultation responses

add_existing_topics_system_prompt = system_prompt

force_existing_topics_prompt = """Create a new markdown table with the headings 'Placeholder', 'Subtopics', 'Sentiment', 'Response references', and 'Summary'.
In the first column, write 'Not assessed'. In the second column, assign Subtopics from the above table to Responses. Assign topics only if they are very relevant to the text of the Response. The assigned Subtopics should be chosen from the topics table above, exactly as written. Do not add any new topics, or modify existing topic names."""

allow_new_topics_prompt = """Create a new markdown table with the headings 'General Topic', 'Subtopic', 'Sentiment', 'Response references', and 'Summary'.
In the first and second columns, assign General Topics and Subtopics to Responses. Assign topics from the Topics table above only if they are very relevant to the text of the Response. Fill in the General Topic and Sentiment for the Subtopic if they do not already exist. If you find a new topic that does not exist in the Topics table, add a new row to the new table. Make the General Topic and Subtopic as specific as possible. The subtopic should never be blank or empty."""

add_existing_topics_prompt = """Responses are shown in the following Response table: 
{response_table}

Topics known to be relevant to this dataset are shown in the following Topics table: 
{topics}

Your task is to create one new markdown table, assigning responses from the Response table to existing topics, or to create new topics if no existing topics are relevant.  
{topic_assignment}
{sentiment_choices}.
In the fourth column list each specific Response reference number that is relevant to the Subtopic, separated by commas. Do no write any other text in this column.
In the fifth column, write a short summary of the Subtopic based on relevant responses - highlight specific issues that appear.
Do not add any other columns. Do not add any other text to your response.

New table:"""

# Return only one table in markdown format containing all relevant topics. Remove topics from the table that are not assigned to any response. Do not repeat Subtopics with the same Sentiment.


summarise_topic_descriptions_system_prompt = system_prompt

summarise_topic_descriptions_prompt = """Below is a table with number of paragraphs related to the data from the open text column:

'{summaries}'

Your task is to make a consolidated summary of the above text. {summary_format}. Return only the summary and no other text.

Summary:"""


## The following didn't work well in testing and so is not currently used

create_general_topics_system_prompt = system_prompt

create_general_topics_prompt = """Subtopics known to be relevant to this dataset are shown in the following Topics table: 
{topics}

Your task is to create a General Topic name for each Subtopic. The new Topics table should have the columns 'General Topic' and 'Subtopic' only. Write a 'General Topic' text label relevant to the Subtopic next to it in the new table. The text label should describe the general theme of the Subtopic. Do not add any other text, thoughts, or notes to your response.

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