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

default_response_reference_format = "In the next column named 'Response References', list each specific Response reference number that is relevant to the Subtopic, separated by commas. Do not write any other text in this column."

initial_table_prompt = """{validate_prompt_prefix}Your task is to create one new markdown table based on open text responses in the reponse table below.
In the first column named 'General topic', identify general topics relevant to responses. Create as many general topics as you can.
In the second column named 'Subtopic', list subtopics relevant to responses. Make the subtopics as specific as possible and make sure they cover every issue mentioned. The subtopic should never be empty.
{sentiment_choices}{response_reference_format}
In the final column named 'Summary', write a summary of the subtopic based on relevant responses - highlight specific issues that appear. {add_existing_topics_summary_format}
Do not add any other columns. Do not add any other text to your response. Only mention topics that are relevant to at least one response.

Response table: 
{response_table}

New table:{previous_table_introduction}{previous_table}{validate_prompt_suffix}"""

###
# Adding existing topics to consultation responses
###

add_existing_topics_system_prompt = system_prompt + markdown_additional_prompt

add_existing_topics_assistant_prefill = "|"

force_existing_topics_prompt = """Create a new markdown table. In the first column named 'Placeholder', write 'Not assessed'. In the second column named 'Subtopics', assign Topics from the above table to Responses. Assign topics only if they are very relevant to the text of the Response. The assigned Subtopics should be chosen from the topics table above, exactly as written. Do not add any new topics, or modify existing topic names."""

allow_new_topics_prompt = """Create a new markdown table. In the first column named 'General topic', and the second column named 'Subtopic', assign General Topics and Subtopics to Responses. Assign topics from the Topics table above only if they are very relevant to the text of the Response. Fill in the General topic and Subtopic for the Topic if they do not already exist. If you find a new topic that does not exist in the Topics table, add a new row to the new table. Make the General topic and Subtopic as specific as possible. The subtopic should never be blank or empty."""

force_single_topic_prompt = """ Assign each response to one single topic only."""

add_existing_topics_prompt = """{validate_prompt_prefix}Your task is to create one new markdown table, assigning responses from the Response table below to topics.
{topic_assignment}{force_single_topic}
{sentiment_choices}{response_reference_format}
In the final column named 'Summary', write a summary of the Subtopic based on relevant responses - highlight specific issues that appear. {add_existing_topics_summary_format}
Do not add any other columns. Do not add any other text to your response. Only mention topics that are relevant to at least one response.

Choose from among the following topic names to assign to the responses, only if they are directlyrelevant to responses from the response table below: 
{topics}

{response_table}

New table:{previous_table_introduction}{previous_table}{validate_prompt_suffix}"""

###
# VALIDATION PROMPTS
###
# These are prompts used to validate previous LLM outputs, and create corrected versions of the outputs if errors are found.
validation_system_prompt = system_prompt

validation_prompt_prefix_default = """The following instructions were previously provided to create an output table:\n'"""

previous_table_introduction_default = """'\n\nThe following output table was created based on the above instructions:\n"""

validation_prompt_suffix_default = """\n\nBased on the above information, you need to create a corrected version of the output table. Examples of issues to correct include:

- Remove rows where responses are not relevant to the assigned topic, or where responses are not relevant to any topic.
- Remove rows where a topic is not assigned to any specific response.
- If the current topic assignment does not cover all information in a response, assign responses to relevant topics from the suggested topics table, or create a new topic if necessary.
- Correct any false information in the summary column, which is a summary of the relevant response text.
{additional_validation_issues}
- Any other obvious errors that you can identify.

With the above issues in mind, create a new, corrected version of the markdown table below. If there are no issues to correct, write simply "No change". Return only the corrected table without additional text, or 'no change' alone."""

validation_prompt_suffix_struct_summary_default = """\n\nBased on the above information, you need to create a corrected version of the output table. Examples of issues to correct include:

- Any misspellings in the Main heading or Subheading columns
- Correct any false information in the summary column, which is a summary of the relevant response text.
{additional_validation_issues}
- Any other obvious errors that you can identify.

With the above issues in mind, create a new, corrected version of the markdown table below. If there are no issues to correct, write simply "No change". Return only the corrected table without additional text, or 'no change' alone."""

###
# SENTIMENT CHOICES
###

negative_neutral_positive_sentiment_prompt = "write the sentiment of the Subtopic: Negative, Neutral, or Positive"
negative_or_positive_sentiment_prompt = "write the sentiment of the Subtopic: Negative or Positive"
do_not_assess_sentiment_prompt = "write the text 'Not assessed'" # Not used anymore. Instead, the column is filled in automatically with 'Not assessed'
default_sentiment_prompt = "write the sentiment of the Subtopic: Negative, Neutral, or Positive"

###
# STRUCTURED SUMMARY PROMPT
###

structured_summary_prompt = """Your task is to write a structured summary for open text responses.  

Create a new markdown table based on the response table below with the headings 'Main heading', 'Subheading' and 'Summary'.

For each of the responses in the Response table, you will create a row for each summary associated with each of the Main headings and Subheadings from the Headings table. If there is no Headings table, created your own headings. In the first and second columns, write a Main heading and Subheading from the Headings table.  Then in Summary, write a detailed and comprehensive summary that covers all information relevant to the Main heading and Subheading on the same row.
{summary_format}

Do not add any other columns. Do not add any other text to your response.

{response_table}

Headings to structure the summary are in the following table: 
{topics}

New table:"""

###
# SUMMARISE TOPICS PROMPT
###

summary_assistant_prefill = ""

summarise_topic_descriptions_system_prompt = system_prompt

summarise_topic_descriptions_prompt = """Your task is to make a consolidated summary of the text below. {summary_format}

Return only the summary and no other text:

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

comprehensive_summary_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary. Format the output for Excel display using: **bold text** for main headings, • bullet points for sub-items, and line breaks between sections. Avoid markdown symbols like # or ##."

comprehensive_summary_format_prompt_by_group = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary. Compare and contrast differences between the topics and themes from each Group. Format the output for Excel display using: **bold text** for main headings, • bullet points for sub-items, and line breaks between sections. Avoid markdown symbols like # or ##."

# Alternative Excel formatting options
excel_rich_text_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary. Format for Excel using: BOLD for main headings, bullet points (•) for sub-items, and line breaks between sections. Use simple text formatting that Excel can interpret."

excel_plain_text_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the table. Structure the summary with General Topics as headings, with significant Subtopics described in bullet points below them in order of relative significance. Do not explicitly mention the Sentiment, Number of responses, or Group values. Do not use the words 'General topic' or 'Subtopic' directly in the summary. Format as plain text with clear structure: use ALL CAPS for main headings, bullet points (•) for sub-items, and line breaks between sections. Avoid any special formatting symbols."

###
# LLM-BASED TOPIC DEDUPLICATION PROMPTS
###

llm_deduplication_system_prompt = """You are an expert at analysing and consolidating topic categories. Your task is to identify semantically similar topics that should be merged together, even if they use different wording or synonyms."""

llm_deduplication_prompt = """You are given a table of topics with their General topics, Subtopics, and Sentiment classifications. Your task is to identify topics that are semantically similar and should be merged together. Only merge topics that are almost identical in terms of meaning - if in doubt, do not merge.

Analyse the following topics table and identify groups of topics that describe essentially the same concept but may use different words or phrases. For example:
- "Transportation issues" and "Public transport problems" 
- "Housing costs" and "Rent prices"
- "Environmental concerns" and "Green issues"

Create a markdown table with the following columns:
1. 'Original General topic' - The current general topic name
2. 'Original Subtopic' - The current subtopic name  
3. 'Original Sentiment' - The current sentiment
4. 'Merged General topic' - The consolidated general topic name (use the most descriptive)
5. 'Merged Subtopic' - The consolidated subtopic name (use the most descriptive)
6. 'Merged Sentiment' - The consolidated sentiment (use 'Mixed' if sentiments differ)
7. 'Merge Reason' - Brief explanation of why these topics should be merged

Only include rows where topics should actually be merged. If a topic has no semantic duplicates, do not include it in the table. Produce only a markdown table in the format described above. Do not add any other text to your response.

Topics to analyse:
{topics_table}

Merged topics table:"""

llm_deduplication_prompt_with_candidates = """You are given a table of topics with their General topics, Subtopics, and Sentiment classifications. Your task is to identify topics that are semantically similar and should be merged together, even if they use different wording.

Additionally, you have been provided with a list of candidate topics that represent preferred topic categories. When merging topics, prioritise fitting similar topics into these existing candidate categories rather than creating new ones. Only merge topics that are almost identical in terms of meaning - if in doubt, do not merge.

Analyse the following topics table and identify groups of topics that describe essentially the same concept but may use different words or phrases. For example:
- "Transportation issues" and "Public transport problems" 
- "Housing costs" and "Rent prices"
- "Environmental concerns" and "Green issues"

When merging topics, consider the candidate topics provided below and try to map similar topics to these preferred categories when possible.

Create a markdown table with the following columns:
1. 'Original General topic' - The current general topic name
2. 'Original Subtopic' - The current subtopic name  
3. 'Original Sentiment' - The current sentiment
4. 'Merged General topic' - The consolidated general topic name (prefer candidate topics when similar)
5. 'Merged Subtopic' - The consolidated subtopic name (prefer candidate topics when similar)
6. 'Merged Sentiment' - The consolidated sentiment (use 'Mixed' if sentiments differ)
7. 'Merge Reason' - Brief explanation of why these topics should be merged

Only include rows where topics should actually be merged. If a topic has no semantic duplicates, do not include it in the table. Produce only a markdown table in the format described above. Do not add any other text to your response.

Topics to analyse:
{topics_table}

Candidate topics to consider for mapping:
{candidate_topics_table}

Merged topics table:"""

###
# VERIFY EXISTING DESCRIPTIONS/TITLES - Currently not used
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

