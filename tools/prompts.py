system_prompt = """You are a researcher analysing responses from a public consultation. . The subject of this consultation is: {consultation_context}. You are analysing a single question from this consultation that is {column_name}."""

initial_table_prompt = """The responses from the consultation are shown in the following table that contains two columns - Reference and Response:
'{response_table}'
Based on the above table, create a markdown table to summarise the consultation responses.
In the first column identify general topics relevant to responses. Create as many general topics as you can.
In the second column list subtopics relevant to responses. Make the subtopics as specific as possible and make sure they cover every issue mentioned.
In the third column write the sentiment of the subtopic: Negative, Neutral, or Positive.
In the fourth column, write a short summary of the subtopic based on relevant responses. Highlight specific issues that appear relevant responses.
In the fifth column list the Response reference numbers of responses relevant to the Subtopic separated by commas.

Do not add any other columns. Return the table in markdown format, and don't include any special characters in the table. Do not add any other text to your response."""

prompt2 = ""

prompt3 = ""

## Adding existing topics to consultation responses

add_existing_topics_system_prompt = """You are a researcher analysing responses from a public consultation. The subject of this consultation is: {consultation_context}. You are analysing a single question from this consultation that is {column_name}."""

add_existing_topics_prompt = """Responses from a recent consultation are shown in the following table:

'{response_table}'

And below is a table of topics currently known to be relevant to this consultation:

'{topics}'

Your job is to assign responses from the Response column to existing general topics and subtopics, or to new topics if no existing topics are relevant.  
Create a new markdown table to summarise the consultation responses.
In the first and second columns, assign responses to the General Topics and Subtopics from the Topics table if they are relevant. If you cannot find a relevant topic, add new General Topics and Subtopics to the table. Make the new Subtopics as specific as possible.
In the third column, write the sentiment of the Subtopic: Negative, Neutral, or Positive.
In the fourth column, a short summary of the Subtopic based on relevant responses. Highlight specific issues that appear in relevant responses.
In the fifth column, a list of Response reference numbers relevant to the Subtopic separated by commas.

Do not add any other columns. Exclude rows for topics that are not assigned to any response. Return the table in markdown format, and do not include any special characters in the table. Do not add any other text to your response."""


summarise_topic_descriptions_system_prompt = """You are a researcher analysing responses from a public consultation."""

summarise_topic_descriptions_prompt = """Below is a table with number of paragraphs related to consultation responses:

'{summaries}'

Your job is to make a consolidated summary of the above text. Return a summary up to two paragraphs long that includes as much detail as possible from the original text. Return only the summary and no other text.

Summary:"""