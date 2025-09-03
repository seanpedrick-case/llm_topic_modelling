---
title: Large language model topic modelling
emoji: 📝
colorFrom: purple
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: true
license: agpl-3.0
---

# Large language model topic modelling

Extract topics and summarise outputs using Large Language Models (LLMs, Gemma 3 4b/GPT-OSS 20b if local (see config.py), Gemini 2.5, or Bedrock models e.g. (Claude 3 Haiku/Claude Sonnet 3.7). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and relevant text rows related to them. The prompts are designed for topic modelling public consultations, but they can be adapted to different contexts (see the LLM settings tab to modify).

Instructions on use can be found in the README.md file. Try it out with this [dummy development consultation dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/tree/main), which you can also try with [zero-shot topics](https://huggingface.co/datasets/seanpedrickcase/dummy_development_consultation/tree/main). Try also this [dummy case notes dataset](https://huggingface.co/datasets/seanpedrickcase/dummy_case_notes/tree/main).

You can use an AWS Bedrock model (paid), or Gemini (a free API for Flash). The use of Gemini requires an API key. To set up your own Gemini API key, go [here](https://aistudio.google.com/app/u/1/plan_information). 

NOTE: Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.

Basic use:
1. On the front page, choose your model for inference. Gemma 3/GPT-OSS will use 'on-device' inference. Calls to Gemini or AWS will require an API key that can be input on the 'LLM and topic extraction' page.
1. Upload a csv/xlsx/parquet file containing at least one open text column.
2. Select the relevant open text column from the dropdown.
3. If you have your own suggested (zero shot) topics, upload this (see examples folder for an example file)
4. Write a one sentence description of the consultation/context of the open text.
5. Extract topics.
6. If topic extraction fails part way through, you can upload the latest 'reference_table' and 'unique_topics_table' csv outputs on the 'Continue previous topic extraction' tab to continue from where you left off.
7. Summaries will be produced for each topic for each 'batch' of responses. If you want consolidated summaries, go to the tab 'Summarise topic outputs', upload your output reference_table and unique_topics csv files, and press summarise.