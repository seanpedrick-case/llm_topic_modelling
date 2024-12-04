---
title: Large language model topic modeller
emoji: 📝
colorFrom: purple
colorTo: yellow
sdk: 5.6.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# Large language model topic modelling

Extract topics and summarise outputs using Large Language Models (LLMs, Gemini Flash/Pro, or Claude 3 through AWS Bedrock if running on AWS). The app will query the LLM with batches of responses to produce summary tables, which are then compared iteratively to output a table with the general topics, subtopics, topic sentiment, and relevant text rows related to them. The prompts are designed for topic modelling public consultations, but they can be adapted to different contexts (see the LLM settings tab to modify).

You can use an AWS Bedrock model (Claude 3, paid), or Gemini (a free API, but with strict limits for the Pro model). Due to the strict API limits for the best model (Pro 1.5), the use of Gemini requires an API key. To set up your own Gemini API key, go here: https://aistudio.google.com/app/u/1/plan_information.

NOTE: that **API calls to Gemini are not considered secure**, so please only submit redacted, non-sensitive tabular files to this source. AWS Bedrock API calls are considered to be secure.

Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this app **absolutely need to be checked by a human** to check for harmful outputs, hallucinations, and accuracy.