"""
Functions for viewing and filtering JSON log files containing LLM prompts and responses.
"""

import json
from typing import Dict, List, Optional, Tuple

import gradio as gr


def load_log_file_handler(log_file):
    """Handle log file upload and initialize dropdowns."""
    if log_file is None:
        return (
            [],  # log_data_state
            gr.Dropdown(choices=[]),  # log_batch_dropdown
            gr.Dropdown(choices=[]),  # log_task_type_dropdown
            gr.Dropdown(choices=[]),  # log_group_dropdown
            gr.Dropdown(choices=[]),  # log_model_choice_dropdown
            gr.Dropdown(choices=[]),  # log_validated_dropdown
            "### Prompt\n\nNo file uploaded.",  # log_prompt_markdown
            "### Response\n\nNo file uploaded.",  # log_response_markdown
        )

    file_path = log_file.name if hasattr(log_file, "name") else log_file
    log_data = load_log_file(file_path)

    if not log_data:
        return (
            [],  # log_data_state
            gr.Dropdown(choices=[]),  # log_batch_dropdown
            gr.Dropdown(choices=[]),  # log_task_type_dropdown
            gr.Dropdown(choices=[]),  # log_group_dropdown
            gr.Dropdown(choices=[]),  # log_model_choice_dropdown
            gr.Dropdown(choices=[]),  # log_validated_dropdown
            "### Prompt\n\nError: Could not load log file or file is empty.",  # log_prompt_markdown
            "### Response\n\nError: Could not load log file or file is empty.",  # log_response_markdown
        )

    # Extract unique values for all filters
    batches, task_types, groups, model_choices, validated_values = (
        extract_unique_filter_values(log_data)
    )

    # Format choices as strings for dropdown, add "All" option
    batch_choices = ["All"] + [str(b) for b in batches] if batches else ["All"]
    task_type_choices = ["All"] + task_types if task_types else ["All"]
    group_choices = ["All"] + groups if groups else ["All"]
    model_choice_choices = ["All"] + model_choices if model_choices else ["All"]
    validated_choices = ["All"] + validated_values if validated_values else ["All"]

    # Get default prompt and response (show all if no filters)
    prompt, response = get_prompt_and_response(log_data)

    # Format for markdown display
    prompt_display = f"### Prompt\n\n{prompt}"
    response_display = f"### Response\n\n{response}"

    return (
        log_data,  # log_data_state
        gr.Dropdown(choices=batch_choices, value="All"),  # log_batch_dropdown
        gr.Dropdown(choices=task_type_choices, value="All"),  # log_task_type_dropdown
        gr.Dropdown(choices=group_choices, value="All"),  # log_group_dropdown
        gr.Dropdown(
            choices=model_choice_choices, value="All"
        ),  # log_model_choice_dropdown
        gr.Dropdown(choices=validated_choices, value="All"),  # log_validated_dropdown
        prompt_display,  # log_prompt_markdown
        response_display,  # log_response_markdown
    )


def filter_log_display(log_data, batch_str, task_type, group, model_choice, validated):
    """Filter and display log entries based on all filter criteria."""
    if not log_data:
        return (
            "### Prompt\n\nNo log data available.",
            "### Response\n\nNo log data available.",
        )

    # Convert batch string to int, handle "All" option
    batch = None
    if batch_str and batch_str != "All":
        try:
            batch = int(batch_str)
        except (ValueError, TypeError):
            batch = None

    # Handle "All" option for all filters
    task_type_filter = None if task_type == "All" else task_type
    group_filter = None if group == "All" else group
    model_choice_filter = None if model_choice == "All" else model_choice
    validated_filter = None if validated == "All" else validated

    # Get filtered prompt and response
    prompt, response = get_prompt_and_response(
        log_data,
        batch,
        task_type_filter,
        group_filter,
        model_choice_filter,
        validated_filter,
    )

    # Format for markdown display
    prompt_display = f"### Prompt\n\n{prompt}"
    response_display = f"### Response\n\n{response}"

    return prompt_display, response_display


# Handle batch and task type changes
def update_log_display_on_filter(
    log_data, batch_str, task_type, group, model_choice, validated
):
    """Update display when any filter changes."""
    return filter_log_display(
        log_data, batch_str, task_type, group, model_choice, validated
    )


def load_log_file(file_path: str) -> List[Dict]:
    """
    Load a JSON log file containing LLM prompts and responses.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing log entries
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error loading log file: {e}")
        return []


def parse_batch_number(batch: str) -> int:
    """
    Parse batch number from string format.
    Handles formats like "1", "1:", "2", "2:", etc.

    Args:
        batch: Batch string (e.g., "1", "1:", "2")

    Returns:
        Integer batch number
    """
    if not batch:
        return 0
    # Remove colon if present and strip whitespace
    batch_clean = batch.strip().rstrip(":")
    try:
        return int(batch_clean)
    except (ValueError, TypeError):
        return 0


def extract_unique_batches_and_task_types(
    log_data: List[Dict],
) -> Tuple[List[int], List[str]]:
    """
    Extract unique batch numbers and task types from log data.

    Args:
        log_data: List of log entry dictionaries

    Returns:
        Tuple of (sorted list of unique batch numbers, sorted list of unique task types)
    """
    batches = set()
    task_types = set()

    for entry in log_data:
        if "batch" in entry:
            batch_num = parse_batch_number(str(entry["batch"]))
            if batch_num > 0:
                batches.add(batch_num)
        if "task_type" in entry:
            task_type = entry["task_type"]
            if task_type:
                task_types.add(task_type)

    return sorted(list(batches)), sorted(list(task_types))


def extract_unique_filter_values(
    log_data: List[Dict],
) -> Tuple[List[int], List[str], List[str], List[str], List[str]]:
    """
    Extract unique values for all filter fields from log data.

    Args:
        log_data: List of log entry dictionaries

    Returns:
        Tuple of (batches, task_types, groups, model_choices, validated_values)
    """
    batches = set()
    task_types = set()
    groups = set()
    model_choices = set()
    validated_values = set()

    for entry in log_data:
        if "batch" in entry:
            batch_num = parse_batch_number(str(entry["batch"]))
            if batch_num > 0:
                batches.add(batch_num)
        if "task_type" in entry:
            task_type = entry["task_type"]
            if task_type:
                task_types.add(task_type)
        if "group" in entry:
            group = entry["group"]
            if group:
                groups.add(str(group))
        if "model_choice" in entry:
            model_choice = entry["model_choice"]
            if model_choice:
                model_choices.add(str(model_choice))
        if "validated" in entry:
            validated = entry["validated"]
            if validated:
                validated_values.add(str(validated))

    return (
        sorted(list(batches)),
        sorted(list(task_types)),
        sorted(list(groups)),
        sorted(list(model_choices)),
        sorted(list(validated_values)),
    )


def filter_log_entries(
    log_data: List[Dict],
    batch: Optional[int] = None,
    task_type: Optional[str] = None,
    group: Optional[str] = None,
    model_choice: Optional[str] = None,
    validated: Optional[str] = None,
) -> List[Dict]:
    """
    Filter log entries by batch number, task type, group, model_choice, and/or validated.

    Args:
        log_data: List of log entry dictionaries
        batch: Optional batch number to filter by
        task_type: Optional task type to filter by
        group: Optional group to filter by
        model_choice: Optional model choice to filter by
        validated: Optional validated value to filter by

    Returns:
        Filtered list of log entries
    """
    filtered = []

    for entry in log_data:
        match_batch = True
        match_task_type = True
        match_group = True
        match_model_choice = True
        match_validated = True

        if batch is not None:
            entry_batch = parse_batch_number(str(entry.get("batch", "0")))
            match_batch = entry_batch == batch

        if task_type is not None:
            entry_task_type = entry.get("task_type", "")
            match_task_type = entry_task_type == task_type

        if group is not None:
            entry_group = str(entry.get("group", ""))
            match_group = entry_group == group

        if model_choice is not None:
            entry_model_choice = str(entry.get("model_choice", ""))
            match_model_choice = entry_model_choice == model_choice

        if validated is not None:
            entry_validated = str(entry.get("validated", ""))
            match_validated = entry_validated == validated

        if (
            match_batch
            and match_task_type
            and match_group
            and match_model_choice
            and match_validated
        ):
            filtered.append(entry)

    return filtered


def get_prompt_and_response(
    log_data: List[Dict],
    batch: Optional[int] = None,
    task_type: Optional[str] = None,
    group: Optional[str] = None,
    model_choice: Optional[str] = None,
    validated: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Get prompt and response text for filtered log entries.
    If multiple entries match, concatenate them.

    Args:
        log_data: List of log entry dictionaries
        batch: Optional batch number to filter by
        task_type: Optional task type to filter by
        group: Optional group to filter by
        model_choice: Optional model choice to filter by
        validated: Optional validated value to filter by

    Returns:
        Tuple of (prompt_text, response_text)
    """
    filtered = filter_log_entries(
        log_data, batch, task_type, group, model_choice, validated
    )

    if not filtered:
        return "No entries found for the selected batch and task type.", ""

    # If multiple entries, combine them with separators
    prompts = []
    responses = []

    for idx, entry in enumerate(filtered, 1):
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")

        # Add entry number if multiple entries
        entry_header = f"**Entry {idx}**" if len(filtered) > 1 else ""

        if prompt:
            if entry_header:
                prompts.append(f"{entry_header}\n\n{prompt}")
            else:
                prompts.append(prompt)
        if response:
            if entry_header:
                responses.append(f"{entry_header}\n\n{response}")
            else:
                responses.append(response)

    prompt_text = "\n\n---\n\n".join(prompts) if prompts else "No prompt found."
    response_text = "\n\n---\n\n".join(responses) if responses else "No response found."

    return prompt_text, response_text


def load_and_initialize_log_viewer(
    file_path: str,
) -> Tuple[List[int], List[str], str, str]:
    """
    Load log file and initialize viewer with default values.

    Args:
        file_path: Path to the JSON log file

    Returns:
        Tuple of (batch_numbers, task_types, default_prompt, default_response)
    """
    log_data = load_log_file(file_path)

    if not log_data:
        return [], [], "No log data found in file.", ""

    batches, task_types = extract_unique_batches_and_task_types(log_data)

    # Get default prompt and response (first entry or all if no filters)
    prompt, response = get_prompt_and_response(log_data)

    return batches, task_types, prompt, response
