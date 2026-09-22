import os
import re
from datetime import date, datetime
from typing import List, Optional, Union

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from tools.config import (
    ALL_IN_ONE_USE_LLM_DEDUP,
    ENABLE_ORIGINAL_DATA_REDACTION,
    ENABLE_VALIDATION,
    EXPORT_FORMAT,
    INCLUDE_RESPONSE_LEVEL_SUMMARY,
    OUTPUT_FOLDER,
)
from tools.config import model_name_map as global_model_name_map
from tools.helper_functions import (
    clean_column_name,
    convert_reference_table_to_pivot_table,
    ensure_model_in_map,
    ensure_safe_output_folder,
    get_basic_response_data,
    load_in_data_file,
    read_file,
    write_candidate_topics_csv,
)


def markdown_to_richtext(
    text: Union[str, float, int, None],
) -> Union[CellRichText, str, float, int, None]:
    """
    Convert markdown formatting in text to Excel RichText formatting.

    Supports:
    - **text** or __text__ for bold
    - *text* or _text_ for italic (when not bold)
    - ***text*** or ___text___ for bold+italic

    Also removes HTML <br> tags (and variants like <br/> and <br />).

    Args:
        text: The text to convert (can be string, number, or None)

    Returns:
        CellRichText object if markdown is found, otherwise returns the original value
    """
    # Return non-string values as-is
    if not isinstance(text, str):
        return text

    # Remove <br> tags (case-insensitive, handles <br>, <br/>, <br />)
    text = re.sub(r"<br\s*/?>", "", text, flags=re.IGNORECASE)

    # Check if text contains markdown formatting
    if not re.search(r"(\*\*|__|\*|_)(?=\S)", text):
        return text

    # Create RichText object
    rich_text = CellRichText()

    # Process in order: triple markers first, then double, then single
    # This prevents conflicts (e.g., ***text*** being matched as **text** + *text*)
    # Pattern order matters: longer patterns first
    # Use word boundaries to ensure markers are not part of words (e.g., filenames with underscores)
    # (?<!\w) = not preceded by word character, (?!\w) = not followed by word character
    patterns = [
        (r"(?<!\w)\*\*\*([^*]+?)\*\*\*(?!\w)", True, True),  # ***bold+italic***
        (r"(?<!\w)___([^_]+?)___(?!\w)", True, True),  # ___bold+italic___
        (r"(?<!\w)\*\*([^*]+?)\*\*(?!\w)", True, False),  # **bold**
        (r"(?<!\w)__([^_]+?)__(?!\w)", True, False),  # __bold__
        (r"(?<!\w)\*([^*]+?)\*(?!\w)", False, True),  # *italic*
        (r"(?<!\w)_([^_]+?)_(?!\w)", False, True),  # _italic_
    ]

    # Track which parts of the string have been processed
    processed = [False] * len(text)

    # Find all matches with their positions, processing longer patterns first
    all_matches = []
    for pattern, is_bold, is_italic in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            # Check if this region overlaps with already processed area
            # Allow if it's completely within unprocessed area
            if start < len(processed) and end <= len(processed):
                if not any(processed[i] for i in range(start, end)):
                    all_matches.append((start, end, match.group(1), is_bold, is_italic))
                    # Mark as processed
                    for i in range(start, end):
                        if i < len(processed):
                            processed[i] = True

    # Sort matches by position
    all_matches.sort(key=lambda x: x[0])

    last_pos = 0

    for start, end, content, is_bold, is_italic in all_matches:
        # Add plain text before the match
        if start > last_pos:
            plain_text = text[last_pos:start]
            if plain_text:
                rich_text.append(plain_text)

        # Create font for this segment (use InlineFont for RichText)
        font = InlineFont(b=is_bold, i=is_italic)
        rich_text.append(TextBlock(font, content))

        last_pos = end

    # Add remaining plain text
    if last_pos < len(text):
        remaining = text[last_pos:]
        if remaining:
            rich_text.append(remaining)

    # If we didn't add anything, return original text
    if len(rich_text) == 0:
        return text

    return rich_text


def _resolve_output_path(candidate_path: str, allowed_root: str = OUTPUT_FOLDER) -> str:
    """
    Resolve and validate that a path is contained within allowed_root.
    """
    safe_root = os.path.realpath(os.path.abspath(allowed_root))
    resolved_path = os.path.realpath(os.path.abspath(candidate_path))
    try:
        common = os.path.commonpath([safe_root, resolved_path])
    except ValueError:
        raise ValueError(
            f"Path '{candidate_path}' is outside allowed output folder"
        ) from None
    if common != safe_root:
        raise ValueError(f"Path '{candidate_path}' is outside allowed output folder")
    return resolved_path


def _remove_temp_for_xlsx_csv(csv_file: str, allowed_root: str = OUTPUT_FOLDER) -> None:
    """
    Delete an intermediate '*_for_xlsx.csv' under allowed_root.

    Rebuilds the path from the allowlisted root + basename only so the raw
    caller/output_folder-derived string is never used in exists/remove (CodeQL).
    """
    safe_name = os.path.basename(str(csv_file or "").strip())
    if (
        not safe_name
        or safe_name in {".", ".."}
        or not safe_name.endswith("_for_xlsx.csv")
    ):
        print(f"Skipping unexpected cleanup path: {csv_file}")
        return

    try:
        root_real = os.path.realpath(os.path.abspath(str(allowed_root)))
    except OSError as exc:
        raise ValueError(f"Invalid allowed root '{allowed_root}'") from exc

    search_dirs = [root_real]
    try:
        for entry in os.listdir(root_real):
            sub = os.path.join(root_real, entry)
            if os.path.isdir(sub):
                search_dirs.append(sub)
    except OSError:
        pass

    for directory in search_dirs:
        candidate = os.path.join(directory, safe_name)
        try:
            candidate_real = os.path.realpath(candidate)
            if os.path.commonpath([root_real, candidate_real]) != root_real:
                continue
            if os.path.isfile(candidate_real):
                os.remove(candidate_real)
                return
        except (OSError, ValueError):
            continue


def convert_xlsx_to_ods(
    xlsx_path: str, ods_path: str, allowed_root: str = OUTPUT_FOLDER
):
    """
    Convert an Excel (.xlsx) file to OpenDocument Spreadsheet (.ods) format.

    Args:
        xlsx_path (str): Path to the source Excel file
        ods_path (str): Path where the ODS file should be saved
    """
    try:
        import pyexcel

        safe_xlsx_path = _resolve_output_path(xlsx_path, allowed_root)
        safe_ods_path = _resolve_output_path(ods_path, allowed_root)

        pyexcel.save_as(file_name=safe_xlsx_path, dest_file_name=safe_ods_path)
        print(f"Output ods summary saved as '{safe_ods_path}'")

        if os.path.exists(safe_xlsx_path):
            os.remove(safe_xlsx_path)

        return safe_ods_path
    except ImportError:
        print(
            "Warning: pyexcel or pyexcel_ods not installed. Install with: "
            "pip install pyexcel pyexcel-ods\n"
            "Keeping output as xlsx format instead."
        )
        return xlsx_path
    except Exception as e:
        print(
            f"Warning: Could not convert xlsx to ods due to: {e}\n"
            "Keeping output as xlsx format instead."
        )
        return xlsx_path


def _resolve_candidate_topics_file_name(candidate_topics) -> str:
    """Return basename of the suggested topics file, or empty if not provided."""
    if candidate_topics is None:
        return ""

    if isinstance(candidate_topics, list):
        if not candidate_topics:
            return ""
        candidate_topics = candidate_topics[0]

    if isinstance(candidate_topics, str):
        path = candidate_topics.strip()
        if not path:
            return ""
        return os.path.basename(path)

    path = getattr(candidate_topics, "name", None)
    if path:
        return os.path.basename(str(path))

    return ""


def _coerce_usage_number(value, cast=int, default=0):
    """Coerce Gradio/CLI usage values to int or float; return default on failure."""
    if value is None or value == "":
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return cast(value)
    except (TypeError, ValueError):
        return default


def _resolve_llm_usage_stats(
    usage_logs_location: str,
    reference_data_file_name_textbox: str,
    model_choice: str,
    chosen_cols,
    llm_call_number=None,
    input_tokens=None,
    output_tokens=None,
    time_taken=None,
) -> tuple[int, int, int, float]:
    """
    Resolve LLM usage stats for the cover sheet.

    Prefer in-memory values from the current run when they indicate real usage.
    Fall back to filtering the usage logs CSV when direct values are absent/zero.
    """
    direct_calls = _coerce_usage_number(llm_call_number, int, 0)
    direct_input = _coerce_usage_number(input_tokens, int, 0)
    direct_output = _coerce_usage_number(output_tokens, int, 0)
    direct_time = _coerce_usage_number(time_taken, float, 0.0)

    if direct_calls or direct_input or direct_output or direct_time:
        return direct_calls, direct_input, direct_output, direct_time

    if not usage_logs_location:
        print("LLM call logs location not provided")
        return 0, 0, 0, 0.0

    try:
        usage_logs = pd.read_csv(usage_logs_location)
        column_for_filter = (
            chosen_cols[0]
            if isinstance(chosen_cols, list) and chosen_cols
            else chosen_cols
        )
        relevant_logs = usage_logs.loc[
            (
                usage_logs["Response ID data file name"]
                == reference_data_file_name_textbox
            )
            & (
                usage_logs[
                    "Large language model for topic extraction and summarisation"
                ]
                == model_choice
            )
            & (
                usage_logs[
                    "Select the open text column of interest. In an Excel file, this shows columns across all sheets."
                ]
                == column_for_filter
            ),
            :,
        ]
        return (
            int(sum(relevant_logs["Total LLM calls"].astype(int))),
            int(sum(relevant_logs["Total input tokens"].astype(int))),
            int(sum(relevant_logs["Total output tokens"].astype(int))),
            float(sum(relevant_logs["Estimated time taken (seconds)"].astype(float))),
        )
    except Exception as e:
        print("Could not obtain usage logs due to:", e)
        return 0, 0, 0, 0.0


def _resolve_excel_sheet_display_name(
    file_path: str, excel_sheets: Union[str, List[str], None]
) -> str:
    """Return the Excel sheet name for cover-sheet display, or empty if not applicable."""
    if os.path.splitext(file_path)[1].lower() != ".xlsx":
        return ""

    if excel_sheets:
        if isinstance(excel_sheets, list):
            sheet_names = [str(s).strip() for s in excel_sheets if str(s).strip()]
            if sheet_names:
                return ", ".join(sheet_names)
        elif str(excel_sheets).strip():
            return str(excel_sheets).strip()

    try:
        return pd.ExcelFile(file_path).sheet_names[0]
    except Exception:
        return ""


def _yes_no_label(value) -> str:
    """Normalise common truthy/falsey values to Yes/No for cover-sheet display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text in ("True", "1", "true", "TRUE", "Yes", "yes"):
        return "Yes"
    if text in ("False", "0", "false", "FALSE", "No", "no"):
        return "No"
    return text


def _build_run_settings_metadata(
    temperature=None,
    batch_size=None,
    force_zero_shot: str = "",
    force_single_topic: str = "",
    structured_summaries=False,
    sentiment_analysis: str = "",
    validation_used: str = "",
    llm_deduplication_used: str = "",
) -> dict:
    """Build ordered cover-sheet metadata describing analysis run settings."""
    metadata = {}

    if temperature is not None and str(temperature).strip() != "":
        try:
            metadata["Model temperature"] = float(temperature)
        except (TypeError, ValueError):
            metadata["Model temperature"] = temperature

    if batch_size is not None and str(batch_size).strip() != "":
        try:
            metadata["Batch size"] = int(batch_size)
        except (TypeError, ValueError):
            metadata["Batch size"] = batch_size

    force_zero_shot_label = _yes_no_label(force_zero_shot)
    if force_zero_shot_label:
        metadata["Force responses into suggested topics"] = force_zero_shot_label

    force_single_topic_label = _yes_no_label(force_single_topic)
    if force_single_topic_label:
        metadata["Force single topic assignment"] = force_single_topic_label

    metadata["Produce structured summary"] = _yes_no_label(structured_summaries) or (
        "Yes" if structured_summaries else "No"
    )

    if sentiment_analysis and str(sentiment_analysis).strip():
        metadata["Sentiment analysis"] = str(sentiment_analysis).strip()

    validation_label = _yes_no_label(validation_used)
    if not validation_label:
        validation_label = "Yes" if ENABLE_VALIDATION == "True" else "No"
    metadata["LLM validation used"] = validation_label

    llm_dedup_label = _yes_no_label(llm_deduplication_used)
    if not llm_dedup_label:
        llm_dedup_label = "Yes" if ALL_IN_ONE_USE_LLM_DEDUP else "No"
    metadata["LLM deduplication used"] = llm_dedup_label

    return metadata


def add_cover_sheet(
    wb: Workbook,
    intro_paragraphs: list[str],
    model_name: str,
    analysis_date: str,
    analysis_cost: str,
    number_of_responses: int,
    number_of_responses_with_text: int,
    number_of_responses_with_text_five_plus_words: int,
    llm_call_number: int,
    input_tokens: int,
    output_tokens: int,
    time_taken: float,
    file_name: str,
    column_name: str,
    number_of_responses_with_topic_assignment: int,
    excel_sheet_name: str = "",
    candidate_topics_file_name: str = "",
    custom_title: str = "Cover sheet",
    run_settings: Optional[dict] = None,
):
    ws = wb.create_sheet(title=custom_title, index=0)

    # Freeze top row
    ws.freeze_panes = "A2"

    # Write title
    ws["A1"] = "Large Language Model thematic analysis"
    ws["A1"].font = Font(size=14, bold=True)
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")

    # Add intro paragraphs
    row = 3
    for paragraph in intro_paragraphs:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        formatted_paragraph = markdown_to_richtext(paragraph)
        cell = ws.cell(row=row, column=1, value=formatted_paragraph)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 60  # Adjust height as needed
        row += 2

    # Add metadata
    meta_start = row + 1
    metadata = {
        "Date Excel file created": date.today().strftime("%Y-%m-%d"),
        "File name": file_name,
        "Column name": column_name,
    }
    if excel_sheet_name:
        metadata["Excel sheet name"] = excel_sheet_name
    if candidate_topics_file_name:
        metadata["Suggested topics file name"] = candidate_topics_file_name
    metadata.update(
        {
            "Model name": model_name,
            "Analysis date": analysis_date,
            # "Analysis cost": analysis_cost,
        }
    )
    if run_settings:
        metadata.update(run_settings)
    metadata.update(
        {
            "Number of responses": number_of_responses,
            "Number of responses with text": number_of_responses_with_text,
            "Number of responses with text five plus words": number_of_responses_with_text_five_plus_words,
            "Number of responses with at least one assigned topic": number_of_responses_with_topic_assignment,
            "Number of LLM calls": llm_call_number,
            "Total number of input tokens from LLM calls": input_tokens,
            "Total number of output tokens from LLM calls": output_tokens,
            "Total time taken for all LLM calls (seconds)": round(float(time_taken), 1),
        }
    )

    # Define which metadata fields should have number formatting with thousand separators
    number_format_fields = {
        "Model temperature": "0.00",
        "Batch size": "#,##0",
        "Number of responses": "#,##0",
        "Number of responses with text": "#,##0",
        "Number of responses with text five plus words": "#,##0",
        "Number of responses with at least one assigned topic": "#,##0",
        "Number of LLM calls": "#,##0",
        "Total number of input tokens from LLM calls": "#,##0",
        "Total number of output tokens from LLM calls": "#,##0",
        "Total time taken for all LLM calls (seconds)": "#,##0.0",
    }

    for i, (label, value) in enumerate(metadata.items()):
        row_num = meta_start + i
        ws[f"A{row_num}"] = label
        ws[f"A{row_num}"].font = Font(bold=True)

        cell = ws[f"B{row_num}"]
        # Convert markdown to RichText if applicable
        formatted_value = markdown_to_richtext(value)
        cell.value = formatted_value
        # Set left alignment for all metadata values (including numbers)
        cell.alignment = Alignment(horizontal="left", wrap_text=True)

        # Apply number formatting with thousand separators for numeric fields
        if label in number_format_fields:
            # Only apply formatting if value is not RichText (numbers can't be RichText)
            if not isinstance(cell.value, CellRichText):
                cell.number_format = number_format_fields[label]

        # Optional: Adjust column widths
        ws.column_dimensions["A"].width = 55
        ws.column_dimensions["B"].width = 75

    # Ensure first row cells are wrapped on the cover sheet
    for col_idx in range(1, ws.max_column + 1):
        header_cell = ws.cell(row=1, column=col_idx)
        header_cell.alignment = Alignment(wrap_text=True, vertical="center")


def csvs_to_excel(
    csv_files: list[str],
    output_filename: str,
    allowed_root: str = OUTPUT_FOLDER,
    sheet_names: list[str] = None,
    column_widths: dict = None,  # Dict of {sheet_name: {col_letter: width}}
    wrap_text_columns: dict = None,  # Dict of {sheet_name: [col_letters]}
    intro_text: list[str] = None,
    model_name: str = "",
    analysis_date: str = "",
    analysis_cost: str = "",
    llm_call_number: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    time_taken: float = 0,
    number_of_responses: int = 0,
    number_of_responses_with_text: int = 0,
    number_of_responses_with_text_five_plus_words: int = 0,
    column_name: str = "",
    number_of_responses_with_topic_assignment: int = 0,
    file_name: str = "",
    excel_sheet_name: str = "",
    candidate_topics_file_name: str = "",
    unique_reference_numbers: list = [],
    run_settings: Optional[dict] = None,
):
    if intro_text is None:
        intro_text = list()

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    def _read_csv_with_fallback_encodings(path: str) -> pd.DataFrame:
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None
        for enc in encodings_to_try:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception:
                raise

        # Final fallback: replacement characters so XLSX generation can proceed
        try:
            return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        except Exception:
            if last_err is not None:
                raise last_err
            raise

    for idx, csv_path in enumerate(csv_files):
        # Use provided sheet name or derive from file name
        sheet_name = (
            sheet_names[idx]
            if sheet_names and idx < len(sheet_names)
            else os.path.splitext(os.path.basename(csv_path))[0]
        )
        df = _read_csv_with_fallback_encodings(csv_path)

        if sheet_name == "Original data":
            try:
                # Create a copy to avoid modifying the original
                df_copy = df.copy()
                # Insert the Response ID column at position 0 (first column)
                df_copy.insert(0, "Response ID", unique_reference_numbers)
                df = df_copy
            except Exception as e:
                print("Could not add reference number to original data due to:", e)

        ws = wb.create_sheet(title=sheet_name)

        for r_idx, row in enumerate(
            dataframe_to_rows(df, index=False, header=True), start=1
        ):
            ws.append(row)

            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=r_idx, column=col_idx)

                # Convert markdown to RichText if applicable
                formatted_value = markdown_to_richtext(value)
                if formatted_value != value:
                    cell.value = formatted_value
                else:
                    cell.value = value

                # Bold header row
                if r_idx == 1:
                    # If cell already has RichText, we need to apply bold to all segments
                    if isinstance(cell.value, CellRichText):
                        # Create new RichText with bold applied to all segments
                        bold_rich_text = CellRichText()
                        for segment in cell.value:
                            if isinstance(segment, TextBlock):
                                # Preserve italic if present, add bold
                                is_italic = segment.font.i if segment.font else False
                                bold_font = InlineFont(b=True, i=is_italic)
                                bold_rich_text.append(
                                    TextBlock(bold_font, segment.text)
                                )
                            else:
                                bold_rich_text.append(
                                    TextBlock(InlineFont(b=True), str(segment))
                                )
                        cell.value = bold_rich_text
                    else:
                        cell.font = Font(bold=True)

                # Set vertical alignment to middle by default
                cell.alignment = Alignment(vertical="center")

            # Apply wrap text if needed
            if wrap_text_columns and sheet_name in wrap_text_columns:
                for col_letter in wrap_text_columns[sheet_name]:
                    cell = ws[f"{col_letter}{r_idx}"]
                    cell.alignment = Alignment(vertical="center", wrap_text=True)

        # Freeze top row for all data sheets
        ws.freeze_panes = "A2"

        # Ensure all header cells (first row) are wrapped
        for col_idx in range(1, ws.max_column + 1):
            header_cell = ws.cell(row=1, column=col_idx)
            header_cell.alignment = Alignment(vertical="center", wrap_text=True)

        # Set column widths
        if column_widths and sheet_name in column_widths:
            for col_letter, width in column_widths[sheet_name].items():
                ws.column_dimensions[col_letter].width = width

    add_cover_sheet(
        wb,
        intro_paragraphs=intro_text,
        model_name=model_name,
        analysis_date=analysis_date,
        analysis_cost=analysis_cost,
        number_of_responses=number_of_responses,
        number_of_responses_with_text=number_of_responses_with_text,
        number_of_responses_with_text_five_plus_words=number_of_responses_with_text_five_plus_words,
        llm_call_number=llm_call_number,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time_taken=time_taken,
        file_name=file_name,
        column_name=column_name,
        number_of_responses_with_topic_assignment=number_of_responses_with_topic_assignment,
        excel_sheet_name=excel_sheet_name,
        candidate_topics_file_name=candidate_topics_file_name,
        run_settings=run_settings,
    )

    wb.save(output_filename)

    # Convert to ODS format if requested
    if EXPORT_FORMAT == "ods":
        # Change file extension from .xlsx to .ods
        ods_filename = output_filename.replace(".xlsx", ".ods")
        output_filename = convert_xlsx_to_ods(
            output_filename, ods_filename, allowed_root=allowed_root
        )
    else:
        print(f"Output xlsx summary saved as '{output_filename}'")

    return output_filename


def stage_original_data_csv(
    original_data_file_path: str,
    output_folder: str,
    excel_sheets: str = "",
    redact: Optional[bool] = None,
) -> tuple:
    """Prepare the Original data sheet CSV.

    When redaction is off and the source is already a CSV, the original path is
    used. Otherwise a temp CSV is written (redacted across all cells when enabled).

    Returns:
        tuple: (csv_path, is_temp_file)
    """
    if redact is None:
        redact = ENABLE_ORIGINAL_DATA_REDACTION

    original_ext = os.path.splitext(original_data_file_path)[1].lower()
    if not redact and original_ext == ".csv":
        return original_data_file_path, False

    if original_ext not in (".csv", ".xlsx", ".parquet"):
        raise Exception(f"Unsupported file type for original data: {original_ext}")

    df = read_file(original_data_file_path, excel_sheets if excel_sheets else "")

    if redact:
        from tools.data_redaction import redact_dataframe

        df = redact_dataframe(df)
        print("Applied PII redaction to Original data tab (all columns).")

    original_data_csv_path = os.path.join(
        output_folder,
        os.path.splitext(os.path.basename(original_data_file_path))[0]
        + "_for_xlsx.csv",
    )
    df.to_csv(original_data_csv_path, index=False)
    return original_data_csv_path, True


###
# Run the functions
###
def collect_output_csvs_and_create_excel_output(
    in_data_files: List,
    chosen_cols: list[str],
    reference_data_file_name_textbox: str,
    in_group_col: str,
    model_choice: str,
    master_reference_df_state: pd.DataFrame,
    master_unique_topics_df_state: pd.DataFrame,
    summarised_output_df: pd.DataFrame,
    missing_df_state: pd.DataFrame,
    excel_sheets: str = "",
    usage_logs_location: str = "",
    model_name_map: dict = dict(),
    output_folder: str = OUTPUT_FOLDER,
    structured_summaries: str = "No",
    candidate_topics=None,
    create_topics_csv: str = "Yes",
    llm_call_number=None,
    input_tokens=None,
    output_tokens=None,
    time_taken=None,
    temperature=None,
    batch_size=None,
    force_zero_shot: str = "",
    force_single_topic: str = "",
    sentiment_analysis: str = "",
    validation_used: str = "",
    llm_deduplication_used: str = "",
):
    """
    Collect together output CSVs from various output boxes and combine them into a single output Excel file.

    Args:
        in_data_files (List): A list of paths to the input data files.
        chosen_cols (list[str]): A list of column names selected for analysis.
        reference_data_file_name_textbox (str): The name of the reference data file.
        in_group_col (str): The column used for grouping the data.
        model_choice (str): The LLM model chosen for the analysis.
        master_reference_df_state (pd.DataFrame): The master DataFrame containing reference data.
        master_unique_topics_df_state (pd.DataFrame): The master DataFrame containing unique topics data.
        summarised_output_df (pd.DataFrame): DataFrame containing the summarised output.
        missing_df_state (pd.DataFrame): DataFrame containing information about missing data.
        excel_sheets (str): Information regarding Excel sheets, typically sheet names or structure.
        usage_logs_location (str, optional): Path to the usage logs CSV file. Defaults to "".
        model_name_map (dict, optional): A dictionary mapping model choices to their display names. Defaults to {}.
        output_folder (str, optional): The directory where the output Excel file will be saved. Defaults to OUTPUT_FOLDER.
        structured_summaries (str, optional): Indicates whether structured summaries are being produced ("Yes" or "No"). Defaults to "No".
        candidate_topics (optional): Suggested topics file uploaded by the user (path string or Gradio FileData).
        create_topics_csv (str, optional): Whether to write a suggested-topics CSV with unique
            General topic / Subtopic pairs from the analysis. Defaults to "Yes".
        llm_call_number (optional): In-memory LLM call count from the current run.
        input_tokens (optional): In-memory input token count from the current run.
        output_tokens (optional): In-memory output token count from the current run.
        time_taken (optional): In-memory LLM time taken (seconds) from the current run.
        temperature (optional): LLM temperature used for the run.
        batch_size (optional): Batch size used for the run.
        force_zero_shot (str, optional): Whether responses were forced into suggested topics.
        force_single_topic (str, optional): Whether responses were forced to a single topic.
        sentiment_analysis (str, optional): Sentiment analysis option used for the run.
        validation_used (str, optional): Whether LLM validation was used. Defaults from ENABLE_VALIDATION.
        llm_deduplication_used (str, optional): Whether LLM deduplication was used.
            Defaults from ALL_IN_ONE_USE_LLM_DEDUP.

    Returns:
        tuple: A tuple containing:
            - list: Paths to generated output files (xlsx and, when requested, suggested topics CSV).
            - list: Duplicate of the first list (for UI compatibility).
    """
    # Use passed model_name_map if provided and not empty, otherwise use global one
    if not model_name_map:
        model_name_map = global_model_name_map

    # Ensure custom model_choice is registered in model_name_map
    ensure_model_in_map(model_choice, model_name_map)

    if structured_summaries == "Yes":
        structured_summaries = True
    else:
        structured_summaries = False

    run_settings = _build_run_settings_metadata(
        temperature=temperature,
        batch_size=batch_size,
        force_zero_shot=force_zero_shot,
        force_single_topic=force_single_topic,
        structured_summaries=structured_summaries,
        sentiment_analysis=sentiment_analysis,
        validation_used=validation_used,
        llm_deduplication_used=llm_deduplication_used,
    )

    if not chosen_cols:
        raise Exception("Could not find chosen column")

    # Harden client-supplied Gradio output_folder_state against path traversal.
    # Keep a trailing separator so existing `output_folder + "file.csv"` joins work.
    output_folder = ensure_safe_output_folder(output_folder, allowed_root=OUTPUT_FOLDER)
    if not output_folder.endswith(("/", "\\", os.sep)):
        output_folder = output_folder + os.sep

    today_date = datetime.today().strftime("%Y-%m-%d")
    original_data_file_path = os.path.abspath(in_data_files[0])
    excel_sheet_display_name = _resolve_excel_sheet_display_name(
        original_data_file_path, excel_sheets
    )
    candidate_topics_file_name = _resolve_candidate_topics_file_name(candidate_topics)

    csv_files = list()
    sheet_names = list()
    column_widths = dict()
    wrap_text_columns = dict()
    short_file_name = os.path.basename(reference_data_file_name_textbox)
    reference_pivot_table = pd.DataFrame()
    reference_table_csv_path = ""
    reference_pivot_table_csv_path = ""
    unique_topic_table_csv_path = ""
    missing_df_state_csv_path = ""
    overall_summary_csv_path = ""
    number_of_responses_with_topic_assignment = 0
    # Track all temporary CSV files created for xlsx conversion
    temp_csv_files_for_cleanup = list()

    if in_group_col:
        group = in_group_col
    else:
        group = "All"

    overall_summary_csv_path = output_folder + "overall_summary_for_xlsx.csv"
    temp_csv_files_for_cleanup.append(overall_summary_csv_path)

    if structured_summaries is True and not master_unique_topics_df_state.empty:
        print("Producing overall summary based on structured summaries.")
        # Create structured summary from master_unique_topics_df_state
        structured_summary_data = list()

        # Group by 'Group' column
        for group_name, group_df in master_unique_topics_df_state.groupby("Group"):
            group_summary = f"## {group_name}\n\n"

            # Group by 'General topic' within each group
            for general_topic, topic_df in group_df.groupby("General topic"):
                group_summary += f"### {general_topic}\n\n"

                # Add subtopics under each general topic
                for _, row in topic_df.iterrows():
                    subtopic = row["Subtopic"]
                    summary = row["Summary"]
                    # sentiment = row.get('Sentiment', '')
                    # num_responses = row.get('Number of responses', '')

                    # Create subtopic entry
                    subtopic_entry = f"**{subtopic}**"
                    # if sentiment:
                    #     subtopic_entry += f" ({sentiment})"
                    # if num_responses:
                    #     subtopic_entry += f" - {num_responses} responses"
                    subtopic_entry += "\n\n"

                    if summary and pd.notna(summary):
                        subtopic_entry += f"{summary}\n\n"

                    group_summary += subtopic_entry

            # Add to structured summary data
            structured_summary_data.append(
                {"Group": group_name, "Summary": group_summary.strip()}
            )

        # Create DataFrame for structured summary
        structured_summary_df = pd.DataFrame(structured_summary_data)
        structured_summary_df.to_csv(overall_summary_csv_path, index=False)
    else:
        # Use original summarised_output_df
        structured_summary_df = summarised_output_df
        structured_summary_df.to_csv(overall_summary_csv_path, index=None)

    if not structured_summary_df.empty:
        csv_files.append(overall_summary_csv_path)
        sheet_names.append("Overall summary")
        column_widths["Overall summary"] = {"A": 15, "B": 120}
        wrap_text_columns["Overall summary"] = ["A", "B"]

    # Always load response-level source data for cover-sheet stats (and for the
    # pivot when reference rows exist). Previously this only ran when
    # master_reference_df_state was non-empty, which crashed structured-summary
    # runs that produced no Response ID assignments.
    file_data, file_name, num_batches = load_in_data_file(
        in_data_files, chosen_cols, 1, in_excel_sheets=excel_sheets
    )
    basic_response_data = get_basic_response_data(
        file_data, chosen_cols, verify_titles="No"
    )
    short_file_name = os.path.basename(file_name)

    if not master_reference_df_state.empty:
        # Simplify table to just responses column and the Response reference number
        reference_pivot_table = convert_reference_table_to_pivot_table(
            master_reference_df_state, basic_response_data
        )

        unique_reference_numbers = basic_response_data["Response ID"].tolist()

        try:
            master_reference_df_state.rename(
                columns={"Topic_number": "Topic number"}, inplace=True, errors="ignore"
            )
            master_reference_df_state.drop(
                columns=["1", "2", "3"], inplace=True, errors="ignore"
            )
        except Exception as e:
            print("Could not rename Topic_number due to", e)

        preferred_reference_cols = [
            "Response ID",
            "Original Response ID",
            "General topic",
            "Subtopic",
            "Sentiment",
            "Confidence",
            "Summary",
            "Revised summary",
            "Start row of group",
            "Group",
            "Topic number",
        ]
        response_level_df = master_reference_df_state
        response_level_summary_cols = ["Summary", "Revised summary"]
        if not INCLUDE_RESPONSE_LEVEL_SUMMARY:
            preferred_reference_cols = [
                col
                for col in preferred_reference_cols
                if col not in response_level_summary_cols
            ]
            response_level_df = master_reference_df_state.drop(
                columns=response_level_summary_cols, errors="ignore"
            )
        existing_reference_cols = [
            col for col in preferred_reference_cols if col in response_level_df.columns
        ]
        remaining_reference_cols = [
            col
            for col in response_level_df.columns
            if col not in existing_reference_cols
        ]
        if existing_reference_cols:
            response_level_df = response_level_df[
                existing_reference_cols + remaining_reference_cols
            ]

        number_of_responses_with_topic_assignment = len(
            response_level_df["Response ID"].unique()
        )

        reference_table_csv_path = output_folder + "reference_df_for_xlsx.csv"
        response_level_df.to_csv(reference_table_csv_path, index=None)
        temp_csv_files_for_cleanup.append(reference_table_csv_path)

        reference_pivot_table_csv_path = (
            output_folder + "reference_pivot_df_for_xlsx.csv"
        )
        # Reorder columns to ensure 'Original Response ID' comes before 'Response'
        cols = list(reference_pivot_table.columns)
        if "Original Response ID" in cols and "Response" in cols:
            cols.remove("Original Response ID")
            cols.remove("Response")
            cols = ["Original Response ID", "Response"] + cols
            reference_pivot_table = reference_pivot_table[cols]
        reference_pivot_table.to_csv(reference_pivot_table_csv_path, index=None)
        temp_csv_files_for_cleanup.append(reference_pivot_table_csv_path)

        short_file_name = os.path.basename(file_name)

    if not master_unique_topics_df_state.empty:

        master_unique_topics_df_state.drop(
            columns=["1", "2", "3"], inplace=True, errors="ignore"
        )

        unique_topic_table_csv_path = (
            output_folder + "unique_topic_table_df_for_xlsx.csv"
        )
        master_unique_topics_df_state.to_csv(unique_topic_table_csv_path, index=None)
        temp_csv_files_for_cleanup.append(unique_topic_table_csv_path)

    if unique_topic_table_csv_path:
        csv_files.append(unique_topic_table_csv_path)
        sheet_names.append("Topic summary")
        column_widths["Topic summary"] = {
            "A": 25,
            "B": 25,
            "C": 12,
            "D": 15,
            "E": 10,
            "F": 100,
        }
        wrap_text_columns["Topic summary"] = ["A", "B", "D", "F"]
    else:
        print("Relevant unique topic files not found, excluding from xlsx output.")

    if reference_table_csv_path:
        if structured_summaries:
            print(
                "Structured summaries are being produced, excluding response level data from xlsx output."
            )
        else:
            csv_files.append(reference_table_csv_path)
            sheet_names.append("Response level data")
            has_confidence_col = "Confidence" in response_level_df.columns
            has_summary_col = any(
                col in response_level_df.columns
                for col in ("Summary", "Revised summary")
            )
            if has_confidence_col and has_summary_col:
                column_widths["Response level data"] = {
                    "A": 12,
                    "B": 30,
                    "C": 40,
                    "D": 10,
                    "E": 12,
                    "F": 12,
                    "G": 100,
                }
                wrap_text_columns["Response level data"] = ["C", "G"]
            elif has_confidence_col:
                column_widths["Response level data"] = {
                    "A": 12,
                    "B": 30,
                    "C": 40,
                    "D": 10,
                    "E": 12,
                    "F": 12,
                }
                wrap_text_columns["Response level data"] = ["C"]
            elif has_summary_col:
                column_widths["Response level data"] = {
                    "A": 12,
                    "B": 30,
                    "C": 40,
                    "D": 10,
                    "E": 10,
                    "F": 100,
                }
                wrap_text_columns["Response level data"] = ["C", "F"]
            else:
                column_widths["Response level data"] = {
                    "A": 12,
                    "B": 30,
                    "C": 40,
                    "D": 10,
                    "E": 10,
                }
                wrap_text_columns["Response level data"] = ["C"]
    else:
        print("Relevant reference files not found, excluding from xlsx output.")

    if reference_pivot_table_csv_path:
        if structured_summaries:
            print(
                "Structured summaries are being produced, excluding topic response pivot table from xlsx output."
            )
        else:
            csv_files.append(reference_pivot_table_csv_path)
            sheet_names.append("Topic response pivot table")

            if reference_pivot_table.empty:
                reference_pivot_table = pd.read_csv(reference_pivot_table_csv_path)

            # Base widths and wrap
            column_widths["Topic response pivot table"] = {"A": 12, "B": 100, "C": 12}
            wrap_text_columns["Topic response pivot table"] = ["B", "C"]

            num_cols = len(reference_pivot_table.columns)
            col_letters = [get_column_letter(i) for i in range(4, num_cols + 1)]

            for col_letter in col_letters:
                column_widths["Topic response pivot table"][col_letter] = 20

            wrap_text_columns["Topic response pivot table"].extend(col_letters)
    else:
        print(
            "Relevant reference pivot table files not found, excluding from xlsx output."
        )

    if not missing_df_state.empty:
        missing_df_state_csv_path = output_folder + "missing_df_state_df_for_xlsx.csv"
        missing_df_state.to_csv(missing_df_state_csv_path, index=None)
        temp_csv_files_for_cleanup.append(missing_df_state_csv_path)

    if missing_df_state_csv_path:
        if structured_summaries:
            print(
                "Structured summaries are being produced, excluding missing responses from xlsx output."
            )
        else:
            csv_files.append(missing_df_state_csv_path)
            sheet_names.append("Missing responses")
            column_widths["Missing responses"] = {"A": 25, "B": 30, "C": 50}
            wrap_text_columns["Missing responses"] = ["C"]
    else:
        print("Relevant missing responses files not found, excluding from xlsx output.")

    # Original data file
    original_data_csv_path, original_data_is_temp = stage_original_data_csv(
        original_data_file_path,
        output_folder,
        excel_sheets=excel_sheets,
    )
    csv_files.append(original_data_csv_path)
    if original_data_is_temp:
        temp_csv_files_for_cleanup.append(original_data_csv_path)

    sheet_names.append("Original data")
    column_widths["Original data"] = {"A": 10, "B": 20, "C": 20}
    wrap_text_columns["Original data"] = ["C"]
    if isinstance(chosen_cols, list) and chosen_cols:
        chosen_cols = chosen_cols[0]
    else:
        chosen_cols = str(chosen_cols) if chosen_cols else ""

    # Intro page text
    excel_sheet_intro = (
        f", from Excel sheet '{excel_sheet_display_name}'"
        if excel_sheet_display_name
        else ""
    )
    intro_text = [
        "This workbook contains outputs from the Large Language Model (LLM) thematic analysis of open text data. Each sheet corresponds to a different CSV report included in the analysis.",
        f"The file analysed was {short_file_name}, the column analysed was '{chosen_cols}'{excel_sheet_intro} and the data was grouped by column '{group}'."
        " Please contact the app administrator if you need any explanation on how to use the results."
        "LLMs are not 100% accurate and may produce biased or harmful outputs. All outputs from this analysis **need to be checked by a human** to check for harmful outputs, false information, and bias.",
    ]

    # Get values for number of rows, number of responses, and number of responses longer than five words
    number_of_responses = basic_response_data.shape[0]
    # number_of_responses_with_text = basic_response_data["Response"].str.strip().notnull().sum()
    number_of_responses_with_text = (
        basic_response_data["Response"].str.strip().notnull()
        & (basic_response_data["Response"].str.split().str.len() >= 1)
    ).sum()
    number_of_responses_with_text_five_plus_words = (
        basic_response_data["Response"].str.strip().notnull()
        & (basic_response_data["Response"].str.split().str.len() >= 5)
    ).sum()

    # Prefer in-memory stats from the current run; fall back to usage logs CSV
    llm_call_number, input_tokens, output_tokens, time_taken = _resolve_llm_usage_stats(
        usage_logs_location=usage_logs_location,
        reference_data_file_name_textbox=reference_data_file_name_textbox,
        model_choice=model_choice,
        chosen_cols=chosen_cols,
        llm_call_number=llm_call_number,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time_taken=time_taken,
    )

    # Create short filename:
    model_choice_clean_short = clean_column_name(
        model_name_map[model_choice]["short_name"],
        max_length=20,
        front_characters=False,
    )
    # Extract first column name as string for cleaning and Excel output
    chosen_col_str = (
        chosen_cols[0]
        if isinstance(chosen_cols, list) and chosen_cols
        else str(chosen_cols) if chosen_cols else ""
    )
    in_column_cleaned = clean_column_name(chosen_col_str, max_length=20)
    file_name_cleaned = clean_column_name(
        file_name, max_length=20, front_characters=True
    )

    # Save outputs for each batch. If master file created, label file as master
    file_path_details = (
        f"{file_name_cleaned}_col_{in_column_cleaned}_{model_choice_clean_short}"
    )
    output_xlsx_filename = (
        output_folder
        + file_path_details
        + ("_structured_summaries" if structured_summaries else "_theme_analysis")
        + ".xlsx"
    )

    xlsx_output_filename = csvs_to_excel(
        csv_files=csv_files,
        output_filename=output_xlsx_filename,
        allowed_root=OUTPUT_FOLDER,
        sheet_names=sheet_names,
        column_widths=column_widths,
        wrap_text_columns=wrap_text_columns,
        intro_text=intro_text,
        model_name=model_choice,
        analysis_date=today_date,
        analysis_cost="",
        llm_call_number=llm_call_number,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time_taken=time_taken,
        number_of_responses=number_of_responses,
        number_of_responses_with_text=number_of_responses_with_text,
        number_of_responses_with_text_five_plus_words=number_of_responses_with_text_five_plus_words,
        column_name=chosen_col_str,
        number_of_responses_with_topic_assignment=number_of_responses_with_topic_assignment,
        file_name=short_file_name,
        excel_sheet_name=excel_sheet_display_name,
        candidate_topics_file_name=candidate_topics_file_name,
        unique_reference_numbers=unique_reference_numbers,
        run_settings=run_settings,
    )

    xlsx_output_filenames = [xlsx_output_filename]
    topics_csv_filenames = []

    if create_topics_csv in (True, "Yes") and not master_unique_topics_df_state.empty:
        topics_csv_filename = (
            output_folder
            + file_path_details
            + ("_structured_summaries" if structured_summaries else "_theme_analysis")
            + "_suggested_topics.csv"
        )
        written_path = write_candidate_topics_csv(
            master_unique_topics_df_state, topics_csv_filename
        )
        if written_path:
            topics_csv_filenames.append(written_path)

    all_output_filenames = xlsx_output_filenames + topics_csv_filenames

    # Delete intermediate '_for_xlsx.csv' files (only under OUTPUT_FOLDER)
    for csv_file in temp_csv_files_for_cleanup:
        try:
            _remove_temp_for_xlsx_csv(csv_file, allowed_root=OUTPUT_FOLDER)
        except Exception as e:
            print(f"Could not delete temporary CSV file '{csv_file}' due to: {e}")

    return all_output_filenames, all_output_filenames
