import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Font
from datetime import date, datetime
import os
from typing import List

from tools.config import OUTPUT_FOLDER
from tools.helper_functions import convert_reference_table_to_pivot_table, get_basic_response_data, load_in_data_file

def add_cover_sheet(
    wb:Workbook,
    intro_paragraphs:list[str],
    model_name:str,
    analysis_date:str,
    analysis_cost:str,
    file_name:str,
    custom_title:str="Cover Sheet"
):
    ws = wb.create_sheet(title=custom_title, index=0)

    # Write title
    ws["A1"] = "Large Language Model Topic analysis"
    ws["A1"].font = Font(size=14, bold=True)

    # Add intro paragraphs
    row = 3
    for paragraph in intro_paragraphs:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        cell = ws.cell(row=row, column=1, value=paragraph)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 50  # Adjust height as needed
        row += 2

    # Add metadata
    meta_start = row + 1
    metadata = {
        "Date Generated": date.today().strftime("%Y-%m-%d"),
        "File name": file_name,
        "Model name": model_name,
        "Analysis date": analysis_date,
        "Analysis cost": analysis_cost
    }

    for i, (label, value) in enumerate(metadata.items()):
        row_num = meta_start + i
        ws[f"A{row_num}"] = label
        ws[f"A{row_num}"].font = Font(bold=True)

        cell = ws[f"B{row_num}"]
        cell.value = value
        cell.alignment = Alignment(wrap_text=True)
        # Optional: Adjust column widths
        ws.column_dimensions["A"].width = 25
        ws.column_dimensions["B"].width = 50

def csvs_to_excel(
    csv_files:list[str],
    output_filename:str,
    sheet_names:list[str]=None,
    column_widths:dict=None,  # Dict of {sheet_name: {col_letter: width}}
    wrap_text_columns:dict=None,  # Dict of {sheet_name: [col_letters]}
    intro_text: list[str] = None,
    model_name:str="",
    analysis_date:str="",
    analysis_cost:str="",
    file_name:str=""
):
    if intro_text is None:
        intro_text = []

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    for idx, csv_path in enumerate(csv_files):
        # Use provided sheet name or derive from file name
        sheet_name = sheet_names[idx] if sheet_names and idx < len(sheet_names) else os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)

        ws = wb.create_sheet(title=sheet_name)

        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
            ws.append(row)

            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=r_idx, column=col_idx)

                # Bold header row
                if r_idx == 1:
                    cell.font = Font(bold=True)

                # Set vertical alignment to middle by default
                cell.alignment = Alignment(vertical="center")

            # Apply wrap text if needed
            if wrap_text_columns and sheet_name in wrap_text_columns:
                for col_letter in wrap_text_columns[sheet_name]:
                    cell = ws[f"{col_letter}{r_idx}"]
                    cell.alignment = Alignment(wrap_text=True)

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
        file_name=file_name
    )

    wb.save(output_filename)
    
    print(f"Workbook saved as '{output_filename}'")

    return output_filename

###
# Run the functions
###
def collect_output_csvs_and_create_excel_output(in_data_files:List, chosen_cols:list[str], reference_data_file_name_textbox:str, in_group_col:str, model_choice:str, master_reference_df_state:pd.DataFrame, master_unique_topics_df_state:pd.DataFrame, summarised_output_df:pd.DataFrame, missing_df_state:pd.DataFrame, output_folder:str=OUTPUT_FOLDER):
    '''
    Collect together output csvs from various output boxes and combine into a single output Excel file.
    '''
    print("Inside xlsx function")

    if not chosen_cols:
        raise Exception("Could not find chosen column")

    today_date = datetime.today().strftime('%Y-%m-%d')
    print("in_data_files:", in_data_files)
    original_data_file_path = os.path.abspath(in_data_files[0])
    
    csv_files = []
    sheet_names = []
    column_widths = {}
    wrap_text_columns = {}
    short_file_name = os.path.basename(reference_data_file_name_textbox)
    reference_table_csv_path = ""
    reference_pivot_table_csv_path = ""
    unique_topic_table_csv_path = ""
    missing_df_state_csv_path = ""
    overall_summary_csv_path = ""

    if in_group_col: group = in_group_col
    else: group = "All"

    print("Creating sheet list")

    if not summarised_output_df.empty:

        overall_summary_csv_path = output_folder + "overall_summary_for_xlsx.csv"
        summarised_output_df.to_csv(overall_summary_csv_path, index = None)

        #overall_summary_csv_path = [x for x in file_output_list if "overall" in x][0]
        csv_files.append(overall_summary_csv_path)
        sheet_names.append("Overall summary")
        column_widths["Overall summary"] = {"A": 20, "B": 100}
        wrap_text_columns["Overall summary"] = ['B']

    file_output_list = []

    if not master_reference_df_state.empty:
        # Simplify table to just responses column and the Response reference number
        file_data, file_name, num_batches = load_in_data_file(in_data_files, chosen_cols, 1, "")
        basic_response_data = get_basic_response_data(file_data, chosen_cols, verify_titles="No")
        reference_pivot_table = convert_reference_table_to_pivot_table(master_reference_df_state, basic_response_data)
        
        reference_table_csv_path = output_folder + "reference_df_for_xlsx.csv"
        master_reference_df_state.to_csv(reference_table_csv_path, index = None)

        reference_pivot_table_csv_path = output_folder + "reference_pivot_df_for_xlsx.csv"
        reference_pivot_table.to_csv(reference_pivot_table_csv_path, index = None)

        short_file_name = os.path.basename(file_name)

    if not master_unique_topics_df_state.empty:
        #unique_topic_table_csv_path = [x for x in file_output_list if "unique_topic" in x]
        #reference_table_csv_path = [x for x in file_output_list if "reference_table" in x]

        unique_topic_table_csv_path = output_folder + "unique_topic_table_df_for_xlsx.csv"
        master_unique_topics_df_state.to_csv(unique_topic_table_csv_path, index = None)

    if unique_topic_table_csv_path:
        #unique_topic_table_csv_path = unique_topic_table_csv_path[0]
        csv_files.append(unique_topic_table_csv_path)
        sheet_names.append("Topic summary")
        column_widths["Topic summary"] = {"A": 25, "B": 25, "C": 15, "D": 15, "F":100}
        wrap_text_columns["Topic summary"] = ["F"]
    else:
        raise Exception("Could not find unique topic files to put into Excel format")
    if reference_table_csv_path:
        #reference_table_csv_path = reference_table_csv_path[0]
        csv_files.append(reference_table_csv_path)
        sheet_names.append("Response level data")
        column_widths["Response level data"] = {"A": 15, "B": 30, "C": 40, "G":100}
        wrap_text_columns["Response level data"] = ["C", "G"]        
    else:
        raise Exception("Could not find any reference files to put into Excel format")

    #if log_files_output_paths:
    #reference_table_pivot_csv_path = [x for x in file_output_list if "pivot" in x]
    if reference_pivot_table_csv_path:
        #reference_table_pivot_csv_path = reference_table_pivot_csv_path[0]
        csv_files.append(reference_pivot_table_csv_path)
        sheet_names.append("Topic response pivot table")
        column_widths["Topic response pivot table"] = {"A": 25, "B": 60}
        wrap_text_columns["Topic response pivot table"] = ["B"]

    
    if not missing_df_state.empty:
        #unique_topic_table_csv_path = [x for x in file_output_list if "unique_topic" in x]
        #reference_table_csv_path = [x for x in file_output_list if "reference_table" in x]

        missing_df_state_csv_path = output_folder + "missing_df_state_df_for_xlsx.csv"
        missing_df_state.to_csv(missing_df_state_csv_path, index = None)

    if missing_df_state_csv_path:
        #missing_references_table_csv_path = missing_references_table_csv_path[0]
        csv_files.append(missing_df_state_csv_path)
        sheet_names.append("Missing responses")
        column_widths["Missing responses"] = {"A": 25, "B": 30, "C": 50}
        wrap_text_columns["Missing responses"] = ["C"]

    # Original data file
    csv_files.append(original_data_file_path)
    sheet_names.append("Original data")
    column_widths["Original data"] = {"A": 20, "B": 20, "C": 100}
    wrap_text_columns["Original data"] = ["C"]

    print("Creating intro page and text")
    
    # Intro page text
    intro_text = [
        "This workbook contains outputs from the large language model topic analysis of open text data. Each sheet corresponds to a different CSV report included in the analysis.",
        f"The file analysed was {short_file_name}, the column analysed was '{chosen_cols}' and the data was grouped by column '{group}'."
        "Please contact the LLM Topic Modelling app administrator if you need any explanation on how to use the results."
        "Large language models are not 100% accurate and may produce biased or harmful outputs. All outputs from this analysis **need to be checked by a human** to check for harmful outputs, false information, and bias."
    ]    

    xlsx_output_filename = csvs_to_excel(
        csv_files = csv_files,
        output_filename = output_folder + short_file_name + "_topic_analysis_report.xlsx",
        sheet_names = sheet_names,
        column_widths = column_widths,
        wrap_text_columns = wrap_text_columns,
        intro_text = intro_text,
        model_name = model_choice,
        analysis_date = today_date,
        analysis_cost = "Unknown",
        file_name = short_file_name
    )

    xlsx_output_filenames = [xlsx_output_filename]

    return xlsx_output_filenames




