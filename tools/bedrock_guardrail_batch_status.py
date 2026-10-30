"""Per-batch Bedrock guardrail rejections for topic extraction (CSV + Excel sheet)."""

from __future__ import annotations

import os

import pandas as pd

GUARDRAIL_BATCH_STATUS_BASENAME = "bedrock_guardrail_batch_status.csv"

GUARDRAIL_BATCH_STATUS_COLUMNS = [
    "Batch",
    "Rows",
    "Task",
    "Status",
    "Notes",
]


def guardrail_batch_status_path(output_folder: str) -> str:
    folder = output_folder or ""
    if folder and not folder.endswith(("/", "\\", os.sep)):
        folder = folder + os.sep
    return folder + GUARDRAIL_BATCH_STATUS_BASENAME


def clear_guardrail_batch_status(output_folder: str) -> None:
    path = guardrail_batch_status_path(output_folder)
    if os.path.isfile(path):
        os.remove(path)


def append_guardrail_batch_status(
    output_folder: str,
    *,
    batch_number: int,
    start_row: int,
    end_row: int,
    task_type: str = "Topic extraction",
    notes: str = "Bedrock guardrail intervened; no topic assignments for this batch.",
) -> None:
    """Append one guardrail-rejected batch row (creates file if needed)."""
    rows_label = f"Rows {int(start_row) + 1} to {int(end_row) + 1}"
    row = {
        "Batch": int(batch_number),
        "Rows": rows_label,
        "Task": (task_type or "Topic extraction").strip(),
        "Status": "Guardrail rejected",
        "Notes": notes,
    }
    path = guardrail_batch_status_path(output_folder)
    new_row = pd.DataFrame([row], columns=GUARDRAIL_BATCH_STATUS_COLUMNS)
    if os.path.isfile(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def load_guardrail_batch_status(output_folder: str) -> pd.DataFrame:
    path = guardrail_batch_status_path(output_folder)
    if not os.path.isfile(path):
        return pd.DataFrame(columns=GUARDRAIL_BATCH_STATUS_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=GUARDRAIL_BATCH_STATUS_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=GUARDRAIL_BATCH_STATUS_COLUMNS)
    for col in GUARDRAIL_BATCH_STATUS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[GUARDRAIL_BATCH_STATUS_COLUMNS]
