"""Per-batch Bedrock guardrail rejections for topic extraction (CSV + Excel sheet)."""

from __future__ import annotations

import os

import pandas as pd

GUARDRAIL_BATCH_STATUS_BASENAME = "bedrock_guardrail_batch_status.csv"

GUARDRAIL_INTERVENTIONS_SHEET_NAME = "Guardrails interventions"

GUARDRAIL_BATCH_STATUS_COLUMNS = [
    "Batch",
    "Rows",
    "Task",
    "Status",
    "Intervention categories",
    "Notes",
]

_BLOCKED_ACTIONS = frozenset({"BLOCKED", "GUARDRAIL_INTERVENED", "INTERVENED"})
_ANONYMIZED_ACTIONS = frozenset({"ANONYMIZED", "REDACTED"})


def extract_guardrail_intervention_categories(api_response: dict) -> str:
    """
    Summarise guardrail policy categories from a Converse trace.

    Uses policy types and topic names only — not matched text (PII-safe for CSV/logs).
    """
    if not isinstance(api_response, dict):
        return ""
    trace = api_response.get("trace") or {}
    guardrail_trace = trace.get("guardrail") or {}
    if not isinstance(guardrail_trace, dict):
        return ""

    categories: list[str] = []

    def add(label: str) -> None:
        label = (label or "").strip()
        if label and label not in categories:
            categories.append(label)

    def action_triggered(action: str | None) -> bool:
        if not action:
            return False
        action_upper = str(action).upper()
        return (
            action_upper in _BLOCKED_ACTIONS
            or action_upper in _ANONYMIZED_ACTIONS
            or action_upper == "ANONYMIZE"
        )

    def walk_assessment(assessment: dict) -> None:
        if not isinstance(assessment, dict):
            return

        for topic in (assessment.get("topicPolicy") or {}).get("topics") or []:
            if not isinstance(topic, dict):
                continue
            if action_triggered(topic.get("action")) or topic.get("detected"):
                name = (topic.get("name") or "topic").strip()
                add(f"Topic ({name})")

        for filt in (assessment.get("contentPolicy") or {}).get("filters") or []:
            if not isinstance(filt, dict):
                continue
            if action_triggered(filt.get("action")) or filt.get("detected"):
                filt_type = (filt.get("type") or "content").strip()
                add(f"Content ({filt_type})")

        for filt in (assessment.get("contextualGroundingPolicy") or {}).get(
            "filters"
        ) or []:
            if not isinstance(filt, dict):
                continue
            if action_triggered(filt.get("action")) or filt.get("detected"):
                filt_type = (filt.get("type") or "grounding").strip()
                add(f"Contextual grounding ({filt_type})")

        word_policy = assessment.get("wordPolicy") or {}
        for entry in word_policy.get("managedWordLists") or []:
            if not isinstance(entry, dict):
                continue
            if action_triggered(entry.get("action")):
                word_type = (entry.get("type") or "managed words").strip()
                add(f"Word list ({word_type})")
        for entry in word_policy.get("customWords") or []:
            if isinstance(entry, dict) and action_triggered(entry.get("action")):
                add("Custom word")

        sip = assessment.get("sensitiveInformationPolicy") or {}
        for ent in sip.get("piiEntities") or []:
            if not isinstance(ent, dict):
                continue
            if action_triggered(ent.get("action")) or ent.get("detected"):
                ent_type = (ent.get("type") or "PII").strip()
                add(f"Sensitive information ({ent_type})")
        for ent in sip.get("regexes") or []:
            if not isinstance(ent, dict):
                continue
            if action_triggered(ent.get("action")) or ent.get("detected"):
                name = (ent.get("name") or "regex").strip()
                add(f"Sensitive information ({name})")

    input_assessment = guardrail_trace.get("inputAssessment") or {}
    if isinstance(input_assessment, dict):
        for assessment in input_assessment.values():
            walk_assessment(assessment)

    output_assessments = guardrail_trace.get("outputAssessments") or {}
    if isinstance(output_assessments, dict):
        for assessment_or_list in output_assessments.values():
            if isinstance(assessment_or_list, list):
                for assessment in assessment_or_list:
                    walk_assessment(assessment)
            else:
                walk_assessment(assessment_or_list)

    if not categories:
        action_reason = guardrail_trace.get("actionReason")
        if action_reason:
            add(str(action_reason).strip())

    return "; ".join(categories)


_POLICY_UNIT_LABELS = {
    "topicPolicyUnits": "Topic policy",
    "contentPolicyUnits": "Content policy",
    "contentPolicyImageUnits": "Content policy (image)",
    "wordPolicyUnits": "Word policy",
    "sensitiveInformationPolicyUnits": "Sensitive information policy",
    "contextualGroundingPolicyUnits": "Contextual grounding policy",
    "automatedReasoningPolicyUnits": "Automated reasoning policy",
}


def _policy_units_from_trace(api_response: dict) -> str:
    """Infer which guardrail policies ran from invocationMetrics (no matched text)."""
    trace = (api_response or {}).get("trace") or {}
    guardrail_trace = trace.get("guardrail") or {}
    if not isinstance(guardrail_trace, dict):
        return ""

    labels: list[str] = []

    def add(label: str) -> None:
        if label and label not in labels:
            labels.append(label)

    for assessment in (guardrail_trace.get("inputAssessment") or {}).values():
        if not isinstance(assessment, dict):
            continue
        usage = (assessment.get("invocationMetrics") or {}).get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for key, label in _POLICY_UNIT_LABELS.items():
            try:
                if int(usage.get(key) or 0) > 0:
                    add(label)
            except (TypeError, ValueError):
                continue

    for assessments in (guardrail_trace.get("outputAssessments") or {}).values():
        items = assessments if isinstance(assessments, list) else [assessments]
        for assessment in items:
            if not isinstance(assessment, dict):
                continue
            usage = (assessment.get("invocationMetrics") or {}).get("usage") or {}
            if not isinstance(usage, dict):
                continue
            for key, label in _POLICY_UNIT_LABELS.items():
                try:
                    if int(usage.get(key) or 0) > 0:
                        add(label)
                except (TypeError, ValueError):
                    continue

    return "; ".join(labels)


def summarize_guardrail_intervention_for_export(api_response: dict) -> str:
    """
    Best-effort category summary for Excel (policy types / units only, not assessed text).
    """
    if not isinstance(api_response, dict):
        return ""
    categories = extract_guardrail_intervention_categories(api_response)
    if categories:
        return categories
    units = _policy_units_from_trace(api_response)
    if units:
        return units
    if not api_response.get("trace"):
        return "Tracing disabled (set BEDROCK_GUARDRAIL_TRACE=enabled)"
    action_reason = ((api_response.get("trace") or {}).get("guardrail") or {}).get(
        "actionReason"
    )
    if action_reason:
        return str(action_reason).strip()
    return "Guardrail intervened (category not reported in trace)"


def guardrail_categories_from_responses(responses) -> str:
    """Collect intervention category summaries from Bedrock ResponseObject list."""
    parts: list[str] = []
    for response in responses or []:
        if not getattr(response, "guardrail_intervened", False):
            continue
        raw = getattr(response, "guardrail_intervention_categories", "") or ""
        if raw.strip():
            parts.append(raw.strip())
    if not parts:
        return ""
    merged: list[str] = []
    for part in parts:
        for piece in part.split(";"):
            piece = piece.strip()
            if piece and piece not in merged:
                merged.append(piece)
    return "; ".join(merged)


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
    intervention_categories: str = "",
) -> None:
    """Append one guardrail-rejected batch row (creates file if needed)."""
    rows_label = f"Rows {int(start_row) + 1} to {int(end_row) + 1}"
    row = {
        "Batch": int(batch_number),
        "Rows": rows_label,
        "Task": (task_type or "Topic extraction").strip(),
        "Status": "Guardrail rejected",
        "Intervention categories": (intervention_categories or "").strip(),
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
    df = df[GUARDRAIL_BATCH_STATUS_COLUMNS].fillna("")
    return df
