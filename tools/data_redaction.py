"""Optional Presidio/spaCy redaction for topic-modelling tables."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

DEFAULT_LANGUAGE = "en"
SCORE_THRESHOLD = 0.4

_analyser = None
_batch_analyzer = None
_batch_anonymizer = None

_REDACTION_UNAVAILABLE_MESSAGE = (
    "PII redaction is enabled but spaCy/Presidio could not be loaded. "
    "Install presidio-analyzer, presidio-anonymizer, and spacy, then run "
    "`python -m spacy download en_core_web_sm`. "
)


def reset_redaction_engines() -> None:
    """Drop cached Presidio engines (used by tests)."""
    global _analyser, _batch_analyzer, _batch_anonymizer
    _analyser = None
    _batch_analyzer = None
    _batch_anonymizer = None


def _cell_to_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.lower() in ("nan", "none", "<na>"):
        return ""
    return text


def _normalize_strategy(strategy: str) -> str:
    return (strategy or "entity_type").strip().lower()


def _operator_config(strategy: str):
    from presidio_anonymizer.entities import OperatorConfig

    key = _normalize_strategy(strategy)
    if key in (
        "redact_replace",
        "replace_redacted",
        "replace with 'redacted'",
        "replace with redacted",
    ):
        return {"DEFAULT": OperatorConfig("replace", {"new_value": "REDACTED"})}
    if key in ("redact", "redact completely"):
        return {"DEFAULT": OperatorConfig("redact")}
    # entity_type / replace with <ENTITY_NAME>
    return {"DEFAULT": OperatorConfig("replace")}


def _ensure_engines():
    global _analyser, _batch_analyzer, _batch_anonymizer
    if (
        _analyser is not None
        and _batch_analyzer is not None
        and _batch_anonymizer is not None
    ):
        return _analyser, _batch_analyzer, _batch_anonymizer

    try:
        from presidio_analyzer import BatchAnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine

        from tools.load_spacy_model_custom_recognisers import create_nlp_analyser

        _analyser = create_nlp_analyser()
        _batch_analyzer = BatchAnalyzerEngine(analyzer_engine=_analyser)
        _batch_anonymizer = BatchAnonymizerEngine(anonymizer_engine=AnonymizerEngine())
    except Exception as exc:
        reset_redaction_engines()
        raise RuntimeError(
            _REDACTION_UNAVAILABLE_MESSAGE + f"Original error: {exc}"
        ) from exc

    return _analyser, _batch_analyzer, _batch_anonymizer


def _resolve_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        return list(df.columns)
    if isinstance(columns, str):
        columns = [columns]
    resolved = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(
            "Skipping redaction for columns not found in dataframe: "
            + ", ".join(str(col) for col in missing)
        )
    return resolved


def redact_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    entities: Optional[List[str]] = None,
    strategy: Optional[str] = None,
) -> pd.DataFrame:
    """Redact recognised PII in selected columns. Returns a copy of ``df``."""
    from tools.config import REDACTION_ENTITIES, REDACTION_STRATEGY

    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    chosen_cols = _resolve_columns(df, columns)
    if not chosen_cols:
        return df.copy()

    entity_list = list(entities) if entities is not None else list(REDACTION_ENTITIES)
    entity_list = [str(item).strip() for item in entity_list if str(item).strip()]
    if not entity_list:
        return df.copy()

    chosen_strategy = strategy if strategy is not None else REDACTION_STRATEGY
    _, batch_analyzer, batch_anonymizer = _ensure_engines()

    work = df.copy()
    df_dict = {}
    for col in chosen_cols:
        df_dict[col] = [_cell_to_text(value) for value in work[col].tolist()]

    analyzer_results = list(
        batch_analyzer.analyze_dict(
            df_dict,
            language=DEFAULT_LANGUAGE,
            entities=entity_list,
            score_threshold=SCORE_THRESHOLD,
        )
    )
    anonymizer_results = batch_anonymizer.anonymize_dict(
        analyzer_results, operators=_operator_config(chosen_strategy)
    )

    for col in chosen_cols:
        if col in anonymizer_results:
            work[col] = anonymizer_results[col]

    return work
