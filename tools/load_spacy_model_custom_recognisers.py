"""Slim spaCy/Presidio recognisers for optional PII redaction.

Adapted from the document-redaction app. This module is imported only when
redaction is enabled so Gradio/Lambda cold starts stay unchanged by default.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import (
    NerModelConfiguration,
    NlpArtifacts,
    SpacyNlpEngine,
)

score_threshold = 0.4
DEFAULT_LANGUAGE = "en"

# Custom title recogniser
titles_list = [
    "Sir",
    "Ma'am",
    "Madam",
    "Mr",
    "Mr.",
    "Mrs",
    "Mrs.",
    "Ms",
    "Ms.",
    "Miss",
    "Dr",
    "Dr.",
    "Professor",
]
titles_regex = (
    "\\b" + "\\b|\\b".join(rf"{re.escape(title)}" for title in titles_list) + "\\b"
)
titles_pattern = Pattern(name="titles_pattern", regex=titles_regex, score=1)
titles_recogniser = PatternRecognizer(
    supported_entity="TITLES",
    name="TITLES",
    patterns=[titles_pattern],
    global_regex_flags=re.DOTALL | re.MULTILINE,
)

ukpostcode_pattern = Pattern(
    name="ukpostcode_pattern",
    regex=r"\b([A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0AA)\b",
    score=1,
)
ukpostcode_recogniser = PatternRecognizer(
    supported_entity="UKPOSTCODE", name="UKPOSTCODE", patterns=[ukpostcode_pattern]
)


def extract_street_name(text: str) -> Tuple[List[int], List[int]]:
    """Return start/end character offsets for house-number + street-type phrases."""
    street_types = [
        "Street",
        "St",
        "Boulevard",
        "Blvd",
        "Highway",
        "Hwy",
        "Broadway",
        "Freeway",
        "Causeway",
        "Cswy",
        "Expressway",
        "Way",
        "Walk",
        "Lane",
        "Ln",
        "Road",
        "Rd",
        "Avenue",
        "Ave",
        "Circle",
        "Cir",
        "Cove",
        "Cv",
        "Drive",
        "Dr",
        "Parkway",
        "Pkwy",
        "Park",
        "Court",
        "Ct",
        "Square",
        "Sq",
        "Loop",
        "Place",
        "Pl",
        "Parade",
        "Estate",
        "Alley",
        "Arcade",
        "Bay",
        "Bend",
        "Brae",
        "Byway",
        "Close",
        "Corner",
        "Crescent",
        "Cres",
        "Cul-de-sac",
        "Dell",
        "Esplanade",
        "Glen",
        "Green",
        "Grove",
        "Heights",
        "Hts",
        "Mews",
        "Path",
        "Piazza",
        "Promenade",
        "Quay",
        "Ridge",
        "Row",
        "Terrace",
        "Ter",
        "Track",
        "Trail",
        "View",
        "Villas",
        "Marsh",
        "Embankment",
        "Cut",
        "Hill",
        "Passage",
        "Rise",
        "Vale",
        "Side",
    ]

    street_types_pattern = "|".join(
        rf"{re.escape(street_type)}" for street_type in street_types
    )
    pattern = r"(?P<preceding_word>\w*\d\w*)\s*"
    pattern += rf"(?P<street_name>\w+\s*\b(?:{street_types_pattern})\b)"

    start_positions: List[int] = []
    end_positions: List[int] = []
    for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE):
        start_positions.append(match.start())
        end_positions.append(match.end())
    return start_positions, end_positions


class StreetNameRecognizer(EntityRecognizer):
    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
    ) -> List[RecognizerResult]:
        if entities and "STREETNAME" not in entities:
            return []
        start_pos, end_pos = extract_street_name(text)
        results = []
        for i in range(len(start_pos)):
            results.append(
                RecognizerResult(
                    entity_type="STREETNAME",
                    start=start_pos[i],
                    end=end_pos[i],
                    score=1,
                )
            )
        return results


street_recogniser = StreetNameRecognizer(supported_entities=["STREETNAME"])


class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model, language_code: str):
        super().__init__(
            models=[{"lang_code": language_code, "model_name": "en_core_web_sm"}],
            ner_model_configuration=NerModelConfiguration(
                labels_to_ignore=["CARDINAL", "ORDINAL"]
            ),
        )
        self.nlp = {language_code: loaded_spacy_model}

    def load(self) -> None:
        """The spaCy model is already supplied in ``__init__``."""
        return


def _normalize_language_input(language: str) -> str:
    return language.strip().lower().replace("-", "_")


def _base_language_code(language: str) -> str:
    lang = _normalize_language_input(language)
    if "_" in lang:
        return lang.split("_")[0]
    return lang


def load_spacy_model(language: str = DEFAULT_LANGUAGE):
    """Load the configured English spaCy model, downloading it if missing."""
    import spacy
    from spacy.cli.download import download

    from tools.config import SPACY_MODEL, SPACY_MODEL_PATH

    if SPACY_MODEL_PATH and str(SPACY_MODEL_PATH).strip():
        os.environ["SPACY_DATA"] = SPACY_MODEL_PATH
        print(f"Setting spaCy model path to: {SPACY_MODEL_PATH}")

    requested = (SPACY_MODEL or "en_core_web_sm").strip() or "en_core_web_sm"
    candidates = []
    for name in (requested, "en_core_web_sm"):
        if name and name not in candidates:
            candidates.append(name)

    last_error = None
    for candidate in candidates:
        try:
            module = __import__(candidate)
            print(f"[OK] Successfully imported spaCy model: {candidate}")
            return module.load()
        except Exception as e:
            last_error = e

        try:
            nlp = spacy.load(candidate)
            print(f"[OK] Successfully loaded spaCy model via spacy.load: {candidate}")
            return nlp
        except OSError:
            print(f"Model {candidate} not found, attempting to download...")
            try:
                download(candidate)
                nlp = spacy.load(candidate)
                print(f"[OK] Successfully loaded downloaded spaCy model: {candidate}")
                return nlp
            except Exception as download_error:
                print(f"[ERR] Failed to download or load {candidate}: {download_error}")
                last_error = download_error
                continue
        except Exception as e:
            print(f"[ERR] Failed to load {candidate}: {e}")
            last_error = e
            continue

    error_msg = f"Failed to load spaCy model for language '{language}'"
    if last_error:
        error_msg += f". Last error: {last_error}"
    error_msg += f". Tried candidates: {candidates}"
    raise RuntimeError(error_msg)


def create_nlp_analyser(
    language: str = DEFAULT_LANGUAGE,
    existing_nlp_analyser: Optional[AnalyzerEngine] = None,
):
    """Create a Presidio analyser with UK postcode/street/title recognisers."""
    base_lang_code = _base_language_code(language)

    if existing_nlp_analyser is not None:
        supported = getattr(existing_nlp_analyser, "supported_languages", None) or []
        if supported and supported[0] == base_lang_code:
            print(f"Using existing nlp_analyser for {language}")
            return existing_nlp_analyser

    nlp_model = load_spacy_model(language)
    loaded_nlp_engine = LoadedSpacyNlpEngine(
        loaded_spacy_model=nlp_model, language_code=base_lang_code
    )
    nlp_analyser = AnalyzerEngine(
        nlp_engine=loaded_nlp_engine,
        default_score_threshold=score_threshold,
        supported_languages=[base_lang_code],
        log_decision_process=False,
    )

    if base_lang_code == "en":
        nlp_analyser.registry.add_recognizer(street_recogniser)
        nlp_analyser.registry.add_recognizer(ukpostcode_recogniser)
        nlp_analyser.registry.add_recognizer(titles_recogniser)

    return nlp_analyser
