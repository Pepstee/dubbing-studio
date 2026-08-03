from __future__ import annotations

import unicodedata
from dataclasses import replace

from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


SUPPORTED_LANGUAGES = ("en", "ru", "ro", "ko")


def script_evidence(text: str) -> dict[str, int]:
    counts = {"latin": 0, "cyrillic": 0, "hangul": 0, "other": 0}
    for character in text:
        if not character.isalpha():
            continue
        point = ord(character)
        if 0x0400 <= point <= 0x052F:
            counts["cyrillic"] += 1
        elif 0xAC00 <= point <= 0xD7AF or 0x1100 <= point <= 0x11FF:
            counts["hangul"] += 1
        elif "LATIN" in unicodedata.name(character, ""):
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def annotate_turn_language(
    segment: TranscriptSegment,
    *,
    recording_language: str | None = None,
) -> TranscriptSegment:
    """Resolve language at ASR-turn granularity without translating the source text."""
    probabilities = segment.diagnostics.language_probabilities if segment.diagnostics else None
    language = segment.language
    confidence = segment.language_confidence
    if probabilities:
        eligible = {key: value for key, value in probabilities.items() if key in SUPPORTED_LANGUAGES}
        if eligible:
            language, confidence = max(eligible.items(), key=lambda item: item[1])
    evidence = script_evidence(segment.text)
    letters = sum(evidence.values())
    scripted_language = None
    if letters and evidence["cyrillic"] / letters >= 0.5:
        scripted_language = "ru"
    elif letters and evidence["hangul"] / letters >= 0.5:
        scripted_language = "ko"
    elif letters and evidence["latin"] / letters >= 0.5:
        romanian_marks = sum(character.casefold() in "ăâîșț" for character in segment.text)
        scripted_language = "ro" if romanian_marks >= 2 else None
    if language not in SUPPORTED_LANGUAGES and scripted_language:
        language = scripted_language
    language = language or scripted_language or recording_language
    mismatch = bool(
        scripted_language
        and language
        and scripted_language != language
        and scripted_language in {"ru", "ro", "ko"}
    )
    uncertain = segment.uncertain or mismatch or language not in SUPPORTED_LANGUAGES
    if confidence is not None and confidence < 0.6:
        uncertain = True
    return replace(
        segment,
        language=language,
        language_confidence=confidence,
        uncertain=uncertain,
    )


def annotate_transcript_languages(result: TranscriptionResult) -> TranscriptionResult:
    segments = tuple(
        annotate_turn_language(segment, recording_language=result.language)
        for segment in result.segments
    )
    return replace(result, segments=segments)
