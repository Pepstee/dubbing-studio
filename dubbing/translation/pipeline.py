from __future__ import annotations

from dubbing.transcription.models import TranscriptionResult
from dubbing.translation.base import LanguageDetector, TranslationBackend
from dubbing.translation.models import SegmentTranslation, TranslationResult


def translate_transcript(
    transcript: TranscriptionResult,
    *,
    detector: LanguageDetector,
    backend: TranslationBackend,
    target_language: str = "en",
    supported_source_languages: tuple[str, ...] = ("en", "ko", "ro", "ru"),
) -> TranslationResult:
    segments = []
    for segment in transcript.segments:
        language, confidence = detector.detect(segment.text)
        if language is None:
            target_text = None
            status = "language_uncertain"
        elif language not in supported_source_languages:
            target_text = None
            status = "unsupported_language"
        elif language == target_language:
            target_text = segment.text
            status = "source_is_target"
        else:
            target_text = backend.translate(
                segment.text,
                source_language=language,
                target_language=target_language,
            )
            status = "translated"
        segments.append(
            SegmentTranslation(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                speaker=segment.speaker,
                source_text=segment.text,
                source_language=language,
                language_confidence=confidence,
                target_text=target_text,
                target_language=target_language,
                status=status,
            )
        )
    return TranslationResult(
        target_language=target_language,
        backend=backend.identity,
        supported_source_languages=supported_source_languages,
        segments=tuple(segments),
    )
