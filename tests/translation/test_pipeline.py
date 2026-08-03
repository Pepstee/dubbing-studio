from __future__ import annotations

from dubbing.transcription import TranscriptSegment, TranscriptionResult
from dubbing.translation.base import LanguageDetector, TranslationBackend
from dubbing.translation.lingua_detector import LinguaLanguageDetector
from dubbing.translation.pipeline import translate_transcript


class Detector(LanguageDetector):
    def detect(self, text: str):
        languages = {"Hello": "en", "Привет": "ru", "Bună": "ro", "안녕": "ko"}
        return languages.get(text), 0.99


class Translator(TranslationBackend):
    @property
    def identity(self):
        return "fake:translator"

    def translate(self, text, *, source_language, target_language):
        return f"{source_language}>{target_language}:{text}"


def transcript(*texts: str) -> TranscriptionResult:
    return TranscriptionResult(
        segments=tuple(
            TranscriptSegment(index * 1000, (index + 1) * 1000, text)
            for index, text in enumerate(texts)
        ),
        text=" ".join(texts),
        backend="fake",
        model="fake",
        device="cpu",
        language=None,
        duration_ms=len(texts) * 1000,
        confidence_available=False,
    )


def test_translates_each_supported_language_and_preserves_source():
    result = translate_transcript(
        transcript("Hello", "Привет", "Bună", "안녕"),
        detector=Detector(),
        backend=Translator(),
    )

    assert [item.source_text for item in result.segments] == [
        "Hello",
        "Привет",
        "Bună",
        "안녕",
    ]
    assert [item.status for item in result.segments] == [
        "source_is_target",
        "translated",
        "translated",
        "translated",
    ]
    assert result.segments[1].target_text == "ru>en:Привет"
    assert result.segments[1].source_segment_id.startswith("segment:1000-2000:")
    assert len(result.segments[1].source_segment_sha256) == 64
    assert result.segments[1].to_dict()["original_language_authoritative"] is True


def test_uncertain_language_is_not_guessed_or_translated():
    result = translate_transcript(
        transcript("?"),
        detector=Detector(),
        backend=Translator(),
    )
    assert result.segments[0].status == "language_uncertain"
    assert result.segments[0].target_text is None


def test_lingua_adapter_uses_iso_code_instead_of_enum_display_name():
    class Iso:
        name = "RU"

    class Language:
        iso_code_639_1 = Iso()

    class FakeLingua:
        def detect_language_of(self, text):
            return Language()

        def compute_language_confidence(self, text, language):
            return 0.9

    detector = object.__new__(LinguaLanguageDetector)
    detector._detector = FakeLingua()
    detector.minimum_confidence = 0.55

    assert detector.detect("Привет") == ("ru", 0.9)
