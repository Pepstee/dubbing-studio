from __future__ import annotations

from dubbing.translation.base import LanguageDetector


class LinguaLanguageDetector(LanguageDetector):
    """Short-text detector constrained to Artiom's four daily languages."""

    def __init__(self, *, minimum_confidence: float = 0.55) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as exc:
            raise RuntimeError(
                "Lingua is not installed. Install dubbing-studio[translation-local]."
            ) from exc
        languages = (
            Language.ENGLISH,
            Language.KOREAN,
            Language.ROMANIAN,
            Language.RUSSIAN,
        )
        self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
        self.minimum_confidence = minimum_confidence

    @property
    def identity(self) -> str:
        return (
            "lingua:en,ko,ro,ru:"
            f"minimum_confidence={self.minimum_confidence}"
        )

    def detect(self, text: str) -> tuple[str | None, float | None]:
        value = text.strip()
        if not value:
            return None, None
        language = self._detector.detect_language_of(value)
        if language is None:
            return None, None
        confidence = float(self._detector.compute_language_confidence(value, language))
        if confidence < self.minimum_confidence:
            return None, confidence
        return str(language.iso_code_639_1.name).lower(), confidence
