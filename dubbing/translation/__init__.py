from dubbing.translation.base import LanguageDetector, TranslationBackend
from dubbing.translation.lingua_detector import LinguaLanguageDetector
from dubbing.translation.models import SegmentTranslation, TranslationResult
from dubbing.translation.nllb import NLLBTranslationBackend
from dubbing.translation.pipeline import translate_transcript

__all__ = [
    "LanguageDetector",
    "LinguaLanguageDetector",
    "NLLBTranslationBackend",
    "SegmentTranslation",
    "TranslationBackend",
    "TranslationResult",
    "translate_transcript",
]
