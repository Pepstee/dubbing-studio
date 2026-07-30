from dataclasses import replace

import pytest

from dubbing.transcription import TranscriptSegment, TranscriptionResult
from dubbing.translation import (
    LanguageDetector,
    ResumableTranslationJob,
    TranslationBackend,
)


class _Detector(LanguageDetector):
    def __init__(self, identity="test:detector"):
        self._identity = identity

    @property
    def identity(self):
        return self._identity

    def detect(self, text):
        return "ru", 1.0


class _Translator(TranslationBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "test:translation"

    def translate(self, text, *, source_language, target_language):
        self.calls += 1
        return f"translated {text}"


def _transcript():
    return TranscriptionResult(
        segments=(
            TranscriptSegment(0, 1000, "один"),
            TranscriptSegment(1000, 2000, "два"),
        ),
        text="один два",
        backend="test",
        model="fixture",
        device="cpu",
        language="ru",
        duration_ms=2000,
        confidence_available=False,
        source_sha256="a" * 64,
    )


def test_translation_resumes_without_repeating_completed_segments(tmp_path):
    translator = _Translator()
    job = ResumableTranslationJob(
        _Detector(),
        translator,
        tmp_path / "translation",
    )

    first = job.run(_transcript())
    second = job.run(_transcript())

    assert translator.calls == 2
    assert second.to_dict() == first.to_dict()
    assert (tmp_path / "translation" / "progress.json").is_file()


def test_translation_checkpoint_rejects_changed_transcript(tmp_path):
    job = ResumableTranslationJob(
        _Detector(),
        _Translator(),
        tmp_path / "translation",
    )
    transcript = _transcript()
    job.run(transcript)

    changed = replace(
        transcript,
        segments=(
            TranscriptSegment(0, 1000, "changed"),
            transcript.segments[1],
        ),
    )
    with pytest.raises(RuntimeError, match="does not match"):
        job.run(changed)


def test_translation_checkpoint_rejects_changed_detector_identity(tmp_path):
    transcript = _transcript()
    checkpoint = tmp_path / "translation"
    ResumableTranslationJob(
        _Detector("detector:v1"),
        _Translator(),
        checkpoint,
    ).run(transcript)

    with pytest.raises(RuntimeError, match="does not match"):
        ResumableTranslationJob(
            _Detector("detector:v2"),
            _Translator(),
            checkpoint,
        ).run(transcript)
