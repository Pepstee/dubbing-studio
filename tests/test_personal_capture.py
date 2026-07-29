from __future__ import annotations

import json
import os
from pathlib import Path

from dubbing.capture import CaptureService
from dubbing.transcription import (
    TranscriptSegment,
    TranscriptionBackend,
    TranscriptionOptions,
    TranscriptionResult,
)
from dubbing.translation.base import LanguageDetector, TranslationBackend


class FakeASR(TranscriptionBackend):
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    @property
    def identity(self):
        return "fake:asr"

    def transcribe(self, audio, options: TranscriptionOptions | None = None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 1000, "Привет"),),
            text="Привет",
            backend="fake",
            model="asr",
            device="cpu",
            language="ru",
            duration_ms=1000,
            confidence_available=False,
            source_sha256="a" * 64,
        )


class Detector(LanguageDetector):
    def detect(self, text):
        return "ru", 0.99


class Translator(TranslationBackend):
    @property
    def identity(self):
        return "fake:translation"

    def translate(self, text, *, source_language, target_language):
        return "Hello"


def media(path: Path, content: bytes = b"audio") -> Path:
    path.write_bytes(content)
    os.utime(path, (1, 1))
    return path


def test_scan_packages_translation_and_deduplicates_by_content(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    media(inbox / "one.wav")
    media(inbox / "copy.wav")
    asr = FakeASR()
    service = CaptureService(
        tmp_path / "workspace",
        asr,
        language_detector=Detector(),
        translation_backend=Translator(),
    )

    outcomes = service.scan(inbox, min_age_seconds=0)

    assert asr.calls == 1
    assert {outcome.state for outcome in outcomes} == {"review"}
    assert sum(outcome.replayed for outcome in outcomes) == 1
    package = next(outcome.package_path for outcome in outcomes if outcome.package_path)
    assert json.loads((package / "manifest.json").read_text())["state"] == "review"
    translation = json.loads((package / "translation.json").read_text())
    assert translation["segments"][0]["source_text"] == "Привет"
    assert translation["segments"][0]["target_text"] == "Hello"
    assert not (package / "giga-event.json").exists()


def test_approval_emits_idempotent_giga_event_with_speaker_alias(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR())
    outcome = service.process(source)

    event_path = service.approve(
        outcome.capture_id,
        speaker_aliases={"SPEAKER_00": "Artiom"},
        notes="reviewed",
    )
    first = event_path.read_text()
    second = service.approve(
        outcome.capture_id,
        speaker_aliases={"SPEAKER_00": "Artiom"},
        notes="reviewed",
    ).read_text()

    assert first == second
    event = json.loads(second)
    assert event["payload"]["speaker_aliases"] == {"SPEAKER_00": "Artiom"}
    assert event["trust_class"] == "operator-reviewed-derived-evidence"
    assert service.store.get(outcome.capture_id).state == "approved"


def test_failure_is_retryable(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR(fail=True))
    failed = service.process(source)
    assert failed.state == "failed"

    recovered = CaptureService(tmp_path / "workspace", FakeASR()).process(source)
    assert recovered.state == "review"
    assert service.store.get(recovered.capture_id).attempt_count == 2


def test_discovery_ignores_symlinks_unknown_files_and_recent_writes(tmp_path):
    old = media(tmp_path / "old.wav")
    recent = tmp_path / "recent.wav"
    recent.write_bytes(b"recent")
    (tmp_path / "notes.txt").write_text("not media")
    (tmp_path / "link.wav").symlink_to(old)

    discovered = CaptureService.discover(
        tmp_path,
        min_age_seconds=10,
        now=20,
    )
    assert discovered == (old,)
