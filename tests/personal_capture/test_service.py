from __future__ import annotations

import hashlib
import json
import os
import wave
from array import array
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

from dubbing.apps.personal_capture import CaptureService
from dubbing.transcription import (
    TranscriptSegment,
    TranscriptionBackend,
    TranscriptionOptions,
    TranscriptionResult,
    evaluate_transcript_quality,
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
            source_sha256=hashlib.sha256(Path(audio).read_bytes()).hexdigest(),
        )


class Detector(LanguageDetector):
    def detect(self, text):
        return "ru", 0.99


class SpeechDetector:
    identity = "fake:speech-regions"


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


def pcm_media(path: Path, *, seconds: int, sample_rate: int = 16_000) -> Path:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(array("h", [0] * (seconds * sample_rate)).tobytes())
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
        resumable=False,
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
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
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
    service = CaptureService(
        tmp_path / "workspace", FakeASR(fail=True), resumable=False
    )
    failed = service.process(source)
    assert failed.state == "failed"

    unchanged = CaptureService(
        tmp_path / "workspace", FakeASR(), resumable=False
    ).process(source)
    assert unchanged.state == "failed"
    assert unchanged.replayed is True

    service.store.retry(failed.capture_id)
    recovered = CaptureService(
        tmp_path / "workspace", FakeASR(), resumable=False
    ).process(source)
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


def test_configurable_runtime_paths_stay_inside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    service = CaptureService(
        workspace,
        FakeASR(),
        packages_dir="outputs/packages",
        state_dir="state",
        resumable=False,
    )
    outcome = service.process(media(tmp_path / "source.wav"))

    assert outcome.package_path.parent == workspace / "outputs" / "packages"
    assert (workspace / "state" / "capture.sqlite3").is_file()

    with pytest.raises(ValueError, match="escapes workspace"):
        CaptureService(
            workspace,
            FakeASR(),
            packages_dir="../outside",
            resumable=False,
        )


def test_resumable_capture_uses_project_processing_directory(tmp_path):
    source = media(tmp_path / "source.wav")
    result = FakeASR().transcribe(source)
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        processing_dir="processing",
    )

    with patch(
        "dubbing.apps.personal_capture.service.ResumableTranscriptionJob.run",
        return_value=result,
    ) as run, patch(
        "dubbing.apps.personal_capture.service.media_duration_ms",
        return_value=1000,
    ):
        outcome = service.process(source)

    assert outcome.state == "review"
    checkpoint_dir = run.call_args_list[0].args[0]
    assert checkpoint_dir == source.resolve()
    assert service.processing == tmp_path / "workspace" / "processing"


def test_adaptive_capture_owns_long_file_chunking_and_preserves_checkpoints(tmp_path):
    source = media(tmp_path / "one-long-recording.wav")
    result = FakeASR().transcribe(source)
    quality = evaluate_transcript_quality(
        result, expected_duration_ms=result.duration_ms
    ).to_dict()
    speech_detector = SpeechDetector()
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        silence_verification_detector=speech_detector,
        targeted_retry_region_detector=None,
        processing_dir="processing",
        transcription_strategy="adaptive",
        transcription_chunk_seconds=240,
        transcription_minimum_chunk_seconds=60,
        transcription_maximum_chunk_seconds=480,
        transcription_overlap_seconds=2,
    )

    with patch(
        "dubbing.apps.personal_capture.service.media_duration_ms",
        return_value=3_600_000,
    ), patch(
        "dubbing.apps.personal_capture.service.AdaptiveLongFormCoordinator"
    ) as coordinator:
        coordinator.return_value.run.return_value = (result, quality)
        outcome = service.process(source)

    assert outcome.state == "review"
    run = coordinator.return_value.run
    assert run.call_count == 1
    assert run.call_args.args[0] == source.resolve()
    assert run.call_args.kwargs["source_digest"] == hashlib.sha256(b"audio").hexdigest()
    assert (
        coordinator.call_args.kwargs["silence_verification_detector"]
        is speech_detector
    )
    assert coordinator.call_args.kwargs["targeted_retry_region_detector"] is None
    manifest = json.loads((outcome.package_path / "manifest.json").read_text())
    assert manifest["execution"]["transcription_strategy"] == "adaptive"
    assert manifest["execution"]["transcription_chunk_seconds"] == 240
    assert (
        manifest["execution"]["silence_verification_detector"]
        == speech_detector.identity
    )
    assert manifest["execution"]["targeted_retry_region_detector"] is None


def test_legacy_capture_detector_binds_both_provenance_roles(tmp_path):
    source = media(tmp_path / "legacy.wav")
    detector = SpeechDetector()
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        speech_region_detector=detector,
        resumable=False,
    )

    outcome = service.process(source)

    manifest = json.loads((outcome.package_path / "manifest.json").read_text())
    assert manifest["execution"]["speech_region_detector"] == detector.identity
    assert manifest["execution"]["silence_verification_detector"] == detector.identity
    assert manifest["execution"]["targeted_retry_region_detector"] == detector.identity


def test_one_long_file_is_automatically_split_into_internal_checkpoints(tmp_path):
    source = pcm_media(tmp_path / "single-upload.wav", seconds=7)
    backend = FakeASR()
    service = CaptureService(
        tmp_path / "workspace",
        backend,
        transcription_strategy="adaptive",
        transcription_chunk_seconds=2,
        transcription_minimum_chunk_seconds=1,
        transcription_maximum_chunk_seconds=3,
        transcription_overlap_seconds=0.25,
    )

    with patch(
        "dubbing.transcription.adaptive.detect_silence_intervals",
        return_value=(),
    ), patch(
        "dubbing.transcription.adaptive.detect_silence_centres",
        return_value=(),
    ):
        outcome = service.process(source)

    assert outcome.state == "review"
    checkpoint_root = (
        tmp_path
        / "workspace"
        / "processing"
        / outcome.capture_id
        / "transcription-adaptive"
    )
    checkpoint_manifest = json.loads(
        (checkpoint_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(checkpoint_manifest["chunks"]) == 3
    assert len(tuple((checkpoint_root / "chunks").glob("*.json"))) == 3
    assert backend.calls >= 3


def test_capture_fails_when_source_changes_during_processing(tmp_path):
    source = media(tmp_path / "source.wav")

    class MutatingASR(FakeASR):
        def transcribe(self, audio, options=None):
            result = super().transcribe(audio, options)
            Path(audio).write_bytes(b"changed")
            return result

    service = CaptureService(
        tmp_path / "workspace",
        MutatingASR(),
        resumable=False,
    )

    outcome = service.process(source)

    assert outcome.state == "failed"
    assert "source changed" in outcome.error


def test_capture_enforces_free_space_reserve(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        resumable=False,
        minimum_free_bytes=1,
    )

    with patch(
        "dubbing.apps.personal_capture.service.shutil.disk_usage",
        return_value=SimpleNamespace(free=0),
    ):
        outcome = service.process(source)

    assert outcome.state == "failed"
    assert "free-space reserve" in outcome.error


def test_unchanged_reviewed_source_is_not_rehashed_on_later_scans(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        resumable=False,
    )
    first = service.process(source)

    with patch(
        "dubbing.apps.personal_capture.service._sha256",
        side_effect=AssertionError("unchanged source was rehashed"),
    ):
        replay = service.process(source)

    assert first.state == replay.state == "review"
    assert replay.replayed is True


def test_processing_capture_is_not_claimed_concurrently(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stat = source.stat()
    record, first_claim = service.store.claim(
        capture_id=digest,
        source_name=source.name,
        source_sha256=digest,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )
    second, second_claim = service.store.claim(
        capture_id=digest,
        source_name=source.name,
        source_sha256=digest,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )

    assert first_claim is True
    assert second_claim is False
    assert second.attempt_count == record.attempt_count == 1


def test_interrupted_work_is_requeued_once_by_new_watcher(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    stat = source.stat()
    service.store.claim(
        capture_id=digest,
        source_name=source.name,
        source_sha256=digest,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )

    assert service.store.requeue_interrupted() == 1
    assert service.store.requeue_interrupted() == 0
    assert service.store.get(digest).state == "queued"
