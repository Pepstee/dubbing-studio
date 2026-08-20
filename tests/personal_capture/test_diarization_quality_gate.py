from __future__ import annotations

import hashlib
import json

import pytest

from dubbing.apps.personal_capture import CaptureService
from dubbing.apps.personal_capture.outbox import verify_outbox_bundle
from dubbing.diarization import DiarizationBackend, DiarizationResult, SpeakerTurn
from dubbing.transcription import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionBackend,
    TranscriptionOptions,
    TranscriptionResult,
)
from tests.personal_capture.test_service import FakeASR, media


class WordASR(TranscriptionBackend):
    @property
    def identity(self):
        return "synthetic:word-asr"

    def transcribe(self, audio, options: TranscriptionOptions | None = None):
        return TranscriptionResult(
            segments=(
                TranscriptSegment(
                    0,
                    1_000,
                    "one two",
                    words=(
                        TranscriptWord(100, 400, "one"),
                        TranscriptWord(500, 900, "two"),
                    ),
                ),
            ),
            text="one two",
            backend="synthetic",
            model="word-asr",
            device="cpu",
            language="en",
            duration_ms=1_000,
            confidence_available=False,
        )


class StaticDiarizer(DiarizationBackend):
    identity = "synthetic:diarizer"

    def __init__(self, *turns: SpeakerTurn):
        self.turns = turns

    def diarize(self, audio, constraints=None):
        return DiarizationResult(
            turns=self.turns,
            backend="synthetic",
            model="static-diarizer",
            device="cpu",
        )


def test_pass_report_is_hash_bound_and_exported_with_approved_package(tmp_path):
    workspace = tmp_path / "workspace"
    outbox = tmp_path / "outbox"
    service = CaptureService(
        workspace,
        WordASR(),
        diarizer=StaticDiarizer(SpeakerTurn(0, 1_000, "SPEAKER_00")),
        resumable=False,
        outbox_dir=outbox,
    )
    outcome = service.process(media(tmp_path / "source.wav"))
    package = outcome.package_path

    manifest = json.loads((package / "manifest.json").read_text())
    report = json.loads((package / "diarization-quality-report.json").read_text())
    assert manifest["diarization_quality"]["status"] == "PASS"
    assert report["status"] == "PASS"
    assert report["transcript_sha256"] == hashlib.sha256(
        (package / "transcript.json").read_bytes()
    ).hexdigest()
    assert report["diarization_sha256"] == manifest["diarization_quality"][
        "diarization_sha256"
    ]
    assert manifest["diarization_quality"]["report_sha256"] == hashlib.sha256(
        (package / "diarization-quality-report.json").read_bytes()
    ).hexdigest()

    event = json.loads(service.approve(outcome.capture_id).read_text())
    assert event["source"]["diarization_quality_status"] == "PASS"
    assert event["source"]["diarization_quality_report_sha256"]
    bundle = outbox / outcome.capture_id
    assert verify_outbox_bundle(bundle)
    assert (bundle / "diarization.json").is_file()
    assert (bundle / "diarization-quality-report.json").is_file()


def test_human_review_status_requires_explicit_diarization_acknowledgement(tmp_path):
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        diarizer=StaticDiarizer(
            SpeakerTurn(0, 1_000, "SPEAKER_00"),
            SpeakerTurn(0, 1_000, "SPEAKER_01"),
        ),
        resumable=False,
    )
    outcome = service.process(media(tmp_path / "source.wav"))
    report = json.loads(
        (outcome.package_path / "diarization-quality-report.json").read_text()
    )
    assert report["status"] == "HUMAN_REVIEW_REQUIRED"

    with pytest.raises(ValueError, match="explicit review or correction"):
        service.approve(outcome.capture_id)

    event_path = service.approve(
        outcome.capture_id,
        diarization_review_acknowledged=True,
    )
    approval = json.loads((outcome.package_path / "approval.json").read_text())
    assert event_path.is_file()
    assert approval["diarization_review_acknowledged"] is True


def test_reprocess_status_blocks_approval_and_outbox_emission(tmp_path):
    outbox = tmp_path / "outbox"
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        diarizer=StaticDiarizer(),
        resumable=False,
        outbox_dir=outbox,
    )
    outcome = service.process(media(tmp_path / "source.wav"))
    report = json.loads(
        (outcome.package_path / "diarization-quality-report.json").read_text()
    )
    assert report["status"] == "REPROCESS_REQUIRED"

    with pytest.raises(ValueError, match="approval and outbox emission are blocked"):
        service.approve(
            outcome.capture_id,
            diarization_review_acknowledged=True,
        )

    assert not (outcome.package_path / "giga-event.json").exists()
    assert not outbox.exists()


def test_approval_rejects_diarization_evidence_that_differs_from_transcript(tmp_path):
    service = CaptureService(
        tmp_path / "workspace",
        WordASR(),
        diarizer=StaticDiarizer(SpeakerTurn(0, 1_000, "SPEAKER_00")),
        resumable=False,
    )
    outcome = service.process(media(tmp_path / "source.wav"))
    evidence = outcome.package_path / "diarization.json"
    document = json.loads(evidence.read_text())
    document["model"] = "tampered"
    evidence.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="does not match diarization.json"):
        service.approve(outcome.capture_id)


def test_outbox_detects_tampered_diarization_quality_report(tmp_path):
    outbox = tmp_path / "outbox"
    service = CaptureService(
        tmp_path / "workspace",
        WordASR(),
        diarizer=StaticDiarizer(SpeakerTurn(0, 1_000, "SPEAKER_00")),
        resumable=False,
        outbox_dir=outbox,
    )
    outcome = service.process(media(tmp_path / "source.wav"))
    service.approve(outcome.capture_id)
    bundle = outbox / outcome.capture_id
    assert verify_outbox_bundle(bundle)

    (bundle / "diarization-quality-report.json").write_text("{}")

    assert not verify_outbox_bundle(bundle)


def test_transcript_only_capture_does_not_require_diarization_gate(tmp_path):
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
    outcome = service.process(media(tmp_path / "source.wav"))
    manifest = json.loads((outcome.package_path / "manifest.json").read_text())

    assert manifest["diarization_quality"] is None
    assert not (outcome.package_path / "diarization-quality-report.json").exists()
    assert service.approve(outcome.capture_id).is_file()
