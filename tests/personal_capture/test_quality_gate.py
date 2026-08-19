import json

import pytest

from dubbing.apps.personal_capture.outbox import export_approved
from dubbing.apps.personal_capture.service import CaptureService
from dubbing.transcription.base import TranscriptionBackend
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


class _Backend(TranscriptionBackend):
    def __init__(self, text):
        self.text = text

    @property
    def identity(self):
        return "fixture:quality"

    def transcribe(self, audio, options=None):
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 10_000, self.text),),
            text=self.text,
            backend="fixture",
            model="quality",
            device="test",
            language="en",
            duration_ms=10_000,
            confidence_available=False,
        )


def test_pathological_transcript_cannot_be_approved_or_exported(tmp_path):
    source = tmp_path / "capture.wav"
    source.write_bytes(b"private")
    service = CaptureService(
        tmp_path / "workspace", _Backend(" ".join(["Loops"] * 109)), resumable=False
    )
    outcome = service.process(source)
    report = json.loads((outcome.package_path / "quality-report.json").read_text())
    assert report["status"] == "REPROCESS_REQUIRED"
    with pytest.raises(ValueError, match="approval and outbox emission are blocked"):
        service.approve(outcome.capture_id, notes="I reviewed it")
    assert export_approved(tmp_path / "workspace", tmp_path / "outbox") == 0


def test_approval_refreshes_stale_quality_policy_before_blocking(tmp_path):
    source = tmp_path / "capture.wav"
    source.write_bytes(b"private")
    service = CaptureService(
        tmp_path / "workspace", _Backend(" ".join(["Loops"] * 109)), resumable=False
    )
    outcome = service.process(source)
    quality_path = outcome.package_path / "quality-report.json"
    manifest_path = outcome.package_path / "manifest.json"
    quality = json.loads(quality_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    quality["policy_version"] = "dubbing.transcript-quality-policy.v1"
    manifest["quality"]["policy_version"] = "dubbing.transcript-quality-policy.v1"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="approval and outbox emission are blocked"):
        service.approve(outcome.capture_id, notes="reviewed")

    refreshed_quality = json.loads(quality_path.read_text())
    refreshed_manifest = json.loads(manifest_path.read_text())
    assert refreshed_quality["policy_version"] == "dubbing.transcript-quality-policy.v3"
    assert refreshed_manifest["quality"]["policy_version"] == (
        "dubbing.transcript-quality-policy.v3"
    )


def test_passing_transcript_emits_quality_bound_evidence(tmp_path):
    source = tmp_path / "capture.wav"
    source.write_bytes(b"private")
    workspace = tmp_path / "workspace"
    service = CaptureService(workspace, _Backend("clean ordinary speech"), resumable=False)
    outcome = service.process(source)
    event_path = service.approve(outcome.capture_id)
    event = json.loads(event_path.read_text())
    assert event["source"]["quality_status"] == "PASS"
    assert event["payload"]["quality_report_file"] == "quality-report.json"
    assert export_approved(workspace, tmp_path / "outbox") == 1
