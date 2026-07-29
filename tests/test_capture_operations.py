from __future__ import annotations

import hashlib
import json

from dubbing.capture import CaptureService
from dubbing.capture.outbox import export_approved
from tests.test_personal_capture import FakeASR, media


def test_review_edit_reject_retry_and_approve_outbox(tmp_path):
    (tmp_path / "inbox").mkdir()
    source = media(tmp_path / "inbox" / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR())
    outcome = service.process(source)
    service.update_review(
        outcome.capture_id,
        transcript_segments=[{"text": "Исправлено"}],
        translation_segments=None,
        speaker_aliases={"SPEAKER_00": "Artiom"},
        notes="checked",
    )
    transcript = json.loads((outcome.package_path / "transcript.json").read_text())
    assert transcript["text"] == "Исправлено"
    service.reject(outcome.capture_id, notes="noise")
    assert service.store.get(outcome.capture_id).state == "rejected"
    service.store.retry(outcome.capture_id)
    recovered = service.process(source)
    assert recovered.state == "review"
    event = service.approve(recovered.capture_id)
    assert event.is_file()
    assert export_approved(tmp_path / "workspace", tmp_path / "outbox") == 1
    assert export_approved(tmp_path / "workspace", tmp_path / "outbox") == 0


def test_outbox_rejects_tampered_transcript(tmp_path):
    (tmp_path / "inbox").mkdir()
    source = media(tmp_path / "inbox" / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR())
    outcome = service.process(source)
    service.approve(outcome.capture_id)
    transcript = outcome.package_path / "transcript.json"
    transcript.write_text("{}")
    assert export_approved(tmp_path / "workspace", tmp_path / "outbox") == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == outcome.capture_id
