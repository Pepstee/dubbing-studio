from __future__ import annotations

import hashlib
import json

from dubbing.apps.personal_capture import CaptureService
import pytest

from dubbing.apps.personal_capture.outbox import export_approved, verify_outbox_bundle
from tests.personal_capture.test_service import Detector, FakeASR, Translator, media


def test_review_edit_reject_retry_and_approve_outbox(tmp_path):
    (tmp_path / "inbox").mkdir()
    source = media(tmp_path / "inbox" / "source.wav")
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
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
    service = CaptureService(tmp_path / "workspace", FakeASR(), resumable=False)
    outcome = service.process(source)
    service.approve(outcome.capture_id)
    transcript = outcome.package_path / "transcript.json"
    transcript.write_text("{}")
    assert export_approved(tmp_path / "workspace", tmp_path / "outbox") == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == outcome.capture_id


def test_outbox_supports_configured_packages_directory(tmp_path):
    workspace = tmp_path / "workspace"
    source = media(tmp_path / "source.wav")
    service = CaptureService(
        workspace,
        FakeASR(),
        packages_dir="outputs/packages",
        state_dir="state",
        resumable=False,
    )
    outcome = service.process(source)
    service.approve(outcome.capture_id)

    assert export_approved(
        workspace,
        workspace / "outbox" / "giga",
        packages_dir="outputs/packages",
    ) == 1


def test_configured_approval_publishes_transactional_outbox(tmp_path):
    workspace = tmp_path / "workspace"
    inbox = workspace / "recordings" / "inbox"
    inbox.mkdir(parents=True)
    source = media(inbox / "source.wav")
    service = CaptureService(
        workspace,
        FakeASR(),
        resumable=False,
        inbox_dir=inbox,
        outbox_dir=workspace / "outbox" / "giga",
    )
    outcome = service.process(source)

    service.approve(outcome.capture_id)
    service.approve(outcome.capture_id)

    bundles = [
        path
        for path in (workspace / "outbox" / "giga").iterdir()
        if path.is_dir()
    ]
    assert [path.name for path in bundles] == [outcome.capture_id]
    assert (bundles[0] / "transcript.json").is_file()
    assert (bundles[0] / "approval.json").is_file()
    assert verify_outbox_bundle(bundles[0])


def test_review_regenerates_all_evidence_and_blocks_stale_translation(tmp_path):
    source = media(tmp_path / "source.wav")
    service = CaptureService(
        tmp_path / "workspace",
        FakeASR(),
        language_detector=Detector(),
        translation_backend=Translator(),
        resumable=False,
    )
    outcome = service.process(source)
    package = outcome.package_path
    service.update_review(
        outcome.capture_id,
        transcript_segments=[{"text": "Исправлено"}],
        translation_segments=[{"target_text": "Hello"}],
        speaker_aliases={},
        notes="source corrected",
    )

    transcript = json.loads((package / "transcript.json").read_text())
    translation = json.loads((package / "translation.json").read_text())
    assert transcript["segments"][0]["words"] == []
    assert (package / "transcript.txt").read_text() == "Исправлено\n"
    assert "Исправлено" in (package / "transcript.srt").read_text()
    assert translation["segments"][0]["source_text"] == "Исправлено"
    assert translation["segments"][0]["status"] == "source_changed_review_required"
    with pytest.raises(ValueError, match="requires review"):
        service.approve(outcome.capture_id)

    service.update_review(
        outcome.capture_id,
        transcript_segments=[{"text": "Исправлено"}],
        translation_segments=[{"target_text": "Corrected"}],
        speaker_aliases={},
        notes="translation corrected",
    )
    service.approve(outcome.capture_id)
    assert (package / "translation.txt").read_text() == "Corrected\n"


def test_outbox_bundle_fails_verification_after_evidence_tampering(tmp_path):
    workspace = tmp_path / "workspace"
    service = CaptureService(workspace, FakeASR(), resumable=False)
    outcome = service.process(media(tmp_path / "source.wav"))
    service.approve(outcome.capture_id)
    outbox = tmp_path / "outbox"
    assert export_approved(workspace, outbox) == 1
    bundle = outbox / outcome.capture_id
    assert verify_outbox_bundle(bundle)

    (bundle / "transcript.json").write_text("{}")
    assert not verify_outbox_bundle(bundle)
    with pytest.raises(RuntimeError, match="invalid"):
        export_approved(workspace, outbox)
