import json
from pathlib import Path
from unittest.mock import patch

from dubbing.apps.transcript_review.app import create_review_app
from dubbing.apps.transcript_review.package import build_review_package
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptionResult,
)


def _fixture(tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"private source recording")
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 1000, "certain words", language="en"),
            TranscriptSegment(1200, 2200, "неясные слова", language="en", uncertain=True),
        ),
        text="certain words неясные слова",
        backend="fixture",
        model="fixture",
        device="test",
        language=None,
        duration_ms=3000,
        confidence_available=False,
        source_sha256=source_sha256(source),
    )
    transcript = tmp_path / "result.json"
    transcript.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    package = tmp_path / "review"
    return source, transcript, package


def _write_clip(source, destination, start_ms, end_ms):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"RIFF-test-wave")


def test_build_review_package_is_hash_bound_and_resumable(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.package._extract_clip",
        side_effect=_write_clip,
    ) as extract:
        manifest = build_review_package(source, transcript, package)
        build_review_package(source, transcript, package)

    assert manifest["item_count"] == 1
    assert manifest["uncertain_audio_duration_ms"] == 1000
    assert manifest["source"]["sha256"] == source_sha256(source)
    assert manifest["items"][0]["segment_index"] == 1
    assert manifest["items"][0]["clip_start_ms"] == 0
    assert manifest["items"][0]["clip_end_ms"] == 3000
    assert extract.call_count == 1
    decisions = json.loads((package / "decisions.json").read_text())
    assert next(iter(decisions["items"].values()))["status"] == "pending"


def test_review_app_persists_correction_and_exports_lineage(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.package._extract_clip",
        side_effect=_write_clip,
    ):
        manifest = build_review_package(source, transcript, package)
    item_id = manifest["items"][0]["id"]
    app = create_review_app(package)
    client = app.test_client()
    page = client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]

    assert page.status_code == 200
    assert b"Uncertain spans" in page.data
    assert page.headers["Content-Security-Policy"].startswith("default-src 'none'")
    assert client.get(f"/clips/{item_id}.wav").status_code == 200
    assert client.get("/clips/not-an-item.wav").status_code == 404
    assert client.post(f"/api/decision/{item_id}", json={}).status_code == 403

    decision = client.post(
        f"/api/decision/{item_id}",
        json={
            "status": "corrected",
            "text": "неясные слова исправлены",
            "language": "ru",
            "notes": "heard clearly with context",
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert decision.status_code == 200
    with patch(
        "dubbing.apps.transcript_review.app.detect_silence_intervals",
        return_value=(),
    ):
        exported = client.post("/api/export", headers={"X-CSRF-Token": csrf})

    assert exported.status_code == 200
    payload = exported.get_json()
    assert payload["quality_status"] == "PASS"
    assert payload["giga_admission_emitted"] is False
    reviewed = json.loads((package / "reviewed-result.json").read_text())
    assert reviewed["segments"][1]["text"] == "неясные слова исправлены"
    assert reviewed["segments"][1]["language"] == "ru"
    assert reviewed["segments"][1]["uncertain"] is False
    lineage = reviewed["provenance"]["uncertain_span_review"]
    assert lineage["reviewed_items"] == 1
    assert lineage["giga_admission_authorized"] is False


def test_export_remains_blocked_while_review_is_pending(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.package._extract_clip",
        side_effect=_write_clip,
    ):
        build_review_package(source, transcript, package)
    client = create_review_app(package).test_client()
    client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]

    response = client.post("/api/export", headers={"X-CSRF-Token": csrf})

    assert response.status_code == 409
    assert response.get_json()["error"] == "review_incomplete"


def test_unclear_decision_exports_fail_closed_and_remains_visible_after_reload(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.package._extract_clip",
        side_effect=_write_clip,
    ):
        manifest = build_review_package(source, transcript, package)
    item_id = manifest["items"][0]["id"]
    client = create_review_app(package).test_client()
    client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]

    decision = client.post(
        f"/api/decision/{item_id}",
        json={
            "status": "unclear",
            "text": "неясные слова",
            "language": "ru",
            "notes": "could not resolve from the audio",
        },
        headers={"X-CSRF-Token": csrf},
    )
    with patch(
        "dubbing.apps.transcript_review.app.detect_silence_intervals",
        return_value=(),
    ):
        exported = client.post("/api/export", headers={"X-CSRF-Token": csrf})

    assert decision.status_code == 200
    assert decision.get_json()["progress"]["export_ready"] is True
    assert exported.status_code == 200
    payload = exported.get_json()
    assert payload["quality_status"] == "PASS_WITH_UNCERTAIN_SPANS"
    assert payload["approval_allowed"] is False
    reviewed = json.loads((package / "reviewed-result.json").read_text())
    assert reviewed["segments"][1]["uncertain"] is True
    state = client.get("/api/state").get_json()
    assert state["export"]["quality_status"] == "PASS_WITH_UNCERTAIN_SPANS"
    assert state["export"]["giga_admission_emitted"] is False


def test_no_speech_decision_removes_segment_with_lineage(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.package._extract_clip",
        side_effect=_write_clip,
    ):
        manifest = build_review_package(source, transcript, package)
    item_id = manifest["items"][0]["id"]
    client = create_review_app(package).test_client()
    client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]

    decision = client.post(
        f"/api/decision/{item_id}",
        json={
            "status": "no_speech",
            "text": "placeholder ignored by server",
            "language": "ru",
            "notes": "no transcribable speech in this interval",
        },
        headers={"X-CSRF-Token": csrf},
    )
    with patch(
        "dubbing.apps.transcript_review.app.detect_silence_intervals",
        return_value=(),
    ):
        exported = client.post("/api/export", headers={"X-CSRF-Token": csrf})

    assert decision.status_code == 200
    assert exported.status_code == 200
    assert exported.get_json()["quality_status"] == "PASS"
    reviewed = json.loads((package / "reviewed-result.json").read_text())
    assert [segment["text"] for segment in reviewed["segments"]] == ["certain words"]
    lineage = reviewed["provenance"]["uncertain_span_review"]["lineage"]
    assert lineage[0]["segment_removed_as_no_speech"] is True
    saved = json.loads((package / "decisions.json").read_text())
    assert saved["items"][item_id]["text"] == ""
    assert saved["items"][item_id]["language"] == "unknown"
