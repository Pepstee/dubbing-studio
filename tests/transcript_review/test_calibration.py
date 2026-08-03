import json
from pathlib import Path
from unittest.mock import patch

from dubbing.apps.transcript_review.calibration import build_calibration_package
from dubbing.apps.transcript_review.calibration_app import create_calibration_app
from dubbing.transcription.job import source_sha256
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


def _fixture(tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"private calibration source")
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(0, 10_000, "English words", language="en"),
            TranscriptSegment(45_000, 55_000, "Русские слова", language="ru"),
            TranscriptSegment(90_000, 100_000, "한국어 단어", language="ko"),
            TranscriptSegment(135_000, 145_000, "English и русский", language="mixed"),
        ),
        text="English words Русские слова 한국어 단어 English и русский",
        backend="fixture",
        model="fixture",
        device="test",
        language=None,
        duration_ms=160_000,
        confidence_available=False,
        source_sha256=source_sha256(source),
    )
    transcript = tmp_path / "reviewed.json"
    transcript.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return source, transcript, tmp_path / "calibration"


def _write_clip(source, destination, start_ms, end_ms):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"RIFF-calibration-wave")


def test_calibration_package_is_stratified_hash_bound_and_resumable(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.calibration._extract_clip",
        side_effect=_write_clip,
    ) as extract:
        manifest = build_calibration_package(
            source, transcript, package, clip_duration_ms=30_000
        )
        build_calibration_package(source, transcript, package, clip_duration_ms=30_000)

    assert manifest["item_count"] == 4
    assert manifest["total_audio_duration_ms"] == 120_000
    assert {item["target"] for item in manifest["items"]} == {
        "en",
        "ru",
        "ko",
        "mixed",
    }
    assert manifest["coverage_gaps"] == ["ro"]
    assert manifest["source"]["sha256"] == source_sha256(source)
    assert extract.call_count == 4
    assert len(tuple((package / "clips").glob("*.wav"))) == 4


def test_calibration_review_exports_timestamped_anonymous_ground_truth(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.calibration._extract_clip",
        side_effect=_write_clip,
    ):
        manifest = build_calibration_package(
            source, transcript, package, clip_duration_ms=30_000
        )
    client = create_calibration_app(package).test_client()
    page = client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]

    assert page.status_code == 200
    assert b"Human ground-truth calibration" in page.data
    assert page.headers["Content-Security-Policy"].startswith("default-src 'none'")
    assert client.post("/api/export", headers={"X-CSRF-Token": csrf}).status_code == 409

    for item in manifest["items"]:
        turns = [
            {
                **turn,
                "speaker": "Speaker A",
                "text": f"verified {turn['text']}",
            }
            for turn in item["candidate_turns"]
        ]
        response = client.post(
            f"/api/decision/{item['id']}",
            json={"status": "complete", "turns": turns, "notes": "listened fully"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200

    exported = client.post("/api/export", headers={"X-CSRF-Token": csrf})

    assert exported.status_code == 200
    payload = exported.get_json()
    assert payload["giga_admission_emitted"] is False
    ground_truth = json.loads((package / "ground-truth.json").read_text())
    assert ground_truth["human_ground_truth"] is True
    assert ground_truth["speaker_labels_are_anonymous"] is True
    assert ground_truth["coverage_gaps"] == ["ro"]
    assert all(turn["speaker"] == "Speaker A" for turn in ground_truth["turns"])
    assert all(turn["start_ms"] < turn["end_ms"] for turn in ground_truth["turns"])
    assert ground_truth["giga_admission_authorized"] is False


def test_calibration_rejects_identity_and_out_of_clip_timestamps(tmp_path):
    source, transcript, package = _fixture(tmp_path)
    with patch(
        "dubbing.apps.transcript_review.calibration._extract_clip",
        side_effect=_write_clip,
    ):
        manifest = build_calibration_package(
            source, transcript, package, clip_duration_ms=30_000
        )
    client = create_calibration_app(package).test_client()
    client.get("/")
    with client.session_transaction() as flask_session:
        csrf = flask_session["csrf"]
    item = manifest["items"][0]
    turn = dict(item["candidate_turns"][0])
    turn["speaker"] = "Artiom"
    turn["end_ms"] = item["duration_ms"] + 1

    response = client.post(
        f"/api/decision/{item['id']}",
        json={"status": "complete", "turns": [turn], "notes": ""},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 400
