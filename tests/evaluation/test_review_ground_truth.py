import json

from dubbing.evaluation.review_ground_truth import (
    build_conditioned_fixture,
    build_fixture,
    evaluate_candidate,
)
from dubbing.transcription.models import TranscriptSegment, TranscriptWord, TranscriptionResult


def test_build_fixture_excludes_non_exact_corrections(tmp_path):
    package = tmp_path / "review"
    (package / "clips").mkdir(parents=True)
    for name in ("a.wav", "b.wav", "c.wav"):
        (package / "clips" / name).write_bytes(name.encode())
    manifest = {
        "source": {"path": "/source.wav", "sha256": "a" * 64},
        "items": [
            {
                "id": "a",
                "clip_path": "clips/a.wav",
                "clip_start_ms": 0,
                "clip_end_ms": 1000,
                "start_ms": 100,
                "end_ms": 900,
                "segment_sha256": "1" * 64,
            },
            {
                "id": "b",
                "clip_path": "clips/b.wav",
                "clip_start_ms": 1000,
                "clip_end_ms": 2000,
                "start_ms": 1100,
                "end_ms": 1900,
                "segment_sha256": "2" * 64,
            },
            {
                "id": "c",
                "clip_path": "clips/c.wav",
                "clip_start_ms": 2000,
                "clip_end_ms": 3000,
                "start_ms": 2100,
                "end_ms": 2900,
                "segment_sha256": "3" * 64,
            },
        ],
    }
    decisions = {
        "decisions": {
            "a": {"status": "corrected", "text": "hello", "language": "en"},
            "b": {"status": "corrected", "text": "word...", "language": "en"},
            "c": {"status": "no_speech", "text": "", "language": "unknown"},
        }
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    (package / "decisions.json").write_text(json.dumps(decisions))
    fixture = build_fixture(package, tmp_path / "fixture.json")
    assert fixture["included_count"] == 2
    assert fixture["excluded_count"] == 1
    assert fixture["spans"][1]["no_speech"]
    assert fixture["selection_policy"]["reference_text_is_human_ground_truth"]
    assert not fixture["selection_policy"]["reference_timestamps_are_human_ground_truth"]
    assert not fixture["selection_policy"]["accuracy_certification_eligible"]


def test_evaluate_candidate_uses_word_timestamps(tmp_path):
    fixture = {
        "source": {"sha256": "a" * 64},
        "coverage_gaps": ["ro"],
        "spans": [
            {
                "id": "x",
                "start_ms": 0,
                "end_ms": 1000,
                "segment_start_ms": 0,
                "segment_end_ms": 1000,
                "text": "hello world",
                "language": "en",
                "no_speech": False,
            }
        ],
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    result = TranscriptionResult(
        segments=(
            TranscriptSegment(
                0,
                1000,
                "hello world",
                words=(
                    TranscriptWord(0, 400, " hello", 1.0),
                    TranscriptWord(500, 900, " world", 1.0),
                ),
            ),
        ),
        text="hello world",
        backend="test",
        model="test",
        device="cpu",
        language="en",
        duration_ms=1000,
        confidence_available=True,
        source_sha256="a" * 64,
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(result.to_dict()))
    report = evaluate_candidate(fixture_path, candidate_path, tmp_path / "report.json")
    assert report["metrics"]["word_accuracy"] == 1.0
    assert report["metrics"]["target_passed"]


def test_build_conditioned_fixture_rejects_modified_source_clip(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"changed")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "fixture_id": "fixture",
                "spans": [
                    {
                        "id": "x",
                        "clip_path": str(clip),
                        "clip_sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    try:
        build_conditioned_fixture(
            fixture_path,
            tmp_path / "conditioned.json",
            tmp_path / "clips",
            filter_graph="loudnorm=I=-16",
        )
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("modified source clip was accepted")
