from __future__ import annotations

import json

from dubbing.evaluation.timestamped_fixture import evaluate_timestamped_fixture
from dubbing.transcription.models import TranscriptSegment, TranscriptWord, TranscriptionResult


def _fixture() -> dict:
    return {
        "source": {"sha256": "a" * 64, "duration_ms": 2000},
        "coverage_gaps": [],
        "selection_policy": {"accuracy_certification_eligible": False},
        "spans": [
            {
                "id": "en",
                "start_ms": 0,
                "end_ms": 900,
                "text": "hello world",
                "language": "en",
            },
            {
                "id": "ko",
                "start_ms": 1000,
                "end_ms": 1900,
                "text": "한국어 시험",
                "language": "ko",
            },
        ],
    }


def _result(*, with_words: bool = True) -> TranscriptionResult:
    segments = (
        TranscriptSegment(
            0,
            900,
            "hello world",
            language="en",
            words=(
                TranscriptWord(0, 400, " hello", 0.9),
                TranscriptWord(450, 850, " world", 0.9),
            )
            if with_words
            else (),
        ),
        TranscriptSegment(
            1000,
            1900,
            "한국어 시험",
            language="ko",
            words=(
                TranscriptWord(1000, 1400, " 한국어", 0.9),
                TranscriptWord(1450, 1850, " 시험", 0.9),
            )
            if with_words
            else (),
        ),
    )
    return TranscriptionResult(
        segments=segments,
        text="hello world 한국어 시험",
        backend="fixture",
        model="fixture",
        device="cpu",
        language=None,
        duration_ms=2000,
        confidence_available=True,
        source_sha256="a" * 64,
    )


def test_timestamped_fixture_scores_exact_windows_without_reference_trimming(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    candidate = tmp_path / "candidate.json"
    fixture.write_text(json.dumps(_fixture(), ensure_ascii=False), encoding="utf-8")
    candidate.write_text(
        json.dumps(_result().to_dict(), ensure_ascii=False), encoding="utf-8"
    )

    report = evaluate_timestamped_fixture(fixture, candidate, tmp_path / "report.json")

    assert report["metrics"]["word_accuracy"] == 1.0
    assert report["per_language"]["en"]["word_accuracy_threshold_passed"]
    assert report["per_language"]["ko"]["word_accuracy_threshold_passed"]
    assert report["word_timestamp_gate_passed"]
    assert not report["measurement_gate_passed"]
    assert (
        report["accuracy_gate"]["measurement_gate"]["status"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert not report["accuracy_certification_eligible"]
    assert not report["production_claim_passed"]
    assert not report["giga_admission_emitted"]


def test_timestamped_fixture_fails_closed_without_word_timestamps(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    candidate = tmp_path / "candidate.json"
    fixture.write_text(json.dumps(_fixture(), ensure_ascii=False), encoding="utf-8")
    candidate.write_text(
        json.dumps(_result(with_words=False).to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    report = evaluate_timestamped_fixture(fixture, candidate, tmp_path / "report.json")

    assert not report["word_timestamp_gate_passed"]
    assert len(report["missing_word_timestamp_segments"]) == 2
    assert not report["measurement_gate_passed"]
