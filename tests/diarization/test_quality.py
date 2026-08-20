from __future__ import annotations

from dubbing.diarization.quality import (
    DiarizationQualityReport,
    DiarizationQualityStatus,
    evaluate_diarization_quality,
)
from dubbing.transcription.models import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


TRANSCRIPT_HASH = "a" * 64
DIARIZATION_HASH = "b" * 64


def _result(*segments: TranscriptSegment, turns: list[dict] | None = None):
    return TranscriptionResult(
        segments=segments,
        text=" ".join(segment.text for segment in segments),
        backend="synthetic",
        model="synthetic-asr",
        device="cpu",
        language="en",
        duration_ms=max(segment.end_ms for segment in segments),
        confidence_available=True,
        diarization={
            "backend": "synthetic",
            "model": "synthetic-diarizer",
            "device": "cpu",
            "turns": turns
            if turns is not None
            else [
                {
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "speaker": segment.speaker or "SPEAKER_00",
                }
                for segment in segments
                if segment.speaker_status != "no_speech"
            ],
        },
    )


def _word(start_ms: int, end_ms: int, text: str, speaker: str | None):
    return TranscriptWord(start_ms, end_ms, text, speaker=speaker)


def _evaluate(result: TranscriptionResult) -> DiarizationQualityReport:
    return evaluate_diarization_quality(
        result,
        transcript_sha256=TRANSCRIPT_HASH,
        diarization_sha256=DIARIZATION_HASH,
    )


def test_word_attributed_counts_as_success_and_no_speech_is_excluded():
    result = _result(
        TranscriptSegment(
            0,
            1_000,
            "one",
            words=(_word(100, 800, "one", "SPEAKER_00"),),
            speaker="SPEAKER_00",
            speaker_status="attributed",
        ),
        TranscriptSegment(
            1_000,
            2_000,
            "two three",
            words=(
                _word(1_050, 1_400, "two", "SPEAKER_01"),
                _word(1_500, 1_900, "three", "SPEAKER_01"),
            ),
            speaker="SPEAKER_01",
            speaker_status="word_attributed",
        ),
        TranscriptSegment(
            2_000,
            3_000,
            "noise",
            words=(_word(2_100, 2_800, "noise", None),),
            speaker_status="no_speech",
        ),
    )

    report = _evaluate(result)

    assert report.status is DiarizationQualityStatus.PASS
    assert report.metrics["eligible_speech_segment_count"] == 2
    assert report.metrics["single_speaker_attributed_segment_count"] == 2
    assert report.metrics["excluded_no_speech_segment_count"] == 1
    assert report.metrics["eligible_word_count"] == 3
    assert report.metrics["attributed_word_ratio"] == 1.0


def test_overlap_and_speaker_boundary_are_explicit_review_outcomes():
    result = _result(
        TranscriptSegment(
            0,
            1_000,
            "both",
            speakers=("SPEAKER_00", "SPEAKER_01"),
            speaker_status="overlap",
        ),
        TranscriptSegment(
            1_000,
            2_000,
            "handoff",
            speakers=("SPEAKER_00", "SPEAKER_01"),
            speaker_status="speaker_boundary",
        ),
        turns=[
            {"start_ms": 0, "end_ms": 1_500, "speaker": "SPEAKER_00"},
            {"start_ms": 500, "end_ms": 2_000, "speaker": "SPEAKER_01"},
        ],
    )

    report = _evaluate(result)

    assert report.status is DiarizationQualityStatus.HUMAN_REVIEW_REQUIRED
    assert report.metrics["valid_overlap_segment_count"] == 1
    assert report.metrics["speaker_boundary_segment_count"] == 1
    assert {issue["code"] for issue in report.issues} == {
        "overlapping_speech",
        "speaker_boundaries",
    }


def test_low_attribution_coverage_fails_closed():
    result = _result(
        TranscriptSegment(
            0,
            1_000,
            "known",
            speaker="SPEAKER_00",
            speaker_status="attributed",
        ),
        TranscriptSegment(1_000, 2_000, "unknown", speaker_status="not_requested"),
        turns=[{"start_ms": 0, "end_ms": 1_000, "speaker": "SPEAKER_00"}],
    )

    report = _evaluate(result)

    assert report.status is DiarizationQualityStatus.REPROCESS_REQUIRED
    assert report.metrics["attributed_speech_duration_ratio"] == 0.5
    assert "unresolved_speech_segments" in {
        issue["code"] for issue in report.issues
    }


def test_malformed_diarization_turns_fail_closed_without_crashing():
    result = _result(
        TranscriptSegment(
            0,
            1_000,
            "known",
            speaker="SPEAKER_00",
            speaker_status="attributed",
        ),
        turns=[
            {"start_ms": "bad", "end_ms": 900, "speaker": "SPEAKER_00"},
            {"start_ms": 0, "end_ms": 1_000, "speaker": "SPEAKER_00"},
        ],
    )

    report = _evaluate(result)

    assert report.status is DiarizationQualityStatus.REPROCESS_REQUIRED
    assert report.metrics["malformed_turn_count"] == 1


def test_report_round_trip_preserves_evidence_bindings():
    result = _result(
        TranscriptSegment(
            0,
            1_000,
            "known",
            speaker="SPEAKER_00",
            speaker_status="attributed",
        )
    )
    report = _evaluate(result)

    restored = DiarizationQualityReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.transcript_sha256 == TRANSCRIPT_HASH
    assert restored.diarization_sha256 == DIARIZATION_HASH
