import pytest

from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)
from dubbing.transcription.quality import (
    TranscriptQualityReport,
    TranscriptQualityStatus,
    evaluate_transcript_quality,
)


def _result(text: str, segments: tuple[TranscriptSegment, ...]):
    return TranscriptionResult(
        segments=segments,
        text=text,
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=max((item.end_ms for item in segments), default=1000),
        confidence_available=False,
    )


@pytest.mark.parametrize(
    "token,count",
    [("Loops", 109), ("tree", 23), ("second", 16), ("Mm-hmm", 20), ("됐다", 33)],
)
def test_known_decoder_loops_fail_closed(token, count):
    text = " ".join([token] * count)
    report = evaluate_transcript_quality(
        _result(text, (TranscriptSegment(0, 10_000, text),))
    )
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert not report.approval_allowed
    assert "pathological_repetition" in {item.code for item in report.issues}


def test_fallback_exhaustion_requires_reprocessing():
    segment = TranscriptSegment(
        0,
        1000,
        "ordinary speech",
        diagnostics=DecodeDiagnostics(
            temperature=1.0,
            fallback_exhausted=True,
            fallback_history=({"temperature": 0.0}, {"temperature": 1.0}),
        ),
    )
    report = evaluate_transcript_quality(_result(segment.text, (segment,)))
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED


@pytest.mark.parametrize(
    "text",
    [
        "Субтитры сделал DimaTorzok Продолжение следует",
        "다음 영상에서 만나요. 시청해주셔서 감사합니다.",
        "Nu uitați să vă abonați. Mulțumim pentru vizionare.",
    ],
)
def test_stock_whisper_boilerplate_with_no_speech_evidence_fails_closed(text):
    segment = TranscriptSegment(
        0,
        20_000,
        text,
        diagnostics=DecodeDiagnostics(
            avg_log_probability=-0.35,
            no_speech_probability=0.706,
            temperature=0.0,
        ),
    )

    report = evaluate_transcript_quality(_result(text, (segment,)))

    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    issue = next(
        item
        for item in report.issues
        if item.code == "stock_hallucination_under_no_speech"
    )
    assert issue.start_ms == 0
    assert issue.end_ms == 20_000
    assert issue.evidence["no_speech_probability"] == 0.706


def test_legitimate_thanks_for_watching_without_no_speech_evidence_is_not_rejected():
    text = "Thanks for watching the recording with me."
    segment = TranscriptSegment(
        0,
        2000,
        text,
        diagnostics=DecodeDiagnostics(
            avg_log_probability=-0.1,
            no_speech_probability=0.02,
            temperature=0.0,
        ),
    )

    report = evaluate_transcript_quality(_result(text, (segment,)))

    assert "stock_hallucination_under_no_speech" not in {
        item.code for item in report.issues
    }


def test_uncertain_span_is_not_approval_ready():
    segment = TranscriptSegment(0, 1000, "unclear name", uncertain=True)
    report = evaluate_transcript_quality(_result(segment.text, (segment,)))
    assert report.status is TranscriptQualityStatus.PASS_WITH_UNCERTAIN_SPANS
    assert not report.approval_allowed
    assert TranscriptQualityReport.from_dict(report.to_dict()) == report


def test_empty_output_is_failed():
    report = evaluate_transcript_quality(_result("", ()))
    assert report.status is TranscriptQualityStatus.FAILED


def test_real_v5_segment_beyond_65407ms_duration_fails_closed():
    transcript = TranscriptionResult(
        segments=(TranscriptSegment(74_100, 74_120, "late"),),
        text="late",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=65_407,
        confidence_available=False,
    )

    report = evaluate_transcript_quality(
        transcript,
        expected_duration_ms=65_407,
    )

    issue = next(item for item in report.issues if item.code == "timestamp_out_of_bounds")
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert (issue.start_ms, issue.end_ms) == (74_100, 74_120)
    assert issue.evidence == {
        "kind": "segment",
        "segment_index": 0,
        "observed_start_ms": 74_100,
        "observed_end_ms": 74_120,
        "violated_duration_bounds": {
            "expected_duration_ms": 65_407,
            "transcript_duration_ms": 65_407,
        },
    }
    assert report.metrics["timestamp_upper_bound_ms"] == 65_407
    assert report.metrics["segment_timestamp_out_of_bounds_count"] == 1
    assert report.metrics["timestamp_out_of_bounds_count"] == 1


def test_expected_duration_is_enforced_when_transcript_duration_is_longer():
    transcript = TranscriptionResult(
        segments=(TranscriptSegment(60_000, 70_000, "late"),),
        text="late",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=80_000,
        confidence_available=False,
    )

    report = evaluate_transcript_quality(transcript, expected_duration_ms=65_407)
    issue = next(item for item in report.issues if item.code == "timestamp_out_of_bounds")

    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert issue.evidence["violated_duration_bounds"] == {
        "expected_duration_ms": 65_407
    }


def test_transcript_duration_is_enforced_when_expected_duration_is_longer():
    transcript = TranscriptionResult(
        segments=(TranscriptSegment(1_000, 1_500, "late"),),
        text="late",
        backend="fixture",
        model="fixture",
        device="test",
        language="en",
        duration_ms=1_200,
        confidence_available=False,
    )

    report = evaluate_transcript_quality(transcript, expected_duration_ms=2_000)
    issue = next(item for item in report.issues if item.code == "timestamp_out_of_bounds")

    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert issue.evidence["violated_duration_bounds"] == {
        "transcript_duration_ms": 1_200
    }
    assert report.metrics["timestamp_upper_bound_ms"] == 1_200


def test_word_beyond_duration_and_parent_segment_fails_closed_once():
    word = TranscriptWord(900, 1_200, "late")
    segment = TranscriptSegment(0, 1_000, "late", words=(word,))

    report = evaluate_transcript_quality(
        _result("late", (segment,)),
        expected_duration_ms=1_000,
    )
    issue = next(item for item in report.issues if item.code == "timestamp_out_of_bounds")

    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert issue.evidence["kind"] == "word"
    assert issue.evidence["word_index"] == 0
    assert issue.evidence["outside_parent_segment"] is True
    assert issue.evidence["straddles_parent_segment_boundary"] is True
    assert issue.evidence["violated_duration_bounds"] == {
        "expected_duration_ms": 1_000,
        "transcript_duration_ms": 1_000,
    }
    assert report.metrics["word_timestamp_out_of_bounds_count"] == 1
    assert report.metrics["word_recording_bounds_violation_count"] == 1
    assert report.metrics["word_segment_containment_violation_count"] == 1


def test_word_boundary_straddle_owned_by_midpoint_is_not_rejected():
    word = TranscriptWord(900, 1_100, "boundary")
    segment = TranscriptSegment(1_000, 2_000, "boundary", words=(word,))

    report = evaluate_transcript_quality(
        _result("boundary", (segment,)),
        expected_duration_ms=2_000,
    )
    assert report.status is TranscriptQualityStatus.PASS
    assert "timestamp_out_of_bounds" not in {item.code for item in report.issues}
    assert report.metrics["word_recording_bounds_violation_count"] == 0
    assert report.metrics["word_segment_containment_violation_count"] == 0
    assert report.metrics["word_parent_boundary_straddle_count"] == 1


def test_word_midpoint_outside_parent_segment_fails_closed():
    word = TranscriptWord(800, 1_000, "misassigned")
    segment = TranscriptSegment(1_000, 2_000, "misassigned", words=(word,))

    report = evaluate_transcript_quality(
        _result("misassigned", (segment,)),
        expected_duration_ms=2_000,
    )
    issue = next(item for item in report.issues if item.code == "timestamp_out_of_bounds")

    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert issue.evidence["violated_duration_bounds"] == {}
    assert issue.evidence["outside_parent_segment"] is True
    assert issue.evidence["word_midpoint_ms"] == 900
    assert issue.evidence["straddles_parent_segment_boundary"] is True
    assert report.metrics["word_segment_containment_violation_count"] == 1
    assert report.metrics["word_parent_boundary_straddle_count"] == 1


def test_scattered_legitimate_duplicate_segments_do_not_form_a_loop():
    segments = tuple(
        TranscriptSegment(index * 1000, index * 1000 + 900, text)
        for index, text in enumerate(("yes", "yes", "next", "okay", "okay", "done"))
    )
    report = evaluate_transcript_quality(_result(" ".join(item.text for item in segments), segments))
    assert "pathological_repetition" not in {item.code for item in report.issues}


def test_six_conversational_yes_responses_are_not_a_decoder_loop():
    segments = tuple(
        TranscriptSegment(index * 1000, index * 1000 + 500, "yes")
        for index in range(6)
    )

    report = evaluate_transcript_quality(
        _result(" ".join(item.text for item in segments), segments)
    )

    assert "pathological_repetition" not in {item.code for item in report.issues}


def test_seven_token_decoder_loop_is_detected_and_timestamp_localized():
    phrase = "I don't know what to do"
    loop = " ".join([phrase] * 12)
    segments = (
        TranscriptSegment(0, 1000, "ordinary beginning"),
        TranscriptSegment(10_000, 20_000, loop),
        TranscriptSegment(21_000, 22_000, "ordinary ending"),
    )

    report = evaluate_transcript_quality(
        _result(" ".join(item.text for item in segments), segments)
    )

    issue = next(item for item in report.issues if item.code == "pathological_repetition")
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert (issue.start_ms, issue.end_ms) == (10_000, 20_000)
    assert issue.evidence["phrase"] == "i don t know what to do"
    assert issue.evidence["phrase_width"] == 7
    assert issue.evidence["repeat_count"] == 12


def test_long_phrase_repeated_only_three_times_remains_eligible():
    phrase = "I don't know what to do"
    text = " ".join([phrase] * 3)
    report = evaluate_transcript_quality(
        _result(text, (TranscriptSegment(0, 10_000, text),))
    )

    assert "pathological_repetition" not in {item.code for item in report.issues}


def test_decoder_loop_split_across_segments_keeps_full_retry_interval():
    phrase = "we need to leave this place now"
    segments = tuple(
        TranscriptSegment(index * 1000, index * 1000 + 900, phrase)
        for index in range(5)
    )

    report = evaluate_transcript_quality(
        _result(" ".join(item.text for item in segments), segments)
    )

    issue = next(item for item in report.issues if item.code == "pathological_repetition")
    assert (issue.start_ms, issue.end_ms) == (0, 4900)
    assert issue.evidence["repeat_count"] == 5


def test_rotated_phrase_candidate_is_deduplicated_from_the_same_loop():
    phrase = "I don't know what to do"
    loop = " ".join([phrase] * 28)
    segments = (
        TranscriptSegment(1000, 10_000, loop),
        TranscriptSegment(10_000, 11_000, "I am ready now"),
    )

    report = evaluate_transcript_quality(
        _result(f"{loop} I am ready now", segments)
    )
    findings = report.metrics["repetition_findings"]

    assert len(findings) == 1
    assert findings[0]["phrase"] == "i don t know what to do"
    assert (findings[0]["start_ms"], findings[0]["end_ms"]) == (1000, 10_000)


def test_failed_span_placeholder_requires_reprocessing():
    segment = TranscriptSegment(
        0, 1000, "[UNCERTAIN: LOCAL TRANSCRIPTION FAILED]", uncertain=True
    )
    report = evaluate_transcript_quality(_result(segment.text, (segment,)))
    assert report.status is TranscriptQualityStatus.REPROCESS_REQUIRED
    assert "span_transcription_failed" in {item.code for item in report.issues}


def test_large_gap_is_explained_only_when_vad_silence_covers_eighty_percent():
    segments = (
        TranscriptSegment(0, 1000, "before"),
        TranscriptSegment(81_000, 82_000, "after"),
    )
    transcript = _result("before after", segments)
    unexplained = evaluate_transcript_quality(
        transcript,
        expected_duration_ms=82_000,
        known_silence_intervals=((10_000, 60_000),),
    )
    explained = evaluate_transcript_quality(
        transcript,
        expected_duration_ms=82_000,
        known_silence_intervals=((1000, 70_000),),
    )
    assert unexplained.status is TranscriptQualityStatus.HUMAN_REVIEW_REQUIRED
    assert explained.status is TranscriptQualityStatus.PASS
    assert explained.metrics["explained_silence_gap_count"] == 1
    assert explained.metrics["large_gap_count"] == 0
