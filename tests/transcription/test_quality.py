import pytest

from dubbing.transcription.models import (
    DecodeDiagnostics,
    TranscriptSegment,
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


def test_uncertain_span_is_not_approval_ready():
    segment = TranscriptSegment(0, 1000, "unclear name", uncertain=True)
    report = evaluate_transcript_quality(_result(segment.text, (segment,)))
    assert report.status is TranscriptQualityStatus.PASS_WITH_UNCERTAIN_SPANS
    assert not report.approval_allowed
    assert TranscriptQualityReport.from_dict(report.to_dict()) == report


def test_empty_output_is_failed():
    report = evaluate_transcript_quality(_result("", ()))
    assert report.status is TranscriptQualityStatus.FAILED
