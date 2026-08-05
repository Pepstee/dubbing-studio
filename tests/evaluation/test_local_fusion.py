from dubbing.evaluation.local_fusion import fuse_transcripts
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


def _result(segments, backend):
    return TranscriptionResult(
        segments=tuple(segments),
        text=" ".join(segment.text for segment in segments),
        backend=backend,
        model=backend,
        device="local",
        language="en",
        duration_ms=20_000,
        confidence_available=False,
        source_sha256="a" * 64,
    )


def test_fusion_replaces_only_pathological_chain_and_marks_alternate_uncertain():
    primary = _result(
        [
            TranscriptSegment(0, 1000, "good"),
            TranscriptSegment(2000, 3000, "loop"),
            TranscriptSegment(3000, 4000, "loop"),
            TranscriptSegment(4000, 5000, "loop"),
            TranscriptSegment(5000, 6000, "loop"),
            TranscriptSegment(8000, 9000, "also good"),
        ],
        "primary",
    )
    alternate = _result(
        [TranscriptSegment(2500, 5500, "replacement")], "alternate"
    )
    result, receipt = fuse_transcripts(primary, alternate, padding_ms=0)
    assert [segment.text for segment in result.segments] == [
        "good",
        "replacement",
        "also good",
    ]
    assert result.segments[1].uncertain
    assert receipt["pathological_primary_segment_count"] == 4


def test_fusion_rejects_cross_source_inputs():
    primary = _result([TranscriptSegment(0, 1000, "good")], "primary")
    alternate = _result([TranscriptSegment(0, 1000, "good")], "alternate")
    alternate = TranscriptionResult(
        **{**alternate.__dict__, "source_sha256": "b" * 64}
    )
    try:
        fuse_transcripts(primary, alternate)
    except ValueError as exc:
        assert "same source" in str(exc)
    else:
        raise AssertionError("cross-source fusion must fail")
