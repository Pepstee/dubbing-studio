import json

import pytest

from dubbing.evaluation.adjudication import adjudicate_transcripts, token_agreement
from dubbing.transcription.models import TranscriptSegment, TranscriptionResult


SOURCE_HASH = "a" * 64


def _write_result(path, text, *, source_hash=SOURCE_HASH, uncertain=False):
    result = TranscriptionResult(
        segments=(TranscriptSegment(0, 10_000, text, uncertain=uncertain),),
        text=text,
        backend=path.stem,
        model=path.stem,
        device="local",
        language="en",
        duration_ms=10_000,
        confidence_available=False,
        source_sha256=source_hash,
    )
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return path


def test_token_agreement_is_symmetric_and_bounded():
    assert token_agreement("one two three", "one too three") == pytest.approx(2 / 3)
    assert token_agreement("one", "one two") == token_agreement("one two", "one")
    assert token_agreement("", "") == 1.0


def test_adjudication_passes_matching_source_bound_transcripts(tmp_path):
    primary = _write_result(tmp_path / "primary.json", "hello world")
    comparator = _write_result(tmp_path / "comparator.json", "hello world")
    report = adjudicate_transcripts(primary, [comparator])
    assert report["verdict"]["status"] == "AUTOMATED_CONSENSUS_PASS"
    assert report["verdict"]["eligible_for_explicit_approval"]
    assert not report["verdict"]["giga_admission_emitted"]
    assert not report["policy"]["human_calibration_required"]


def test_adjudication_fails_closed_on_disagreement(tmp_path):
    primary = _write_result(tmp_path / "primary.json", "hello world")
    comparator = _write_result(tmp_path / "comparator.json", "completely different")
    report = adjudicate_transcripts(primary, [comparator])
    assert report["verdict"]["status"] == "AUTOMATED_CONSENSUS_WITH_UNCERTAINTY"
    assert not report["verdict"]["eligible_for_explicit_approval"]
    assert report["uncertain_window_count"] == 1


def test_adjudication_rejects_cross_source_comparison(tmp_path):
    primary = _write_result(tmp_path / "primary.json", "hello world")
    comparator = _write_result(
        tmp_path / "comparator.json", "hello world", source_hash="b" * 64
    )
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        adjudicate_transcripts(primary, [comparator])


def test_uncertain_primary_cannot_pass(tmp_path):
    primary = _write_result(tmp_path / "primary.json", "hello world", uncertain=True)
    comparator = _write_result(tmp_path / "comparator.json", "hello world")
    report = adjudicate_transcripts(primary, [comparator])
    assert report["verdict"]["status"] == "REJECTED_PRIMARY_QUALITY"
