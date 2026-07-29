from __future__ import annotations

import pytest

from dubbing.transcription import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionOptions,
    TranscriptionResult,
    transcript_to_srt,
    transcript_to_text,
)


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=(
            TranscriptSegment(
                1000,
                2500,
                "Hello world",
                words=(TranscriptWord(1000, 1500, "Hello"),),
                speaker="SPEAKER_00",
                speakers=("SPEAKER_00",),
                speaker_status="attributed",
            ),
        ),
        text="Hello world",
        backend="test",
        model="fixture",
        device="cpu",
        language="en",
        duration_ms=2500,
        confidence_available=False,
        source_sha256="a" * 64,
    )


def test_word_rejects_invalid_timing():
    with pytest.raises(ValueError, match="word timing"):
        TranscriptWord(10, 10, "bad")


def test_segment_rejects_blank_text():
    with pytest.raises(ValueError, match="blank"):
        TranscriptSegment(0, 10, " ")


def test_result_requires_sorted_segments():
    with pytest.raises(ValueError, match="sorted"):
        TranscriptionResult(
            segments=(
                TranscriptSegment(100, 200, "later"),
                TranscriptSegment(0, 50, "first"),
            ),
            text="",
            backend="test",
            model="fixture",
            device="cpu",
            language=None,
            duration_ms=200,
            confidence_available=False,
        )


def test_options_reject_blank_language_and_prompt():
    with pytest.raises(ValueError, match="language"):
        TranscriptionOptions(language=" ")
    with pytest.raises(ValueError, match="initial_prompt"):
        TranscriptionOptions(initial_prompt=" ")


def test_shift_preserves_text_and_moves_word_timing():
    shifted = _result().shifted(3000)
    assert shifted.segments[0].start_ms == 4000
    assert shifted.segments[0].words[0].end_ms == 4500
    assert shifted.text == "Hello world"


def test_json_contract_is_versioned_and_provenanced():
    document = _result().to_dict()
    assert document["schema_version"] == "dubbing.transcription.v1"
    assert document["source_sha256"] == "a" * 64
    assert document["segments"][0]["speaker"] == "SPEAKER_00"


def test_srt_and_text_render_speaker_without_changing_timing():
    result = _result()
    srt = transcript_to_srt(result)
    text = transcript_to_text(result)
    assert "00:00:01,000 --> 00:00:02,500" in srt
    assert "[SPEAKER_00] Hello world" in srt
    assert text == "SPEAKER_00: Hello world\n"
