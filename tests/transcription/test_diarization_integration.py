from __future__ import annotations

from dubbing.diarization import DiarizationResult, SpeakerTurn
from dubbing.transcription import (
    AudioUnderstandingPipeline,
    TranscriptSegment,
    TranscriptWord,
    TranscriptionBackend,
    TranscriptionResult,
    attribute_transcript,
)


def _transcript() -> TranscriptionResult:
    return TranscriptionResult(
        segments=(
            TranscriptSegment(
                0,
                2000,
                "One two",
                words=(
                    TranscriptWord(100, 800, "One"),
                    TranscriptWord(1200, 1800, "two"),
                ),
            ),
        ),
        text="One two",
        backend="test",
        model="fixture",
        device="cpu",
        language="en",
        duration_ms=2000,
        confidence_available=False,
    )


def test_word_midpoints_receive_speaker_labels_without_text_mutation():
    diarization = DiarizationResult(
        turns=(
            SpeakerTurn(0, 1000, "SPEAKER_00"),
            SpeakerTurn(1000, 2000, "SPEAKER_01"),
        ),
        backend="test",
        model="fixture",
        device="cpu",
    )
    result = attribute_transcript(_transcript(), diarization)
    assert [word.speaker for word in result.segments[0].words] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert result.segments[0].speaker is None
    assert result.segments[0].speaker_status == "speaker_boundary"
    assert result.text == "One two"


def test_one_speaker_segment_is_word_attributed():
    diarization = DiarizationResult(
        turns=(SpeakerTurn(0, 2000, "SPEAKER_00"),),
        backend="test",
        model="fixture",
        device="cpu",
    )
    result = attribute_transcript(_transcript(), diarization)
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].speaker_status == "word_attributed"


class _ASR(TranscriptionBackend):
    @property
    def identity(self):
        return "test:fixture"

    def transcribe(self, audio, options=None):
        return _transcript()


class _Diarizer:
    def diarize(self, audio, constraints=None):
        return DiarizationResult(
            turns=(SpeakerTurn(0, 2000, "SPEAKER_00"),),
            backend="test",
            model="fixture",
            device="cpu",
        )


def test_audio_understanding_pipeline_keeps_backends_injected():
    result = AudioUnderstandingPipeline(_ASR()).run(
        "not-opened-by-injected-backends.wav",
        diarizer=_Diarizer(),
    )
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.diarization["backend"] == "test"
