from pathlib import Path
from unittest.mock import patch

from dubbing.diarization import (
    DiarizationBackend,
    DiarizationResult,
    ResumableDiarizationJob,
    SpeakerTurn,
)
from dubbing.transcription import (
    ResumableTranscriptionJob,
    TranscriptSegment,
    TranscriptionBackend,
    TranscriptionResult,
)

FULL_DAY_MS = 18 * 60 * 60 * 1000


class _ASR(TranscriptionBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "certification:asr"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 1000, f"chunk {self.calls}"),),
            text=f"chunk {self.calls}",
            backend="certification",
            model="fixture",
            device="cpu",
            language="en",
            duration_ms=1000,
            confidence_available=False,
        )


class _Diarizer(DiarizationBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "certification:diarizer"

    def diarize(self, audio, constraints=None):
        self.calls += 1
        return DiarizationResult(
            turns=(SpeakerTurn(0, 1000, "SPEAKER_00"),),
            backend="certification",
            model="fixture",
            device="cpu",
        )


def _extract(source: Path, start_ms: int, end_ms: int, output: Path):
    output.write_bytes(f"{start_ms}:{end_ms}".encode())


def test_eighteen_hour_plan_checkpoints_every_stage_and_resumes(tmp_path):
    source = tmp_path / "full-day.wav"
    source.write_bytes(b"synthetic logical full-day fixture")
    asr = _ASR()
    diarizer = _Diarizer()
    transcription = ResumableTranscriptionJob(
        asr,
        tmp_path / "processing" / "transcription",
        chunk_seconds=1800,
        overlap_seconds=0,
    )
    diarization = ResumableDiarizationJob(
        diarizer,
        tmp_path / "processing" / "diarization",
        chunk_seconds=7200,
    )

    with patch.object(
        transcription,
        "_extract_chunk",
        side_effect=_extract,
    ), patch.object(
        diarization,
        "_extract_chunk",
        side_effect=_extract,
    ):
        transcript_first = transcription.run(
            source,
            source_digest="a" * 64,
            duration_ms=FULL_DAY_MS,
        )
        diarization_first = diarization.run(
            source,
            source_digest="a" * 64,
            duration_ms=FULL_DAY_MS,
        )
        transcript_second = transcription.run(
            source,
            source_digest="a" * 64,
            duration_ms=FULL_DAY_MS,
        )
        diarization_second = diarization.run(
            source,
            source_digest="a" * 64,
            duration_ms=FULL_DAY_MS,
        )

    assert asr.calls == 36
    assert diarizer.calls == 9
    assert len(transcript_first.segments) == 36
    assert len(diarization_first.turns) == 9
    assert transcript_second.to_dict() == transcript_first.to_dict()
    assert diarization_second.to_dict() == diarization_first.to_dict()
