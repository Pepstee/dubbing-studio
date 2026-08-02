from pathlib import Path
from unittest.mock import patch

import pytest

from dubbing.diarization import (
    DiarizationBackend,
    DiarizationError,
    DiarizationResult,
    ResumableDiarizationJob,
    SpeakerTurn,
)


class _Backend(DiarizationBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "test:diarization"

    def diarize(self, audio, constraints=None):
        self.calls += 1
        return DiarizationResult(
            turns=(SpeakerTurn(0, 900, "SPEAKER_00"),),
            backend="test",
            model="fixture",
            device="cpu",
        )


def _fake_extract(source: Path, start_ms: int, end_ms: int, output: Path):
    output.write_bytes(f"{start_ms}:{end_ms}".encode())


def test_long_diarization_checkpoints_and_uses_chunk_local_labels(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"source")
    backend = _Backend()
    job = ResumableDiarizationJob(
        backend,
        tmp_path / "job",
        chunk_seconds=60,
    )
    with patch(
        "dubbing.diarization.job.media_duration_ms",
        return_value=125_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        first = job.run(audio)
        second = job.run(audio)

    assert backend.calls == 3
    assert [turn.start_ms for turn in first.turns] == [0, 60_000, 120_000]
    assert [turn.speaker for turn in first.turns] == [
        "CHUNK_0000_SPEAKER_00",
        "CHUNK_0001_SPEAKER_00",
        "CHUNK_0002_SPEAKER_00",
    ]
    assert second.to_dict() == first.to_dict()
    assert (tmp_path / "job" / "result.json").is_file()
    assert (tmp_path / "job" / "progress.json").is_file()


def test_diarization_checkpoint_rejects_changed_source(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"first")
    backend = _Backend()
    job = ResumableDiarizationJob(
        backend,
        tmp_path / "job",
        chunk_seconds=60,
    )
    with patch(
        "dubbing.diarization.job.media_duration_ms",
        return_value=60_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        job.run(audio)
        audio.write_bytes(b"changed")
        with pytest.raises(DiarizationError, match="does not match"):
            job.run(audio)
