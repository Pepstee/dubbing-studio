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

    def speaker_embeddings(self, audio, diarization):
        return {"SPEAKER_00": (1.0, 0.0)}


def _fake_extract(source: Path, start_ms: int, end_ms: int, output: Path):
    output.write_bytes(f"{start_ms}:{end_ms}".encode())


def test_long_diarization_checkpoints_and_reconciles_global_speaker_labels(tmp_path):
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
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_00",
    ]
    reconciliation = first.provenance["global_speaker_reconciliation"]
    assert reconciliation["scope"] == "recording-global-embedding-cluster"
    assert reconciliation["global_speaker_count"] == 1
    assert reconciliation["embedding_count"] == 3
    assert second.to_dict() == first.to_dict()
    assert (tmp_path / "job" / "result.json").is_file()
    assert (tmp_path / "job" / "progress.json").is_file()
    assert len(tuple((tmp_path / "job" / "embeddings").glob("*.json"))) == 3


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


class _TwoSpeakerBackend(DiarizationBackend):
    @property
    def identity(self):
        return "test:two-speaker-diarization"

    def diarize(self, audio, constraints=None):
        return DiarizationResult(
            turns=(
                SpeakerTurn(0, 400, "SPEAKER_00"),
                SpeakerTurn(500, 900, "SPEAKER_01"),
            ),
            backend="test",
            model="fixture",
            device="cpu",
        )

    def speaker_embeddings(self, audio, diarization):
        return {
            "SPEAKER_00": (1.0, 0.0),
            "SPEAKER_01": (0.0, 1.0),
        }


def test_global_reconciliation_never_merges_two_speakers_from_same_chunk(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"source")
    job = ResumableDiarizationJob(
        _TwoSpeakerBackend(),
        tmp_path / "job",
        chunk_seconds=60,
    )
    with patch(
        "dubbing.diarization.job.media_duration_ms",
        return_value=125_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        result = job.run(audio)

    assert result.speakers == ("SPEAKER_00", "SPEAKER_01")
    assert [turn.speaker for turn in result.turns] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_ambiguous_embedding_stays_a_new_anonymous_speaker(tmp_path):
    job = ResumableDiarizationJob(
        _Backend(),
        tmp_path / "job",
        global_speaker_threshold=0.5,
        global_speaker_margin=0.1,
    )
    turns = [
        SpeakerTurn(0, 1000, "CHUNK_0000_SPEAKER_00"),
        SpeakerTurn(0, 1000, "CHUNK_0000_SPEAKER_01"),
        SpeakerTurn(60_000, 61_000, "CHUNK_0001_SPEAKER_00"),
    ]
    embeddings = {
        "CHUNK_0000_SPEAKER_00": (1.0, 0.0),
        "CHUNK_0000_SPEAKER_01": (0.98, 0.2),
        "CHUNK_0001_SPEAKER_00": (0.995, 0.1),
    }
    global_turns, receipt = job._globalise_speakers(
        turns,
        {
            speaker: job._normalise_embedding(values)
            for speaker, values in embeddings.items()
        },
    )
    assert global_turns[-1].speaker == "SPEAKER_02"
    assert receipt["decisions"][-1]["reason"] == "ambiguous-margin"


def test_missing_embedding_never_guesses_cross_chunk_identity(tmp_path):
    job = ResumableDiarizationJob(_Backend(), tmp_path / "job")
    turns = [
        SpeakerTurn(0, 1000, "CHUNK_0000_A"),
        SpeakerTurn(60_000, 61_000, "CHUNK_0001_A"),
    ]
    global_turns, receipt = job._globalise_speakers(turns, {})
    assert [turn.speaker for turn in global_turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert receipt["unresolved_count"] == 2
    assert all(
        item["reason"] == "missing-embedding" for item in receipt["decisions"]
    )
