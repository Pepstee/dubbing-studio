from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dubbing.transcription import (
    ResumableTranscriptionJob,
    TranscriptSegment,
    TranscriptionBackend,
    TranscriptionError,
    TranscriptionResult,
)


class _Backend(TranscriptionBackend):
    def __init__(self):
        self.calls = 0

    @property
    def identity(self):
        return "test:fixture"

    def transcribe(self, audio, options=None):
        self.calls += 1
        return TranscriptionResult(
            segments=(TranscriptSegment(0, 900, f"chunk {self.calls}"),),
            text=f"chunk {self.calls}",
            backend="test",
            model="fixture",
            device="cpu",
            language="en",
            duration_ms=900,
            confidence_available=False,
        )


def _fake_extract(source: Path, start_ms: int, end_ms: int, output: Path):
    output.write_bytes(f"{start_ms}:{end_ms}".encode())


def test_job_checkpoints_chunks_and_resumes_without_backend_calls(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"source")
    backend = _Backend()
    job = ResumableTranscriptionJob(backend, tmp_path / "job", chunk_seconds=30)
    with patch(
        "dubbing.transcription.job.media_duration_ms",
        return_value=65_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        first = job.run(audio)
        second = job.run(audio)
    assert backend.calls == 3
    assert [item.start_ms for item in first.segments] == [0, 30_000, 60_000]
    assert second.to_dict() == first.to_dict()
    assert (tmp_path / "job" / "result.json").is_file()
    assert len(list((tmp_path / "job" / "chunks").glob("*.json"))) == 3


def test_checkpoint_rejects_changed_source(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"first")
    backend = _Backend()
    job = ResumableTranscriptionJob(backend, tmp_path / "job", chunk_seconds=30)
    with patch(
        "dubbing.transcription.job.media_duration_ms",
        return_value=30_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        job.run(audio)
        audio.write_bytes(b"changed")
        with pytest.raises(TranscriptionError, match="does not match"):
            job.run(audio)


def test_malformed_chunk_fails_closed(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"source")
    backend = _Backend()
    job = ResumableTranscriptionJob(backend, tmp_path / "job", chunk_seconds=30)
    with patch(
        "dubbing.transcription.job.media_duration_ms",
        return_value=30_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        job.run(audio)
        chunk = next((tmp_path / "job" / "chunks").glob("*.json"))
        chunk.write_text("{", encoding="utf-8")
        with pytest.raises(TranscriptionError, match="malformed"):
            job.run(audio)


def test_manifest_contains_no_absolute_source_path(tmp_path):
    audio = tmp_path / "day.wav"
    audio.write_bytes(b"source")
    backend = _Backend()
    job = ResumableTranscriptionJob(backend, tmp_path / "job", chunk_seconds=30)
    with patch(
        "dubbing.transcription.job.media_duration_ms",
        return_value=30_000,
    ), patch.object(job, "_extract_chunk", side_effect=_fake_extract):
        job.run(audio)
    manifest = json.loads((tmp_path / "job" / "manifest.json").read_text())
    assert manifest["source_name"] == "day.wav"
    assert str(tmp_path) not in json.dumps(manifest)
