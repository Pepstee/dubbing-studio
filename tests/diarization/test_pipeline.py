from __future__ import annotations

import io
import wave

from dubbing.backends.base import TTSBackend
from dubbing.diarization import (
    DiarizationBackend,
    DiarizationResult,
    SpeakerConstraints,
    SpeakerTurn,
)
from dubbing.models import TTSResult
from dubbing.pipeline import DubbingPipeline


def _wav(duration_ms: int = 100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * (duration_ms * 8))
    return buffer.getvalue()


class _TTS(TTSBackend):
    def synthesize(self, segments):
        return [
            TTSResult(segment=segment, audio_bytes=_wav(), duration_ms=100)
            for segment in segments
        ]


class _Diarizer(DiarizationBackend):
    def __init__(self):
        self.constraints = None

    def diarize(self, audio, constraints=None):
        self.constraints = constraints
        return DiarizationResult(
            turns=(
                SpeakerTurn(0, 900, "SPEAKER_00"),
                SpeakerTurn(1100, 2000, "SPEAKER_01"),
            ),
            backend="test",
            model="injected",
            device="cpu",
        )


def test_full_pipeline_maps_real_timing_without_mutating_windows(tmp_path):
    srt = """\
1
00:00:00,000 --> 00:00:01,000
First

2
00:00:01,000 --> 00:00:02,000
Second
"""
    audio = tmp_path / "source.wav"
    audio.write_bytes(_wav(2000))
    diarizer = _Diarizer()
    constraints = SpeakerConstraints(num_speakers=2)

    timed, results, diarization, attribution = DubbingPipeline(
        _TTS()
    ).run_full_with_diarization(srt, audio, diarizer, constraints=constraints)

    assert [(item.start_ms, item.end_ms) for item in timed] == [(0, 1000), (1000, 2000)]
    assert len(results) == 2
    assert diarization.speakers == ("SPEAKER_00", "SPEAKER_01")
    assert [item.speaker for item in attribution] == ["SPEAKER_00", "SPEAKER_01"]
    assert diarizer.constraints == constraints
