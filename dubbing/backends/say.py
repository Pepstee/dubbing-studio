from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult

_SAMPLE_RATE = 22050
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit PCM


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return int(wf.getnframes() * 1000 / wf.getframerate())


def _synthesize_with_say(text: str) -> bytes:
    if shutil.which("say") is None:
        raise RuntimeError("'say' command not found; macOS TTS is unavailable")
    if shutil.which("afconvert") is None:
        raise RuntimeError("'afconvert' command not found; macOS TTS is unavailable")
    with tempfile.TemporaryDirectory() as tmpdir:
        aiff_path = Path(tmpdir) / "out.aiff"
        wav_path = Path(tmpdir) / "out.wav"
        subprocess.run(
            ["say", "-o", str(aiff_path), text],
            check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)],
            check=True, capture_output=True, timeout=30,
        )
        return wav_path.read_bytes()


class SayTTSBackend(TTSBackend):
    """TTS backend using macOS `say` and `afconvert`."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        results: list[TTSResult] = []
        for seg in segments:
            audio = _synthesize_with_say(seg.entry.text)
            results.append(TTSResult(segment=seg, audio_bytes=audio, duration_ms=_wav_duration_ms(audio)))
        return results
