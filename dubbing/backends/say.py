from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult

_MS_PER_CHAR = 60
_SAMPLE_RATE = 22050
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit PCM


def _make_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _silence_wav(duration_ms: int) -> bytes:
    num_samples = max(1, int(_SAMPLE_RATE * duration_ms / 1000))
    return _make_wav(b"\x00" * num_samples * _SAMPLE_WIDTH * _CHANNELS)


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return int(wf.getnframes() * 1000 / wf.getframerate())


def _synthesize_with_say(text: str) -> bytes:
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
    """TTS backend using macOS `say`; falls back to silence PCM when unavailable."""

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        has_say = shutil.which("say") is not None
        results: list[TTSResult] = []
        for seg in segments:
            text = seg.entry.text
            fallback_ms = max(len(text) * _MS_PER_CHAR, 500)
            if has_say:
                try:
                    audio = _synthesize_with_say(text)
                    duration_ms = _wav_duration_ms(audio)
                except Exception:
                    audio = _silence_wav(fallback_ms)
                    duration_ms = fallback_ms
            else:
                audio = _silence_wav(fallback_ms)
                duration_ms = fallback_ms
            results.append(TTSResult(segment=seg, audio_bytes=audio, duration_ms=duration_ms))
        return results
