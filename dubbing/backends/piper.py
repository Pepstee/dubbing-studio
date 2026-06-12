from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import wave

from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult

_DEFAULT_SAMPLE_RATE = 22050
_CHANNELS = 1
_SAMPLE_WIDTH = 2  # 16-bit PCM


def _raw_to_wav(raw_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return buf.getvalue()


def _wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return int(wf.getnframes() * 1000 / wf.getframerate())


class PiperTTSBackend(TTSBackend):
    """TTS backend that shells out to the piper CLI.

    Voice model path: constructor ``model`` arg or PIPER_MODEL env var.
    The piper binary must be on PATH at synthesis time; absence raises
    RuntimeError (not at import or construction time).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("PIPER_MODEL", "")

    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        if shutil.which("piper") is None:
            raise RuntimeError(
                "piper binary not found on PATH; install piper TTS "
                "(https://github.com/rhasspy/piper) or add it to PATH"
            )
        if not self._model:
            raise RuntimeError(
                "no piper voice model configured; pass model= to PiperTTSBackend "
                "or set the PIPER_MODEL environment variable"
            )
        results: list[TTSResult] = []
        for seg in segments:
            wav = self._synthesize_one(seg.entry.text)
            results.append(
                TTSResult(segment=seg, audio_bytes=wav, duration_ms=_wav_duration_ms(wav))
            )
        return results

    def _synthesize_one(self, text: str) -> bytes:
        proc = subprocess.run(
            ["piper", "--model", self._model, "--output_raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
            check=True,
        )
        sample_rate = _DEFAULT_SAMPLE_RATE
        # piper writes a JSON line per utterance to stderr; extract sample_rate when present.
        for line in proc.stderr.splitlines():
            try:
                info = json.loads(line)
                if isinstance(info, dict) and "audio" in info:
                    sample_rate = int(info["audio"].get("sample_rate", sample_rate))
                    break
            except (ValueError, TypeError):
                continue
        return _raw_to_wav(proc.stdout, sample_rate)
