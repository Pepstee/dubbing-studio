from __future__ import annotations

import io
import shutil
import wave

import pytest

from dubbing.backends.espeak import EspeakTTSBackend
from dubbing.models import Segment, SRTEntry


def _segment(language: str = "") -> Segment:
    return Segment(
        entry=SRTEntry(index=1, start_ms=0, end_ms=1000, text="Hello"),
        tags=[],
        language=language,
    )


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_espeak_real_backend_produces_wav_when_installed():
    backend = EspeakTTSBackend()
    result = backend.synthesize([_segment()])[0]

    with wave.open(io.BytesIO(result.audio_bytes)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0
    assert result.duration_ms > 0


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_unknown_language_does_not_silently_use_default_voice():
    backend = EspeakTTSBackend()
    with pytest.raises(RuntimeError, match="no installed eSpeak voice"):
        backend.synthesize([_segment("zz-ZZ")])
