from __future__ import annotations

import os
import shutil

import dubbing.backends.espeak as _espeak_mod
import dubbing.backends.piper as _piper_mod
import dubbing.backends.say as _say_mod
from dubbing.backends.base import TTSBackend
from dubbing.backends.espeak import EspeakTTSBackend
from dubbing.backends.piper import PiperTTSBackend
from dubbing.backends.say import SayTTSBackend

__all__ = ["EspeakTTSBackend", "PiperTTSBackend", "SayTTSBackend", "select_backend"]


def select_backend() -> TTSBackend:
    """Select an actually configured local backend without silent substitution."""
    if shutil.which("piper") is not None and os.environ.get("PIPER_MODEL"):
        return _piper_mod.PiperTTSBackend()
    if shutil.which("say") is not None and shutil.which("afconvert") is not None:
        return _say_mod.SayTTSBackend()
    if shutil.which("espeak-ng") is not None:
        return _espeak_mod.EspeakTTSBackend()
    raise RuntimeError(
        "no usable local TTS backend found; configure Piper, install eSpeak NG, "
        "or run on macOS with say/afconvert"
    )
