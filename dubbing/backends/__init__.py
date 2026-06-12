from __future__ import annotations

import shutil

from dubbing.backends.piper import PiperTTSBackend
from dubbing.backends.say import SayTTSBackend

__all__ = ["PiperTTSBackend", "SayTTSBackend", "select_backend"]


def select_backend() -> PiperTTSBackend | SayTTSBackend:
    """Return PiperTTSBackend when piper is on PATH, else SayTTSBackend."""
    if shutil.which("piper") is not None:
        return PiperTTSBackend()
    return SayTTSBackend()
