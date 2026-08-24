from __future__ import annotations

import importlib
import shutil


def ffmpeg_executable() -> str | None:
    """Find system FFmpeg or the binary bundled by imageio-ffmpeg."""
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        return None
