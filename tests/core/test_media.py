from __future__ import annotations

from unittest.mock import patch

from dubbing.media import ffmpeg_executable


def test_prefers_system_ffmpeg():
    with patch("dubbing.media.shutil.which", return_value="/usr/bin/ffmpeg"):
        assert ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_uses_bundled_ffmpeg_when_system_binary_is_absent():
    class _ImageIO:
        @staticmethod
        def get_ffmpeg_exe():
            return "/venv/imageio_ffmpeg"

    with (
        patch("dubbing.media.shutil.which", return_value=None),
        patch("dubbing.media.importlib.import_module", return_value=_ImageIO()),
    ):
        assert ffmpeg_executable() == "/venv/imageio_ffmpeg"


def test_returns_none_when_no_decoder_is_available():
    with (
        patch("dubbing.media.shutil.which", return_value=None),
        patch("dubbing.media.importlib.import_module", side_effect=ImportError),
    ):
        assert ffmpeg_executable() is None
