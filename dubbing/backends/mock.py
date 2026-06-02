from __future__ import annotations

from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, TTSResult

_MS_PER_CHAR = 60


class MockTTSBackend(TTSBackend):
    def synthesize(self, segments: list[Segment]) -> list[TTSResult]:
        return [
            TTSResult(
                segment=seg,
                audio_bytes=b"",
                duration_ms=len(seg.entry.text) * _MS_PER_CHAR,
            )
            for seg in segments
        ]
