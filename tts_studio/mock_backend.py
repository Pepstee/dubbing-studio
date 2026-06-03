from __future__ import annotations

from tts_studio.models import Segment, SpeechResult

_MS_PER_CHAR = 60


class MockTTSBackend:
    """Returns deterministic fake audio without any model or network calls."""

    def synthesize(self, segment: Segment) -> SpeechResult:
        # Deterministic bytes: repeat the segment index byte for the estimated duration
        duration_ms = len(segment.text) * _MS_PER_CHAR
        audio_bytes = bytes([segment.index % 256]) * max(1, duration_ms // 8)
        return SpeechResult(
            segment_index=segment.index,
            audio_bytes=audio_bytes,
            duration_ms=duration_ms,
        )
