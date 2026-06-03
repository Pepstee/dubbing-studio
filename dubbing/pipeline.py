from __future__ import annotations

from pathlib import Path

from dubbing.aligner import TimedSegment, TimelineAligner
from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, SRTEntry
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import parse_srt, parse_srt_string


class DubbingPipeline:
    def __init__(self, backend: TTSBackend) -> None:
        self._backend = backend
        self._aligner = TimelineAligner()

    def run(self, input: str | Path) -> list[TimedSegment]:
        if isinstance(input, Path):
            entries = parse_srt(input)
        else:
            entries = parse_srt_string(input)
            if not entries:
                entries = [SRTEntry(index=1, start_ms=0, end_ms=0, text=input)]

        segments: list[Segment] = []
        for entry in entries:
            clean_text, tags = parse_prosody(entry.text)
            segments.append(
                Segment(
                    entry=SRTEntry(
                        index=entry.index,
                        start_ms=entry.start_ms,
                        end_ms=entry.end_ms,
                        text=clean_text,
                    ),
                    tags=tags,
                    language="",
                )
            )

        results = self._backend.synthesize(segments)
        durations = [r.duration_ms for r in results]
        return self._aligner.align(segments, durations)
