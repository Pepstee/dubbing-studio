from __future__ import annotations

from pathlib import Path

from dubbing.aligner import TimedSegment, TimelineAligner
from dubbing.backends.base import TTSBackend
from dubbing.models import Segment, SRTEntry, TTSResult
from dubbing.prosody import parse_prosody
from dubbing.srt_parser import parse_srt, parse_srt_string


class DubbingPipeline:
    def __init__(self, backend: TTSBackend) -> None:
        self._backend = backend
        self._aligner = TimelineAligner()

    def run_full(self, input: str | Path) -> tuple[list[TimedSegment], list[TTSResult]]:
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
        timed = self._aligner.align(segments, durations)
        return timed, results

    def run(self, input: str | Path) -> list[TimedSegment]:
        timed, _ = self.run_full(input)
        return timed
